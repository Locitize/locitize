"""Big-model placement: serve a mixture-of-experts model that is larger than the card.

What this adds
--------------
llama.cpp's fit mode (backends.py) handles a dense model that does not quite fit by
keeping whole layers on the CPU. For a mixture-of-experts (MoE) model that is much
bigger than the card - gpt-oss-120b is 60 GB against a 16 GB card - whole-layer
offload is the wrong cut: every token then runs most of the network on the CPU.
The right cut keeps attention, the shared weights and the KV cache on the GPU and
moves only the EXPERT weights of the first N blocks to the CPU (`--n-cpu-moe N`);
each token touches a few experts, so most of the CPU-side weight is never read.

When those CPU-side experts exceed system RAM as well, llama.cpp's memory-mapped
weights page in from the drive on demand: the model streams from the SSD. That
mode needs its own tuning (large prompt batches worked on the CPU in place, a
fixed thread count), and a server build that knows `GGML_EXPERT_STREAM` reads the
experts directly into its own RAM cache instead of through the Windows file cache.

The planner here reads the GGUF tensor table (sizes per block, which blocks carry
experts, the token-embedding table that stays in host memory), the card's real free
memory and the machine's RAM, and decides:

    gpu          everything fits: no change, the engine's own placement applies
    experts-cpu  experts of the first N blocks in RAM; leftover VRAM caches hot experts
    stream       as above, but the experts do not fit RAM either: SSD streaming tuning
    fit          the planner has nothing better than the engine's fit mode (a dense
                 model, or a MoE whose shared part alone does not fit at this context)

Only `experts-cpu` and `stream` change the argv; the other two leave it byte-identical
to a launch without this module. A models.yaml row opts out with `placement: fit`.

Measured on the reference machine (RTX 5070 Ti 16 GB, 32 GB RAM, PCIe 5 NVMe) with the
same recipe in another local tool: gpt-oss-120b (60 GB) loads in about 14 s and
generates 10 to 30 tok/s depending on how warm the expert cache is; the engine's fit
mode alone put it in shared memory at about 5 tok/s or refused to load.

Pure except read_layout (file header reads), engine_flags (one `--help` run, cached)
and the RAM/CPU readings, all injectable for tests. ASCII only.
"""

from __future__ import annotations

import functools
import glob
import os
import re
import struct
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, BinaryIO, Callable

from gguf_meta import (
    _SCALARS,
    _T_ARRAY,
    _T_STRING,
    GGUF_MAGIC,
    SUPPORTED_VERSIONS,
    GgufError,
    _read_exact,
    _read_string,
    _read_value,
    _skip_value,
    _unpack,
)

MB = 1024 * 1024

# ggml tensor types: id -> (elements per block, bytes per block). From ggml's type
# traits; a type missing here is sized as f16 (2 bytes per element), which only
# ever over-estimates and so never over-fills the card.
GGML_TYPES: dict[int, tuple[int, int]] = {
    0: (1, 4),       # F32
    1: (1, 2),       # F16
    2: (32, 18),     # Q4_0
    3: (32, 20),     # Q4_1
    6: (32, 22),     # Q5_0
    7: (32, 24),     # Q5_1
    8: (32, 34),     # Q8_0
    9: (32, 36),     # Q8_1
    10: (256, 84),   # Q2_K
    11: (256, 110),  # Q3_K
    12: (256, 144),  # Q4_K
    13: (256, 176),  # Q5_K
    14: (256, 210),  # Q6_K
    15: (256, 292),  # Q8_K
    16: (256, 66),   # IQ2_XXS
    17: (256, 74),   # IQ2_XS
    18: (256, 98),   # IQ3_XXS
    19: (256, 50),   # IQ1_S
    20: (32, 18),    # IQ4_NL
    21: (256, 110),  # IQ3_S
    22: (256, 82),   # IQ2_S
    23: (256, 136),  # IQ4_XS
    24: (1, 1),      # I8
    25: (1, 2),      # I16
    26: (1, 4),      # I32
    27: (1, 8),      # I64
    28: (1, 8),      # F64
    29: (256, 56),   # IQ1_M
    30: (1, 2),      # BF16
    34: (256, 54),   # TQ1_0
    35: (256, 66),   # TQ2_0
    39: (32, 17),    # MXFP4
}

# Expert weight tensors live in blk.<n>.ffn_{gate,up,down,gate_up}_exps.*
EXPERT_TENSOR = re.compile(r"^blk\.(\d+)\.ffn_(?:gate|up|down|gate_up)_exps\b")
BLOCK_TENSOR = re.compile(r"^blk\.(\d+)\.")
TOKEN_EMBD = "token_embd.weight"
SHARD = re.compile(r"-(\d{5})-of-(\d{5})\.gguf$", re.IGNORECASE)

# CUDA context plus compute buffers llama.cpp needs beyond weights and KV cache.
GPU_OVERHEAD_MB = 700
# The same for a streamed model with the smaller micro-batch below (measured 199 MB compute).
STREAM_OVERHEAD_MB = 450
# Prompt batch and micro-batch when weights live on the CPU: llama.cpp's defaults were
# slower there, and the smaller micro-batch halves the GPU compute buffer.
CPU_BATCH = 2048
CPU_UBATCH = 1024
# RAM kept for Windows and other apps when deciding whether the experts fit in memory.
RAM_RESERVE_MB = 8192
# VRAM kept free beside the plan, for the driver and display.
VRAM_MARGIN_MB = 256
# A hot-expert cache smaller than this is not worth its bookkeeping.
MIN_CACHE_MB = 768

# Flags the planner may use only when the installed llama-server knows them.
FLAG_CPU_MOE = "--n-cpu-moe"
FLAG_MOE_CACHE = "--moe-cache-mib"
FLAG_NO_OP_OFFLOAD = "--no-op-offload"
FLAG_LOAD_MODE = "--load-mode"
PLANNER_FLAGS = (FLAG_CPU_MOE, FLAG_MOE_CACHE, FLAG_NO_OP_OFFLOAD, FLAG_LOAD_MODE)

# models.yaml `placement:` values.
PLACEMENT_AUTO = "auto"   # also the meaning of an absent field
PLACEMENT_FIT = "fit"     # never plan; the engine's fit mode only
PLACEMENTS = ("", PLACEMENT_AUTO, PLACEMENT_FIT)


# --------------------------------------------------------------------------- #
# The model's layout, from its GGUF tensor table
# --------------------------------------------------------------------------- #


@dataclass
class ModelLayout:
    """What the planner needs to know about one GGUF model (all shards)."""

    path: str
    architecture: str
    file_mb: float                      # every tensor, all shards
    block_count: int
    expert_count: int
    context_length: int | None
    kv_bytes_per_token: float           # at 2 bytes per value (f16); scaled by cache type
    token_embd_mb: float                # stays in host memory, so costs no VRAM
    expert_mb_per_block: dict[int, float] = field(default_factory=dict)
    kv_blocks: int = 0                  # blocks with attention (a KV cache); 0 = every block
    ssm_blocks: int = 0                 # blocks with recurrent (Mamba) state instead
    ssm_state_mb: float = 0.0           # that state per block and slot, fixed whatever the context

    @property
    def is_moe(self) -> bool:
        return bool(self.expert_mb_per_block)

    @property
    def expert_mb(self) -> float:
        return sum(self.expert_mb_per_block.values())

    @property
    def non_expert_mb(self) -> float:
        return self.file_mb - self.expert_mb

    def kv_mb(self, context: int, bytes_per_value: float = 2.0) -> float:
        return self.kv_bytes_per_token * (bytes_per_value / 2.0) * context / MB

    def recurrent_mb(self, slots: int = 1) -> float:
        """A hybrid model's Mamba state for `slots` slots: fixed, whatever the context."""
        return self.ssm_blocks * self.ssm_state_mb * slots


def _tensor_bytes(dims: list[int], ggml_type: int) -> int:
    elements = 1
    for d in dims:
        elements *= int(d)
    block, size = GGML_TYPES.get(ggml_type, (1, 2))
    return (elements // block) * size


def _read_part(path: Path) -> tuple[dict[str, Any], list[tuple[str, int]]]:
    """(scalar metadata by key, [(tensor name, bytes)]) for one GGUF file."""
    meta: dict[str, Any] = {}
    tensors: list[tuple[str, int]] = []
    with open(path, "rb", buffering=1024 * 1024) as handle:
        if _read_exact(handle, 4, path, "the magic bytes") != GGUF_MAGIC:
            raise GgufError(f"The file at {path} is not a GGUF model file.")
        version = _unpack("<I", _read_exact(handle, 4, path, "the format version"))
        if version not in SUPPORTED_VERSIONS:
            raise GgufError(f"The file at {path} uses GGUF version {version}, which locitize does not read.")
        tensor_count = _unpack("<Q", _read_exact(handle, 8, path, "the tensor count"))
        kv_count = _unpack("<Q", _read_exact(handle, 8, path, "the metadata count"))
        for _ in range(kv_count):
            key = _read_string(handle, path)
            value_type = _unpack("<I", _read_exact(handle, 4, path, f"the type of '{key}'"))
            if value_type in _SCALARS or value_type == _T_STRING and key.startswith("general."):
                meta[key] = _read_value(handle, value_type, path)
            else:
                _skip_value(handle, value_type, path)   # tokenizer arrays and the like
        for _ in range(tensor_count):
            name = _read_string(handle, path)
            n_dims = _unpack("<I", _read_exact(handle, 4, path, f"the rank of '{name}'"))
            if n_dims > 8:
                raise GgufError(f"The GGUF tensor table at {path} is not plausible (rank {n_dims}).")
            dims = [_unpack("<Q", _read_exact(handle, 8, path, "a dimension")) for _ in range(n_dims)]
            ggml_type = _unpack("<I", _read_exact(handle, 4, path, f"the type of '{name}'"))
            _read_exact(handle, 8, path, "the data offset")
            tensors.append((name, _tensor_bytes(dims, ggml_type)))
    return meta, tensors


def shard_paths(path: str) -> list[str]:
    """Every shard of a split GGUF (`-00001-of-00003.gguf`), or just the file."""
    match = SHARD.search(path)
    if not match:
        return [path]
    pattern = path[: match.start()] + "-*-of-" + match.group(2) + ".gguf"
    found = sorted(glob.glob(pattern))
    return found or [path]


def read_layout(path: str) -> ModelLayout:
    """Read the model's tensor table and the architecture facts the planner needs.

    Milliseconds per shard: only the header and the tensor table are read, never
    the weights. Raises GgufError (or OSError) when the file cannot be read."""
    meta: dict[str, Any] = {}
    tensors: list[tuple[str, int]] = []
    for part in shard_paths(path):
        part_meta, part_tensors = _read_part(Path(part))
        meta = {**part_meta, **meta} if meta else part_meta
        tensors.extend(part_tensors)
    arch = str(meta.get("general.architecture", "") or "")

    def arch_int(key: str, default: int = 0) -> int:
        value = meta.get(f"{arch}.{key}")
        return int(value) if isinstance(value, int) and not isinstance(value, bool) else default

    blocks = arch_int("block_count")
    heads = max(1, arch_int("attention.head_count", 1))
    kv_heads = arch_int("attention.head_count_kv", heads) or heads
    embedding = arch_int("embedding_length")
    key_len = arch_int("attention.key_length", embedding // heads if heads else 0)
    value_len = arch_int("attention.value_length", key_len)
    expert_count = arch_int("expert_count")

    experts: dict[int, float] = {}
    attention_blocks: set[int] = set()
    ssm_blocks: set[int] = set()
    total = 0
    embd = 0
    for name, size in tensors:
        total += size
        if name == TOKEN_EMBD:
            embd = size
        match = EXPERT_TENSOR.match(name)
        if match:
            block = int(match.group(1))
            experts[block] = experts.get(block, 0.0) + size / MB
        block_match = BLOCK_TENSOR.match(name)
        if block_match:
            rest = name[block_match.end():]
            if rest.startswith(("attn_k.", "attn_kv.", "attn_qkv.", "attn_k_")):
                attention_blocks.add(int(block_match.group(1)))
            elif rest.startswith("ssm_"):
                ssm_blocks.add(int(block_match.group(1)))
    # KV cache per token at f16: every attention block keeps K and V for each KV head.
    # A hybrid model (Nemotron-H) has Mamba blocks with a fixed recurrent state instead.
    kv_blocks = len(attention_blocks) if attention_blocks and ssm_blocks else blocks
    kv_per_token = float(kv_blocks * kv_heads * (key_len + value_len) * 2)
    ssm_state_mb = 0.0
    if ssm_blocks:
        state, inner = arch_int("ssm.state_size"), arch_int("ssm.inner_size")
        conv, groups = arch_int("ssm.conv_kernel", 4), max(1, arch_int("ssm.group_count", 1))
        # f32 state matrix plus the convolution window, per block and slot.
        ssm_state_mb = (state * inner * 4 + (conv - 1) * (inner + 2 * groups * state) * 4) / MB
    return ModelLayout(
        path=path,
        architecture=arch,
        file_mb=total / MB,
        block_count=blocks,
        expert_count=expert_count,
        context_length=arch_int("context_length") or None,
        kv_bytes_per_token=kv_per_token,
        token_embd_mb=embd / MB,
        expert_mb_per_block=experts,
        kv_blocks=kv_blocks,
        ssm_blocks=len(ssm_blocks),
        ssm_state_mb=ssm_state_mb,
    )


# --------------------------------------------------------------------------- #
# The plan
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Placement:
    """How a model is laid over the GPU, RAM and (for `stream`) the drive."""

    mode: str                   # gpu | experts-cpu | stream | fit
    gpu_layers: int = 0         # --n-gpu-layers: every block plus the output layer (an exact count, not fit)
    cpu_moe_blocks: int = 0     # --n-cpu-moe: experts of blocks 0..N-1 live on the CPU
    moe_cache_mib: int = 0      # --moe-cache-mib: leftover VRAM caching hot CPU-side experts
    gpu_mb: int = 0
    cpu_mb: int = 0
    slots: int = 1
    context: int = 0
    reason: str = ""

    @property
    def changes_argv(self) -> bool:
        return self.mode in ("experts-cpu", "stream")

    def args(self, flags: set[str], thread_args: list[str] | None = None) -> list[str]:
        """The flags to append after the engine's argv core. `flags` are the ones the
        installed binary understands (engine_flags); anything it lacks is left out."""
        if not self.changes_argv:
            return []
        out = [FLAG_CPU_MOE, str(self.cpu_moe_blocks)]
        if self.moe_cache_mib and FLAG_MOE_CACHE in flags:
            out += [FLAG_MOE_CACHE, str(self.moe_cache_mib)]
        if self.mode == "stream" and FLAG_NO_OP_OFFLOAD in flags:
            # Copying SSD-backed experts to the GPU for every prompt batch is what made
            # prompts crawl; the CPU works on them in place, in large batches.
            out.append(FLAG_NO_OP_OFFLOAD)
        if self.mode == "stream" and FLAG_LOAD_MODE in flags:
            # Streaming is the memory map: a setting or build default that loads whole
            # models into RAM (--load-mode none) would try to read all 60 GB. Later
            # flags win in llama.cpp, so a row's own --load-mode in server_args still can.
            out += [FLAG_LOAD_MODE, "mmap"]
        out += ["-b", str(CPU_BATCH), "-ub", str(CPU_UBATCH)]
        out += thread_args if thread_args is not None else default_thread_args()
        return out

    def env(self) -> dict[str, str]:
        """Process environment for the server. A build that streams experts with its own
        direct reads (instead of the Windows file cache) switches that on here; a stock
        build ignores the variable."""
        return {"GGML_EXPERT_STREAM": "1"} if self.mode == "stream" else {}

    def describe(self) -> str:
        if self.mode == "gpu":
            return "placement: the whole model fits on the GPU"
        if self.mode == "fit":
            return f"placement: the engine's fit mode ({self.reason})"
        where = "streamed from the drive" if self.mode == "stream" else "in RAM"
        cache = f", {self.moe_cache_mib} MiB of VRAM caching hot experts" if self.moe_cache_mib else ""
        return (f"placement: experts of the first {self.cpu_moe_blocks} blocks {where} "
                f"({self.cpu_mb} MB), {self.gpu_mb} MB on the GPU at {self.context} tokens{cache}")


def plan(layout: ModelLayout, vram_free_mb: int, ram_total_mb: int, context: int,
         slots: int = 1, kv_bytes_per_value: float = 2.0, flags: set[str] | None = None) -> Placement:
    """Fit `layout` into `vram_free_mb` of VRAM with `ram_total_mb` of RAM behind it.

    `context` is the total context llama-server gets (its --ctx-size), shared by
    `slots`; `kv_bytes_per_value` is 1 for a q8_0 KV cache, 2 for f16."""
    flags = flags if flags is not None else set(PLANNER_FLAGS)
    allowance = vram_free_mb - VRAM_MARGIN_MB
    # The cache a hybrid model keeps for its Mamba blocks is fixed, so it counts as part of the KV.
    kv = layout.kv_mb(context, kv_bytes_per_value) + layout.recurrent_mb(slots)
    if layout.file_mb + kv + GPU_OVERHEAD_MB <= allowance:
        return Placement("gpu", gpu_mb=int(layout.file_mb + kv + GPU_OVERHEAD_MB), slots=slots, context=context)
    if not layout.is_moe:
        return Placement("fit", reason="a dense model; whole layers move to the CPU as needed", slots=slots,
                         context=context)
    if FLAG_CPU_MOE not in flags:
        return Placement("fit", reason=f"the installed llama-server has no {FLAG_CPU_MOE}", slots=slots,
                         context=context)
    ram_budget = max(0, ram_total_mb - RAM_RESERVE_MB)
    placed = _place_experts(layout, allowance, kv, GPU_OVERHEAD_MB, stream=False, flags=flags, slots=slots,
                            context=context)
    if placed.mode == "experts-cpu" and placed.cpu_mb > ram_budget:
        placed = _place_experts(layout, allowance, kv, STREAM_OVERHEAD_MB, stream=True, flags=flags, slots=slots,
                                context=context)
    return placed


def _place_experts(layout: ModelLayout, allowance: float, kv: float, overhead: int, stream: bool,
                   flags: set[str], slots: int, context: int) -> Placement:
    fixed = kv + overhead
    # The token embedding table stays in host memory, so it takes no VRAM.
    shared_gpu = layout.non_expert_mb - layout.token_embd_mb
    if shared_gpu + fixed > allowance:
        return Placement("fit", reason=f"attention, shared weights and a {int(kv)} MB KV cache alone exceed "
                                       f"the {int(allowance)} MB free; lower context_size", slots=slots,
                         context=context)
    room = allowance - shared_gpu - fixed
    blocks = sorted(layout.expert_mb_per_block)
    on_gpu = layout.expert_mb
    n_cpu = 0
    while n_cpu < len(blocks) and on_gpu > room:
        on_gpu -= layout.expert_mb_per_block[blocks[n_cpu]]
        n_cpu += 1
    leftover = room - on_gpu
    # A hot-expert cache over drive-backed experts measured far slower than none.
    cache = int(leftover - 256) if leftover > MIN_CACHE_MB and not stream and FLAG_MOE_CACHE in flags else 0
    # --n-cpu-moe counts blocks from 0, so experts interleaved with other block kinds
    # (Mamba, dense attention) still end at the right block.
    cpu_blocks = blocks[n_cpu - 1] + 1 if n_cpu else 0
    return Placement(
        "stream" if stream else "experts-cpu",
        gpu_layers=layout.block_count + 1,
        cpu_moe_blocks=cpu_blocks,
        moe_cache_mib=cache,
        gpu_mb=int(shared_gpu + on_gpu + fixed + cache),
        cpu_mb=int(layout.expert_mb - on_gpu),
        slots=slots,
        context=context,
    )


# --------------------------------------------------------------------------- #
# Machine facts (injectable)
# --------------------------------------------------------------------------- #


def default_thread_args(logical: int | None = None) -> list[str]:
    """-t/-tb for weights on the CPU: all cores for prompt reading, four fewer for
    generation on a chip with many (the last few, efficiency cores on hybrid chips,
    cost more than they add). Measured on a 20-thread Core Ultra 7 with gpt-oss-120b:
    17.0 tok/s at 20 threads, 18.1 at 16. Nothing on a small CPU."""
    logical = logical if logical is not None else (os.cpu_count() or 0)
    threads = logical - 4 if logical >= 12 else 0
    return ["-t", str(threads), "-tb", str(max(threads, logical))] if threads > 0 else []


def ram_total_mb() -> int:
    try:
        import psutil
        return int(psutil.virtual_memory().total / MB)
    except Exception:
        return 0


@functools.lru_cache(maxsize=8)
def engine_flags(binary: str, run: Callable[..., Any] = subprocess.run) -> frozenset[str]:
    """Which PLANNER_FLAGS the installed llama-server accepts, from one `--help` run
    (cached per binary path). Empty when the binary cannot be asked: the planner then
    declines and the engine's fit mode applies, which is always safe."""
    if not binary or not Path(binary).is_file():
        return frozenset()
    try:
        proc = run([binary, "--help"], capture_output=True, text=True, timeout=30, check=False,
                   cwd=str(Path(binary).parent), creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    except (OSError, subprocess.SubprocessError):
        return frozenset()
    text = (getattr(proc, "stdout", "") or "") + (getattr(proc, "stderr", "") or "")
    return frozenset(flag for flag in PLANNER_FLAGS if flag in text)


def kv_bytes_per_value(server_args: list[str]) -> float:
    """1 for a q8_0 (or smaller) KV cache set in the row's server_args, else 2 (f16)."""
    for i, arg in enumerate(server_args):
        if arg in ("--cache-type-k", "-ctk") and i + 1 < len(server_args):
            kind = server_args[i + 1].lower()
            return 1.0 if kind.startswith(("q8", "q4", "q5", "iq4", "tq")) else 2.0
    return 2.0


def slots_from(server_args: list[str]) -> int:
    for i, arg in enumerate(server_args):
        if arg in ("--parallel", "-np") and i + 1 < len(server_args):
            try:
                return max(1, int(server_args[i + 1]))
            except ValueError:
                return 1
    return 1


def plan_for_launch(binary: str, location: str, context: int, server_args: list[str], vram_free_mb: int | None,
                    *, layout_fn: Callable[[str], ModelLayout] = read_layout,
                    flags_fn: Callable[[str], frozenset[str]] = engine_flags,
                    ram_fn: Callable[[], int] = ram_total_mb) -> Placement | None:
    """The plan for one launch, or None when the planner has nothing to say (the file
    is missing or unreadable, the card's free memory is unknown, or the model is not a
    MoE bigger than the card). Never raises: a planning problem must not stop a launch."""
    if vram_free_mb is None or not location:
        return None
    try:
        if not os.path.isfile(location):
            return None
        layout = layout_fn(location)
        if not layout.is_moe or layout.file_mb + GPU_OVERHEAD_MB <= vram_free_mb - VRAM_MARGIN_MB:
            return None
        flags = set(flags_fn(binary))
        placed = plan(layout, vram_free_mb, ram_fn(), context, slots_from(server_args),
                      kv_bytes_per_value(server_args), flags)
    except Exception:   # a corrupt header, an odd binary: the engine's fit mode still works
        return None
    return placed if placed.changes_argv else None


# --------------------------------------------------------------------------- #
# Clearing the machine: GPU neighbours
# --------------------------------------------------------------------------- #
#
# Other servers on the same card (a phone line's speech server keeping a voice and a
# model loaded, a model hub holding idle models) are "GPU neighbours": settings
# services.gpu_neighbours lists their base URLs. Each speaks the same small protocol:
#
#     GET  <url>/gpu                   -> {"yield": bool, "in_call": bool}
#     POST <url>/gpu {"yield": true, "by": "locitize"}   -> {"done": bool, "in_call": bool}
#     POST <url>/gpu {"yield": false, "by": "locitize"}
#
# Before a model that needs the whole card starts, locitize asks each neighbour to give
# the GPU up, then waits for the machine's RAM to settle (the streamed-model build sizes
# its expert cache from free RAM at launch, so launching into a half-released machine
# costs tokens per second for the life of the process), re-measures and plans. When the
# model stops, the neighbours that yielded are told to come back. A neighbour on a call
# keeps the GPU and the plan simply has less to work with.

ASKER = "locitize"            # how this app names itself to its neighbours ("by")
RAM_SETTLE_SECONDS = 20.0     # at most this long for freed RAM to come back
RAM_SETTLE_STEADY = 3.0       # ... or until free RAM has not grown for this long
RAM_SETTLE_STEP_MB = 256      # growth smaller than this counts as settled


def _post_json(url: str, body: dict, timeout: float) -> dict | None:
    import json
    import urllib.request
    try:
        req = urllib.request.Request(url, json.dumps(body).encode(), {"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.load(resp)
    except Exception:
        return None


def _get_json(url: str, timeout: float) -> dict | None:
    import json
    import urllib.request
    try:
        with urllib.request.urlopen(urllib.request.Request(url), timeout=timeout) as resp:
            return json.load(resp)
    except Exception:
        return None


def ram_available_mb() -> int:
    try:
        import psutil
        return int(psutil.virtual_memory().available / MB)
    except Exception:
        return 0


def clear_machine(neighbours: list[str], *, post=_post_json, get=_get_json, ram=ram_available_mb,
                  sleep=None, clock=None, log=None) -> list[str]:
    """Ask every neighbour to give the GPU up, then wait for RAM to settle. Returns the
    neighbours that yielded (release_machine takes the same list). Never raises."""
    import time as _time
    sleep = sleep or _time.sleep
    clock = clock or _time.monotonic
    yielded: list[str] = []
    for base in [str(n).rstrip("/") for n in neighbours if str(n).strip()]:
        state = get(f"{base}/gpu", 3.0)
        if state is None:
            if log:
                log(f"GPU neighbour {base} is not answering; going on without it")
            continue
        if state.get("in_call"):
            if log:
                log(f"GPU neighbour {base} is on a call and keeps the GPU")
            continue
        if state.get("yield"):
            yielded.append(base)      # already ours (a previous start); release it later too
            continue
        answer = post(f"{base}/gpu", {"yield": True, "by": ASKER}, 25.0) or {}
        if answer.get("done"):
            yielded.append(base)
            if log:
                log(f"GPU neighbour {base} gave the GPU up")
        elif log:
            log(f"GPU neighbour {base} kept the GPU ({'on a call' if answer.get('in_call') else 'refused'})")
    if yielded:
        _wait_for_ram(ram, sleep, clock, log)
    return yielded


def _wait_for_ram(ram, sleep, clock, log) -> None:
    """Wait until free RAM stops growing (the neighbours' pages have come back)."""
    start = last_growth = clock()
    best = ram()
    while clock() - start < RAM_SETTLE_SECONDS:
        sleep(0.5)
        now = ram()
        if now - best >= RAM_SETTLE_STEP_MB:
            best, last_growth = now, clock()
        elif clock() - last_growth >= RAM_SETTLE_STEADY:
            break
    if log:
        log(f"RAM settled at {best} MB free")


def release_machine(neighbours: list[str], *, post=_post_json, log=None) -> None:
    """Tell the neighbours that yielded to take the GPU back. Never raises."""
    for base in neighbours:
        answer = post(f"{base}/gpu", {"yield": False, "by": ASKER}, 25.0)
        if log:
            log(f"GPU neighbour {base} {'has the GPU back' if answer and answer.get('done') else 'could not be told to come back'}")


def needs_the_whole_card(location: str, vram_total_mb: int | None, *, layout_fn=read_layout) -> bool:
    """True for a mixture-of-experts model that cannot fit the card even when empty: the
    one case worth clearing the machine for. Never raises."""
    if not vram_total_mb or not location:
        return False
    try:
        if not os.path.isfile(location):
            return False
        layout = layout_fn(location)
    except Exception:
        return False
    return layout.is_moe and layout.file_mb + GPU_OVERHEAD_MB > vram_total_mb - VRAM_MARGIN_MB


def placement_mode(value: str | None) -> str:
    """Normalize a row's `placement:` ("" and "auto" plan; "fit" never does)."""
    mode = str(value or "").strip().lower()
    return mode if mode in PLACEMENTS and mode else PLACEMENT_AUTO
