"""Big-model placement (placement.py): a mixture-of-experts model larger than the card
keeps its shared weights on the GPU and its experts in RAM or streamed from the drive.

Every test is headless: GGUF files are written here with a tensor table and no weights,
the binary's --help is faked, and RAM is a number. The contract proven:

  * the tensor table is read correctly (sizes per type, experts per block, shards);
  * the planner picks gpu / experts-cpu / stream / fit for the right reasons and never
    over-fills the card;
  * only experts-cpu and stream change the argv; the start spec for any model that
    fits stays byte-identical to a launch without the planner;
  * a row's `placement: fit` opts out, and nothing the planner does can stop a launch.
"""

from __future__ import annotations

import struct
from pathlib import Path

import pytest

import backends
import placement
from config import Model, ModelRegistryData, Settings
from models import ModelRegistry
from placement import (
    FLAG_CPU_MOE,
    FLAG_MOE_CACHE,
    FLAG_NO_OP_OFFLOAD,
    MB,
    ModelLayout,
    Placement,
    plan,
    plan_for_launch,
    read_layout,
    shard_paths,
)

# --------------------------------------------------------------------------- #
# A GGUF writer small enough to live in the test
# --------------------------------------------------------------------------- #

_U32, _STR = 4, 8


def _gguf_string(text: str) -> bytes:
    data = text.encode("utf-8")
    return struct.pack("<Q", len(data)) + data


def write_gguf(path: Path, arch: str, meta: dict[str, int], tensors: list[tuple[str, list[int], int]]) -> Path:
    """A GGUF v3 header: general.architecture, integer `<arch>.<key>` facts, one string
    array (tokenizer-shaped, to prove arrays are skipped) and a tensor table. No weights."""
    out = bytearray(b"GGUF" + struct.pack("<I", 3) + struct.pack("<Q", len(tensors)) + struct.pack("<Q", len(meta) + 2))
    out += _gguf_string("general.architecture") + struct.pack("<I", _STR) + _gguf_string(arch)
    for key, value in meta.items():
        out += _gguf_string(f"{arch}.{key}") + struct.pack("<I", _U32) + struct.pack("<I", value)
    out += _gguf_string("tokenizer.ggml.tokens") + struct.pack("<I", 9) + struct.pack("<I", _STR) + struct.pack("<Q", 3)
    for token in ("a", "bb", "ccc"):
        out += _gguf_string(token)
    for name, dims, ggml_type in tensors:
        out += _gguf_string(name) + struct.pack("<I", len(dims))
        for d in dims:
            out += struct.pack("<Q", d)
        out += struct.pack("<I", ggml_type) + struct.pack("<Q", 0)
    path.write_bytes(bytes(out))
    return path


def _moe_tensors(blocks: int, expert_elems: int, attn_elems: int, expert_type: int = 39) -> list:
    """`blocks` blocks each with f16 attention weights and MXFP4 expert weights, plus the
    token embedding and output tables."""
    tensors = [("token_embd.weight", [1024, 2048], 1), ("output.weight", [1024, 2048], 1)]
    for b in range(blocks):
        tensors.append((f"blk.{b}.attn_qkv.weight", [attn_elems], 1))
        tensors.append((f"blk.{b}.ffn_gate_up_exps.weight", [expert_elems], expert_type))
        tensors.append((f"blk.{b}.ffn_down_exps.weight", [expert_elems], expert_type))
    return tensors


def _dense_tensors(blocks: int, elems: int) -> list:
    tensors = [("token_embd.weight", [1024, 2048], 1)]
    for b in range(blocks):
        tensors.append((f"blk.{b}.attn_qkv.weight", [elems], 1))
        tensors.append((f"blk.{b}.ffn_up.weight", [elems], 8))
    return tensors


_ARCH_META = {"block_count": 4, "attention.head_count": 8, "attention.head_count_kv": 2,
              "embedding_length": 1024, "context_length": 131072, "expert_count": 32}


# --------------------------------------------------------------------------- #
# Reading the tensor table
# --------------------------------------------------------------------------- #


def test_read_layout_sizes_tensors_by_type_and_finds_the_experts(tmp_path: Path):
    path = write_gguf(tmp_path / "moe.gguf", "gptoss", _ARCH_META,
                      _moe_tensors(blocks=4, expert_elems=32 * 1000, attn_elems=2000))
    layout = read_layout(str(path))
    assert layout.architecture == "gptoss" and layout.block_count == 4 and layout.expert_count == 32
    assert layout.is_moe and sorted(layout.expert_mb_per_block) == [0, 1, 2, 3]
    # MXFP4: 17 bytes per 32 elements, two expert tensors per block.
    assert layout.expert_mb_per_block[0] == pytest.approx(2 * 1000 * 17 / MB)
    # Attention is f16 (2 bytes per element); the embedding table is f16 too and is set apart.
    assert layout.token_embd_mb == pytest.approx(1024 * 2048 * 2 / MB)
    assert layout.non_expert_mb == pytest.approx((2 * 1024 * 2048 * 2 + 4 * 2000 * 2) / MB)
    assert layout.file_mb == pytest.approx(layout.expert_mb + layout.non_expert_mb)
    # KV per token at f16: blocks * kv heads * (key + value) * 2 bytes, head size 1024/8.
    assert layout.kv_bytes_per_token == 4 * 2 * (128 + 128) * 2
    assert layout.kv_mb(1024, 1.0) == pytest.approx(layout.kv_bytes_per_token * 0.5 * 1024 / MB)
    assert layout.context_length == 131072


def test_read_layout_dense_model_has_no_experts(tmp_path: Path):
    path = write_gguf(tmp_path / "dense.gguf", "qwen3", {"block_count": 2, "attention.head_count": 4,
                      "embedding_length": 512}, _dense_tensors(2, 4096))
    layout = read_layout(str(path))
    assert not layout.is_moe and layout.expert_mb == 0
    # Q8_0: 34 bytes per 32 elements.
    assert layout.file_mb == pytest.approx((1024 * 2048 * 2 + 2 * (4096 * 2 + 4096 // 32 * 34)) / MB)
    # No kv head count: every head keeps a cache.
    assert layout.kv_bytes_per_token == 2 * 4 * (128 + 128) * 2


def test_read_layout_sums_every_shard_of_a_split_file(tmp_path: Path):
    first = write_gguf(tmp_path / "big-00001-of-00002.gguf", "gptoss", _ARCH_META,
                       _moe_tensors(blocks=2, expert_elems=3200, attn_elems=100))
    write_gguf(tmp_path / "big-00002-of-00002.gguf", "gptoss", {}, [
        ("blk.2.ffn_gate_up_exps.weight", [3200], 39), ("blk.3.attn_qkv.weight", [100], 1)])
    assert [Path(p).name for p in shard_paths(str(first))] == ["big-00001-of-00002.gguf", "big-00002-of-00002.gguf"]
    layout = read_layout(str(first))
    assert sorted(layout.expert_mb_per_block) == [0, 1, 2] and layout.block_count == 4
    assert shard_paths("plain.gguf") == ["plain.gguf"]


def test_read_layout_charges_a_kv_cache_only_to_attention_blocks_of_a_hybrid(tmp_path: Path):
    # Six blocks: attention in 0 and 3, Mamba (ssm_) in the rest, experts in the odd ones.
    tensors = [("token_embd.weight", [1024, 2048], 1)]
    for b in range(6):
        if b in (0, 3):
            tensors += [(f"blk.{b}.attn_k.weight", [1000], 1), (f"blk.{b}.attn_v.weight", [1000], 1)]
        else:
            tensors += [(f"blk.{b}.ssm_a", [1000], 0), (f"blk.{b}.ssm_in.weight", [1000], 1)]
        if b % 2:
            tensors.append((f"blk.{b}.ffn_up_exps.weight", [3200], 39))
    meta = {"block_count": 6, "attention.head_count": 8, "attention.head_count_kv": 2, "embedding_length": 1024,
            "expert_count": 64, "ssm.state_size": 128, "ssm.inner_size": 2048, "ssm.conv_kernel": 4,
            "ssm.group_count": 8}
    path = write_gguf(tmp_path / "hybrid.gguf", "nemotron_h", meta, tensors)
    layout = read_layout(str(path))
    assert layout.kv_blocks == 2 and layout.ssm_blocks == 4 and sorted(layout.expert_mb_per_block) == [1, 3, 5]
    assert layout.kv_bytes_per_token == 2 * 2 * (128 + 128) * 2, "only the two attention blocks keep a cache"
    state = (128 * 2048 * 4 + 3 * (2048 + 2 * 8 * 128) * 4) / MB
    assert layout.ssm_state_mb == pytest.approx(state) and layout.recurrent_mb(2) == pytest.approx(4 * state * 2)
    # A plain attention model keeps a cache in every block.
    plain = read_layout(str(write_gguf(tmp_path / "plain.gguf", "gptoss", _ARCH_META,
                                       _moe_tensors(blocks=4, expert_elems=3200, attn_elems=100))))
    assert plain.kv_blocks == 4 and plain.ssm_blocks == 0 and plain.recurrent_mb() == 0


def test_read_layout_refuses_a_file_that_is_not_gguf(tmp_path: Path):
    bad = tmp_path / "bad.gguf"
    bad.write_bytes(b"not a model at all")
    with pytest.raises(Exception):
        read_layout(str(bad))


# --------------------------------------------------------------------------- #
# The plan
# --------------------------------------------------------------------------- #


def _layout(expert_per_block_mb: float, blocks: int, shared_mb: float, embd_mb: float = 500.0,
            kv_per_token: float = 4 * 8 * 256 * 2) -> ModelLayout:
    experts = {b: expert_per_block_mb for b in range(blocks)}
    return ModelLayout(path="m.gguf", architecture="gptoss", file_mb=shared_mb + expert_per_block_mb * blocks,
                       block_count=blocks, expert_count=32, context_length=131072,
                       kv_bytes_per_token=kv_per_token, token_embd_mb=embd_mb, expert_mb_per_block=experts)


ALL_FLAGS = {FLAG_CPU_MOE, FLAG_MOE_CACHE, FLAG_NO_OP_OFFLOAD, placement.FLAG_LOAD_MODE}


def test_plan_leaves_a_model_that_fits_alone():
    layout = _layout(expert_per_block_mb=100, blocks=10, shared_mb=2000)   # 3 GB
    placed = plan(layout, vram_free_mb=16000, ram_total_mb=32000, context=8192, flags=ALL_FLAGS)
    assert placed.mode == "gpu" and not placed.changes_argv and placed.args(ALL_FLAGS) == [] and placed.env() == {}


def test_plan_moves_the_first_blocks_experts_to_ram_until_the_rest_fits():
    # 36 blocks of 1,000 MB experts (36 GB) on a 16 GB card with 3 GB of shared weights, 32 GB RAM.
    layout = _layout(expert_per_block_mb=1000, blocks=36, shared_mb=3000, embd_mb=500)
    placed = plan(layout, vram_free_mb=15000, ram_total_mb=64000, context=8192, kv_bytes_per_value=1.0,
                  flags=ALL_FLAGS)
    assert placed.mode == "experts-cpu"
    kv = layout.kv_mb(8192, 1.0)
    room = (15000 - 256) - (3000 - 500) - kv - placement.GPU_OVERHEAD_MB
    on_gpu = 36 - placed.cpu_moe_blocks
    assert on_gpu * 1000 <= room < (on_gpu + 1) * 1000, "as many whole expert blocks as fit, no more"
    assert placed.cpu_mb == placed.cpu_moe_blocks * 1000
    # 64 GB of RAM holds them, so no streaming; the leftover VRAM caches hot experts.
    assert placed.moe_cache_mib == int(room - on_gpu * 1000 - 256) if room - on_gpu * 1000 > placement.MIN_CACHE_MB \
        else placed.moe_cache_mib == 0
    assert placed.gpu_mb <= 15000 - 256
    args = placed.args(ALL_FLAGS, thread_args=["-t", "16", "-tb", "20"])
    assert args[:2] == [FLAG_CPU_MOE, str(placed.cpu_moe_blocks)]
    assert FLAG_NO_OP_OFFLOAD not in args and args[-8:] == ["-b", "2048", "-ub", "1024", "-t", "16", "-tb", "20"]
    assert "--load-mode" not in args, "experts that fit RAM may be loaded whole; the row or the build decides"
    assert placed.env() == {}
    assert "in RAM" in placed.describe()


def test_plan_streams_from_the_drive_when_the_experts_exceed_ram():
    # The same model on a 32 GB machine: 24 GB of RAM budget cannot hold ~25 GB of experts.
    layout = _layout(expert_per_block_mb=1000, blocks=36, shared_mb=3000)
    placed = plan(layout, vram_free_mb=15000, ram_total_mb=32000, context=8192, kv_bytes_per_value=1.0,
                  flags=ALL_FLAGS)
    assert placed.mode == "stream" and placed.cpu_mb > 32000 - placement.RAM_RESERVE_MB
    assert placed.moe_cache_mib == 0, "a hot-expert cache over drive-backed experts is slower than none"
    args = placed.args(ALL_FLAGS, thread_args=[])
    assert FLAG_NO_OP_OFFLOAD in args and args[:2] == [FLAG_CPU_MOE, str(placed.cpu_moe_blocks)]
    assert args[args.index("--load-mode") + 1] == "mmap", "streaming needs the memory map whatever the default"
    assert placed.env() == {"GGML_EXPERT_STREAM": "1"}
    assert "streamed from the drive" in placed.describe()
    # The smaller stream overhead buys the GPU at least as many expert blocks as the RAM plan.
    in_ram = plan(layout, vram_free_mb=15000, ram_total_mb=64000, context=8192, kv_bytes_per_value=1.0, flags=ALL_FLAGS)
    assert placed.cpu_moe_blocks <= in_ram.cpu_moe_blocks


def test_plan_counts_blocks_not_expert_layers_when_experts_are_interleaved():
    # Experts only in the odd blocks (a hybrid like Nemotron-H): --n-cpu-moe must still
    # name a block number, so the first two expert layers (blocks 1 and 3) end at block 4.
    layout = ModelLayout(path="m.gguf", architecture="nemotron_h", file_mb=3000 + 6 * 1000, block_count=12,
                         expert_count=64, context_length=8192, kv_bytes_per_token=1000.0, token_embd_mb=0,
                         expert_mb_per_block={1: 1000, 3: 1000, 5: 1000, 7: 1000, 9: 1000, 11: 1000})
    placed = plan(layout, vram_free_mb=3000 + 256 + placement.GPU_OVERHEAD_MB + 4000 + 10, ram_total_mb=64000,
                  context=1024, flags=ALL_FLAGS)
    assert placed.mode == "experts-cpu" and placed.cpu_moe_blocks == 4 and placed.cpu_mb == 2000


def test_plan_falls_back_to_fit_mode_for_dense_models_and_unfit_shared_weights():
    dense = ModelLayout(path="d.gguf", architecture="qwen3", file_mb=20000, block_count=64, expert_count=0,
                        context_length=32768, kv_bytes_per_token=1000.0, token_embd_mb=500)
    assert plan(dense, 15000, 32000, 8192, flags=ALL_FLAGS).mode == "fit"
    huge_shared = _layout(expert_per_block_mb=1000, blocks=10, shared_mb=20000)
    placed = plan(huge_shared, 15000, 32000, 8192, flags=ALL_FLAGS)
    assert placed.mode == "fit" and "lower context_size" in placed.describe()
    # A KV cache that alone eats the card says the same.
    layout = _layout(expert_per_block_mb=1000, blocks=36, shared_mb=3000, kv_per_token=65536)   # 16 GB at 262k
    assert plan(layout, 15000, 32000, 262144, kv_bytes_per_value=2.0, flags=ALL_FLAGS).mode == "fit"
    assert plan(layout, 15000, 32000, 8192, flags=ALL_FLAGS).mode == "stream"


def test_plan_declines_when_the_binary_lacks_n_cpu_moe_and_omits_flags_it_lacks():
    layout = _layout(expert_per_block_mb=1000, blocks=36, shared_mb=3000)
    assert plan(layout, 15000, 32000, 8192, flags=set()).mode == "fit"
    placed = plan(layout, 15000, 64000, 8192, kv_bytes_per_value=1.0, flags={FLAG_CPU_MOE})
    assert placed.mode == "experts-cpu" and placed.moe_cache_mib == 0, "no --moe-cache-mib on this build"
    streamed = plan(layout, 15000, 32000, 8192, kv_bytes_per_value=1.0, flags={FLAG_CPU_MOE})
    args = streamed.args({FLAG_CPU_MOE}, thread_args=[])
    assert FLAG_NO_OP_OFFLOAD not in args and FLAG_MOE_CACHE not in args and args[0] == FLAG_CPU_MOE


def test_thread_args_spare_the_last_cores_only_on_a_big_cpu():
    assert placement.default_thread_args(logical=20) == ["-t", "16", "-tb", "20"]
    assert placement.default_thread_args(logical=8) == []


def test_server_args_decide_the_kv_bytes_and_the_slots():
    assert placement.kv_bytes_per_value(["--parallel", "1", "--cache-type-k", "q8_0", "--cache-type-v", "q8_0"]) == 1.0
    assert placement.kv_bytes_per_value(["-ctk", "f16"]) == 2.0 and placement.kv_bytes_per_value([]) == 2.0
    assert placement.slots_from(["--parallel", "2"]) == 2 and placement.slots_from([]) == 1


def test_engine_flags_come_from_help_once_per_binary(tmp_path: Path):
    exe = tmp_path / "llama-server.exe"
    exe.write_bytes(b"")
    calls = []

    class Done:
        returncode = 0
        stdout = "--n-cpu-moe N   keep the experts ...\n--moe-cache-mib N  cache\n"
        stderr = ""

    def run(cmd, **kw):
        calls.append(cmd)
        return Done()

    placement.engine_flags.cache_clear()
    assert placement.engine_flags(str(exe), run) == frozenset({FLAG_CPU_MOE, FLAG_MOE_CACHE})
    assert placement.engine_flags(str(exe), run) == frozenset({FLAG_CPU_MOE, FLAG_MOE_CACHE})
    assert len(calls) == 1 and calls[0][1] == "--help"
    assert placement.engine_flags(str(tmp_path / "missing.exe"), run) == frozenset()
    placement.engine_flags.cache_clear()


# --------------------------------------------------------------------------- #
# plan_for_launch: the guard between the registry and the planner
# --------------------------------------------------------------------------- #


def _big_moe(tmp_path: Path) -> Path:
    # 48 blocks of 2 * 960,000,000 MXFP4 elements (17 bytes per 32): about 1,020 MB of experts
    # per block, 48 GB in all, far beyond a 16 GB card and a 32 GB machine's RAM budget.
    return write_gguf(tmp_path / "gpt-oss-120b.gguf", "gptoss", {**_ARCH_META, "block_count": 48},
                      _moe_tensors(blocks=48, expert_elems=32 * 30_000_000, attn_elems=20_000_000))


def test_plan_for_launch_plans_only_a_moe_bigger_than_the_card(tmp_path: Path):
    big = _big_moe(tmp_path)
    flags = lambda binary: frozenset(ALL_FLAGS)   # noqa: E731
    placed = plan_for_launch("llama-server.exe", str(big), 8192, ["--cache-type-k", "q8_0"], 15000,
                             flags_fn=flags, ram_fn=lambda: 32000)
    assert placed is not None and placed.mode == "stream"
    # The same file on a card it fits: nothing to say.
    assert plan_for_launch("llama-server.exe", str(big), 8192, [], 60000, flags_fn=flags, ram_fn=lambda: 32000) is None
    # A missing file, an unknown free-memory reading, a dense model: nothing to say either.
    assert plan_for_launch("llama-server.exe", str(tmp_path / "nope.gguf"), 8192, [], 15000, flags_fn=flags) is None
    assert plan_for_launch("llama-server.exe", str(big), 8192, [], None, flags_fn=flags) is None
    dense = write_gguf(tmp_path / "dense.gguf", "qwen3", {"block_count": 2, "attention.head_count": 4,
                       "embedding_length": 512}, _dense_tensors(2, 400_000_000))
    assert plan_for_launch("llama-server.exe", str(dense), 8192, [], 1000, flags_fn=flags) is None


def test_plan_for_launch_never_raises(tmp_path: Path):
    bad = tmp_path / "bad.gguf"
    bad.write_bytes(b"GGUF" + b"\xff" * 40)
    assert plan_for_launch("llama-server.exe", str(bad), 8192, [], 15000) is None

    def boom(path):
        raise RuntimeError("boom")
    good = _big_moe(tmp_path)
    assert plan_for_launch("llama-server.exe", str(good), 8192, [], 15000, layout_fn=boom) is None


# --------------------------------------------------------------------------- #
# The start spec
# --------------------------------------------------------------------------- #


class _Budget:
    target_mib = 1500
    real_free_mib = 15000

    def describe(self):
        return "fit target 1500 MiB"


def _registry(tmp_path: Path, placement_value: str = "", location: str | None = None) -> ModelRegistry:
    settings = Settings()
    settings.paths.llama_cpp = "llama-server.exe"
    data = ModelRegistryData(version=1, models=[Model(
        id="big", name="Big", description="d", location=location or str(tmp_path / "gpt-oss-120b.gguf"),
        context_size=8192, gpu_layers=999, status="installed", placement=placement_value,
        server_args=["--parallel", "1", "--cache-type-k", "q8_0", "--cache-type-v", "q8_0"])])
    registry = ModelRegistry(data, settings)
    registry._fit_budget_fn = lambda binary: _Budget()
    return registry


def test_start_spec_carries_the_plan_for_a_streamed_model(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    _big_moe(tmp_path)
    registry = _registry(tmp_path)
    monkeypatch.setattr(placement, "engine_flags", lambda binary: frozenset(ALL_FLAGS))
    monkeypatch.setattr(placement, "ram_total_mb", lambda: 32000)
    seen = {}

    def planner(binary, location, context, server_args, vram_free):
        seen.update(binary=binary, location=location, context=context, server_args=server_args, vram_free=vram_free)
        return placement.plan_for_launch(binary, location, context, server_args, vram_free,
                                         flags_fn=lambda b: frozenset(ALL_FLAGS), ram_fn=lambda: 32000)
    registry._placement_fn = planner
    spec = registry.build_start_spec("big")
    assert seen == {"binary": "llama-server.exe", "location": str(tmp_path / "gpt-oss-120b.gguf"), "context": 8192,
                    "server_args": ["--parallel", "1", "--cache-type-k", "q8_0", "--cache-type-v", "q8_0"],
                    "vram_free": 15000}
    cmd = spec.command
    # Every layer on the GPU except the experts the plan names; no fit request.
    assert cmd[cmd.index("--n-gpu-layers") + 1] == "49" and "--fit-target" not in cmd   # 48 blocks + output
    assert FLAG_CPU_MOE in cmd and FLAG_NO_OP_OFFLOAD in cmd and "-ub" in cmd
    # The plan's flags come before the owner's server_args, so an owner flag still wins.
    assert cmd.index(FLAG_CPU_MOE) < cmd.index("--cache-type-k")
    assert spec.env == {"GGML_EXPERT_STREAM": "1"}


def test_start_spec_is_unchanged_for_a_model_that_fits_or_opts_out(tmp_path: Path):
    _big_moe(tmp_path)
    baseline = _registry(tmp_path, placement_value="fit")
    baseline._placement_fn = lambda *a: pytest.fail("placement: fit must never plan")
    without = baseline.build_start_spec("big")
    assert FLAG_CPU_MOE not in without.command and without.env == {}

    fits = _registry(tmp_path)
    fits._placement_fn = lambda *a: None       # the planner had nothing to say
    assert fits.build_start_spec("big").command == without.command

    # Fit mode's --fit-target survives when there is no plan (gpu_layers -1 asks for it).
    data = ModelRegistryData(version=1, models=[Model(id="m", name="m", description="d",
                             location=str(tmp_path / "gpt-oss-120b.gguf"), context_size=8192, gpu_layers=-1)])
    settings = Settings()
    settings.paths.llama_cpp = "llama-server.exe"
    registry = ModelRegistry(data, settings)
    registry._fit_budget_fn = lambda binary: _Budget()
    registry._placement_fn = lambda *a: None
    cmd = registry.build_start_spec("m").command
    assert cmd[cmd.index("--n-gpu-layers") + 1] == "-1" and cmd[cmd.index("--fit-target") + 1] == "1500"


def test_start_spec_without_a_budget_reading_does_not_plan(tmp_path: Path):
    _big_moe(tmp_path)
    registry = _registry(tmp_path)
    registry._fit_budget_fn = lambda binary: None
    registry._placement_fn = lambda *a: pytest.fail("no free-memory reading: no plan")
    assert FLAG_CPU_MOE not in registry.build_start_spec("big").command


# --------------------------------------------------------------------------- #
# Clearing the machine: GPU neighbours
# --------------------------------------------------------------------------- #


class _Neighbours:
    """Fake neighbours keyed by base URL: state dicts the protocol reads and flips."""

    def __init__(self, **states):
        self.states = states
        self.posts: list[tuple[str, dict]] = []

    def get(self, url, timeout):
        base = url.rsplit("/gpu", 1)[0]
        state = self.states.get(base)
        return None if state is None else dict(state)

    def post(self, url, body, timeout):
        base = url.rsplit("/gpu", 1)[0]
        self.posts.append((base, body))
        state = self.states.get(base)
        if state is None:
            return None
        if body["yield"] and state["in_call"]:
            return {"done": False, "in_call": True}
        state["yield"] = body["yield"]
        return {"done": True, "in_call": False}


def _ram_steps(values):
    it = iter(values)
    last = [values[0]]

    def ram():
        try:
            last[0] = next(it)
        except StopIteration:
            pass
        return last[0]
    return ram


def test_clear_machine_asks_each_neighbour_and_waits_for_ram(monkeypatch: pytest.MonkeyPatch):
    fake = _Neighbours(**{"http://speech:47825": {"yield": False, "in_call": False},
                          "http://hub:8787/api/compute": {"yield": False, "in_call": False}})
    slept, logs, now = [], [], [0.0]
    ram = _ram_steps([2000, 6000, 12000, 15000, 15100, 15150, 15160, 15160, 15160, 15160, 15160])
    monkeypatch.setattr(placement, "RAM_SETTLE_STEADY", 1.0)

    def sleep(seconds):
        slept.append(seconds)
        now[0] += seconds
    yielded = placement.clear_machine(["http://speech:47825/", "http://hub:8787/api/compute", ""], post=fake.post,
                                      get=fake.get, ram=ram, sleep=sleep, clock=lambda: now[0], log=logs.append)
    assert yielded == ["http://speech:47825", "http://hub:8787/api/compute"]
    assert [b for _, b in fake.posts] == [{"yield": True, "by": "locitize"}] * 2
    assert all(s["yield"] for s in fake.states.values())
    assert 3 <= len(slept) <= 40 and any(t.startswith("RAM settled at") for t in logs)


def test_clear_machine_leaves_a_busy_or_absent_neighbour_alone():
    fake = _Neighbours(**{"http://speech:47825": {"yield": False, "in_call": True},
                          "http://already": {"yield": True, "in_call": False}})
    logs = []
    yielded = placement.clear_machine(["http://speech:47825", "http://gone:1", "http://already"], post=fake.post,
                                      get=fake.get, ram=lambda: 1000, sleep=lambda s: None, log=logs.append)
    assert yielded == ["http://already"], "one on a call keeps the GPU; one that already yielded is ours to release"
    assert fake.posts == [] and any("on a call" in t for t in logs) and any("not answering" in t for t in logs)


def test_release_machine_tells_the_neighbours_to_come_back():
    fake = _Neighbours(**{"http://speech:47825": {"yield": True, "in_call": False}})
    logs = []
    placement.release_machine(["http://speech:47825", "http://gone:1"], post=fake.post, log=logs.append)
    assert fake.states["http://speech:47825"]["yield"] is False
    assert [b for _, b in fake.posts] == [{"yield": False, "by": "locitize"}] * 2
    assert any("has the GPU back" in t for t in logs) and any("could not be told" in t for t in logs)


def test_needs_the_whole_card_is_true_only_for_a_moe_bigger_than_an_empty_card(tmp_path: Path):
    big = _big_moe(tmp_path)
    assert placement.needs_the_whole_card(str(big), 16303)
    assert not placement.needs_the_whole_card(str(big), 80000)
    assert not placement.needs_the_whole_card(str(big), None)
    assert not placement.needs_the_whole_card(str(tmp_path / "missing.gguf"), 16303)
    dense = write_gguf(tmp_path / "dense.gguf", "qwen3", {"block_count": 2, "attention.head_count": 4,
                       "embedding_length": 512}, _dense_tensors(2, 400_000_000))
    assert not placement.needs_the_whole_card(str(dense), 1000), "a dense model is the engine's fit mode's job"


class _TotalBudget(_Budget):
    adapter_total_mib = 16303


def test_start_spec_clears_the_machine_for_a_model_that_needs_the_whole_card(tmp_path: Path):
    _big_moe(tmp_path)
    registry = _registry(tmp_path)
    registry._settings.services.gpu_neighbours = ["http://speech:47825"]
    readings = iter([_TotalBudget(), type("Bigger", (_TotalBudget,), {"real_free_mib": 15800})()])
    registry._fit_budget_fn = lambda binary: next(readings)
    cleared, released, planned = [], [], []
    registry._clear_fn = lambda neighbours, log=None: cleared.append(list(neighbours)) or ["http://speech:47825"]
    registry._release_fn = lambda neighbours, log=None: released.append(list(neighbours))
    registry._placement_fn = lambda b, loc, ctx, args, free: planned.append(free) or None
    spec = registry.build_start_spec("big")
    assert cleared == [["http://speech:47825"]]
    assert planned == [15800], "the card is measured again after the neighbours stepped aside"
    assert spec.on_stop is not None and released == []
    spec.on_stop()
    assert released == [["http://speech:47825"]]


def test_start_spec_does_not_clear_the_machine_when_nothing_needs_it(tmp_path: Path):
    _big_moe(tmp_path)
    registry = _registry(tmp_path)
    registry._fit_budget_fn = lambda binary: _TotalBudget()
    registry._clear_fn = lambda *a, **k: pytest.fail("no neighbours configured: nobody to ask")
    assert registry.build_start_spec("big").on_stop is None
    registry._settings.services.gpu_neighbours = ["http://speech:47825"]
    opted_out = _registry(tmp_path, placement_value="fit")
    opted_out._settings.services.gpu_neighbours = ["http://speech:47825"]
    opted_out._fit_budget_fn = lambda binary: _TotalBudget()
    opted_out._clear_fn = lambda *a, **k: pytest.fail("placement: fit never clears the machine")
    assert opted_out.build_start_spec("big").on_stop is None
    small = _registry(tmp_path, location=str(write_gguf(tmp_path / "small.gguf", "gptoss", _ARCH_META,
                                                         _moe_tensors(blocks=4, expert_elems=3200, attn_elems=100))))
    small._settings.services.gpu_neighbours = ["http://speech:47825"]
    small._fit_budget_fn = lambda binary: _TotalBudget()
    small._clear_fn = lambda *a, **k: pytest.fail("a model that fits the card asks nobody")
    assert small.build_start_spec("big").on_stop is None


def test_managed_process_runs_on_stop_once_after_a_confirmed_stop():
    import services
    from services import ServiceSpec

    ran = []
    spec = ServiceSpec(name="llama_cpp:big", command=["x"], on_stop=lambda: ran.append("released"))
    proc = services.ManagedProcess.__new__(services.ManagedProcess)
    proc._spec, proc._log = spec, services.get_logger("launcher") if hasattr(services, "get_logger") else __import__("logging").getLogger("t")
    proc._run_on_stop()
    proc._run_on_stop()
    assert ran == ["released"] and spec.on_stop is None
    bad = ServiceSpec(name="llama_cpp:big", command=["x"], on_stop=lambda: 1 / 0)
    proc._spec = bad
    proc._run_on_stop()   # logged, never raised


def test_placement_mode_normalizes_the_row_value():
    assert placement.placement_mode("") == "auto" and placement.placement_mode(None) == "auto"
    assert placement.placement_mode(" Fit ") == "fit" and placement.placement_mode("whatever") == "auto"
    assert backends.ALL_LAYERS == 999
