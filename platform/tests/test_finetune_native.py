"""Native (Docker-free) fine-tune pieces: preflight messages, GGUF export steps,
setup pins. Headless: no GPU, no torch, no network, no llama.cpp process."""

from __future__ import annotations

import io
import json
import os
import sys
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest

TRAINER = Path(__file__).resolve().parents[2] / "finetune-studio" / "trainer"
sys.path.insert(0, str(TRAINER))

import gguf_export  # noqa: E402
import download_model  # noqa: E402
import finetune_advisor as advisor  # noqa: E402
import preflight  # noqa: E402
import setup_env  # noqa: E402
import setup_plan  # noqa: E402


# ----------------------------------------------------------------- preflight

def test_estimate_scales_with_model_size():
    small = preflight.estimate_vram_mb("unsloth/Qwen2.5-0.5B-Instruct")
    mid = preflight.estimate_vram_mb("unsloth/Qwen2.5-1.5B-Instruct")
    big = preflight.estimate_vram_mb("unsloth/Meta-Llama-3.1-8B-Instruct")
    assert small < mid < big
    assert big < 16000


def test_no_gpu_message_is_plain_language():
    res = preflight.check("x-1.5B", gpu=False, finder=lambda n: object())
    assert not res.ok
    assert "NVIDIA" in res.message()
    assert "Traceback" not in res.message()


def test_missing_packages_names_them_and_the_fix():
    res = preflight.check("x-1.5B", gpu=("RTX", 16000, 16000),
                          finder=lambda n: None if n in ("peft", "trl") else object())
    assert not res.ok
    assert "peft" in res.message() and "trl" in res.message()
    assert "setup" in res.message()


def test_not_enough_vram_reports_numbers():
    res = preflight.check("x-8B", gpu=("RTX 5070 Ti", 2000, 16303),
                          finder=lambda n: object())
    assert not res.ok
    assert "free" in res.message() and "RTX 5070 Ti" in res.message()


def test_ok_when_everything_present():
    res = preflight.check("x-1.5B", gpu=("RTX", 15000, 16000), finder=lambda n: object())
    assert res.ok and res.problems == []


def test_query_gpu_parses_nvidia_smi():
    class P:
        returncode = 0
        stdout = "NVIDIA GeForce RTX 5070 Ti, 5632, 16303\n"

    assert preflight.query_gpu(lambda *a, **k: P()) == ("NVIDIA GeForce RTX 5070 Ti", 5632, 16303)


def test_query_gpu_missing_binary_is_none():
    def boom(*a, **k):
        raise FileNotFoundError("nvidia-smi")

    assert preflight.query_gpu(boom) is None


# -------------------------------------------------------------- model advisor

def test_advisor_parses_sizes_and_estimates_qlora_fit():
    assert advisor.params_b_from_id("org/Llama-3.1-8B-Instruct") == 8.0
    assert advisor.params_b_from_id("org/model-270M") == 0.27
    assert advisor.params_b_from_id("org/mix-x7b") == 7.0
    assert advisor.params_b_from_id("org/no-size") is None
    assert advisor.qlora_fit_gb(8) == pytest.approx(7.9)
    assert advisor.fits(8, 8)
    assert not advisor.fits(8, 7)


def test_advisor_reuses_the_gpu_provider_and_picks_the_largest_card():
    cards = [
        type("Gpu", (), {"vram_total_mb": 8192})(),
        type("Gpu", (), {"vram_total_mb": 24576})(),
    ]
    provider = type("Provider", (), {"gpus": lambda self: cards})()
    assert advisor.detect_vram_gb(provider) == 24.0


def test_advisor_fetch_uses_injected_checked_opener_and_filters_artifacts():
    payload = json.dumps([
        {"id": "org/Good-7B-Instruct", "downloads": 50},
        {"id": "org/Serving-7B-GGUF", "downloads": 500},
        {"id": "org/Gated-3B", "downloads": 100, "gated": True},
        {"id": "org/NoSize", "downloads": 1000},
    ]).encode("utf-8")

    class Response(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

    class Opener:
        def __init__(self):
            self.calls = []

        def open(self, request, timeout=None):
            self.calls.append((request.full_url, timeout))
            return Response(payload)

    opener = Opener()
    rows = advisor.fetch_candidates(opener=opener, timeout=7)
    assert rows == [{"id": "org/Good-7B-Instruct", "downloads": 50, "params_b": 7.0}]
    assert opener.calls[0][0].startswith("https://huggingface.co/api/models?")
    assert opener.calls[0][1] == 7


def test_advisor_sorts_biggest_fit_first_and_network_failure_is_plain():
    rows = [
        {"id": "org/Small-1B", "downloads": 1000, "params_b": 1.0},
        {"id": "org/Popular-7B", "downloads": 2000, "params_b": 7.0},
        {"id": "org/Other-7B", "downloads": 1000, "params_b": 7.0},
        {"id": "org/TooBig-20B", "downloads": 5000, "params_b": 20.0},
    ]
    suggestions, note = advisor.suggest(
        8, limit=3, fetch_fn=lambda **_kwargs: rows
    )
    assert [row["id"] for row in suggestions] == [
        "org/Popular-7B",
        "org/Other-7B",
        "org/Small-1B",
    ]
    assert suggestions[0]["est_gb"] == 7.4
    assert "estimate" in note

    failed, error = advisor.suggest(
        8, fetch_fn=lambda **_kwargs: (_ for _ in ()).throw(OSError("offline"))
    )
    assert failed == []
    assert error == (
        "could not reach huggingface.co (offline); type a model id by hand"
    )


@pytest.mark.parametrize("log,word", [
    ("torch.OutOfMemoryError: CUDA out of memory", "ran out of memory"),
    ("ModuleNotFoundError: No module named 'peft'", "missing"),
    ("CUDA error: no kernel image is available for execution", "does not support"),
    ("OSError: ... is a gated repo", "login"),
])
def test_friendly_failure(log, word):
    assert word in preflight.friendly_failure(log)


def test_friendly_failure_unknown_is_empty():
    assert preflight.friendly_failure("all good") == ""


# ----------------------------------------------------------- model download

def test_download_status_counts_only_training_files_and_prefers_safetensors(tmp_path):
    cached_config = tmp_path / "config.json"
    cached_config.write_text("{}", encoding="utf-8")
    cached_tokenizer = tmp_path / "tokenizer.model"
    cached_tokenizer.write_bytes(b"tokenizer")
    siblings = [
        SimpleNamespace(rfilename="config.json", size=100),
        SimpleNamespace(rfilename="tokenizer.model", size=50),
        SimpleNamespace(rfilename="model.safetensors", size=1000),
        SimpleNamespace(rfilename="pytorch_model.bin", size=900),
        SimpleNamespace(rfilename="model.gguf", size=2000),
        SimpleNamespace(rfilename="onnx/model.onnx", size=3000),
        SimpleNamespace(rfilename="README.md", size=10),
    ]
    api = SimpleNamespace(model_info=lambda model, files_metadata: SimpleNamespace(
        siblings=siblings, sha="abc", gated=False
    ))

    def cache_lookup(_model, filename, revision):
        assert revision == "abc"
        return {"config.json": str(cached_config), "tokenizer.model": str(cached_tokenizer)}.get(filename)

    status = download_model.model_status("org/base", api=api, cache_lookup=cache_lookup)
    assert status == {
        "model": "org/base", "cached": False, "total_bytes": 1150,
        "missing_bytes": 1000, "gated": False, "error": None,
        "cache_dir": status["cache_dir"],
    }
    assert status["cache_dir"]


def test_download_status_turns_gated_failure_into_plain_action():
    class GatedRepoError(Exception):
        pass

    api = SimpleNamespace(model_info=lambda *_args, **_kwargs: (_ for _ in ()).throw(
        GatedRepoError("access denied")
    ))
    status = download_model.model_status("org/gated", api=api, cache_lookup=lambda *_a, **_k: None)
    assert status["gated"] is True
    assert "Accept its license" in status["error"] and "HF_TOKEN" in status["error"]


def test_local_download_status_distinguishes_missing_partial_and_complete_cache(tmp_path):
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    config = snapshot / "config.json"
    tokenizer = snapshot / "tokenizer.json"
    index = snapshot / "model.safetensors.index.json"
    shard_one = snapshot / "model-00001-of-00002.safetensors"
    shard_two = snapshot / "model-00002-of-00002.safetensors"
    for path, content in (
        (config, "{}"),
        (tokenizer, "{}"),
        (index, json.dumps({"weight_map": {"a": shard_one.name, "b": shard_two.name}})),
        (shard_one, "weights"),
    ):
        path.write_text(content, encoding="utf-8")

    def cache_info(repo_id="org/base"):
        files = tuple(
            SimpleNamespace(file_path=path, file_name=path.name)
            for path in (config, tokenizer, index, shard_one, shard_two)
        )
        revision = SimpleNamespace(snapshot_path=snapshot, files=files, refs=("main",))
        repo = SimpleNamespace(repo_id=repo_id, repo_type="model", revisions=(revision,))
        return SimpleNamespace(repos=(repo,))

    missing = download_model.local_model_status(
        "org/base", cache_info=SimpleNamespace(repos=frozenset())
    )
    partial = download_model.local_model_status("org/base", cache_info=cache_info())
    shard_two.write_text("weights", encoding="utf-8")
    complete = download_model.local_model_status("org/base", cache_info=cache_info())

    assert (missing["cached"], missing["cache_state"]) == (False, "missing")
    assert (partial["cached"], partial["cache_state"]) == (False, "partial")
    assert (complete["cached"], complete["cache_state"]) == (True, "cached")
    assert complete["total_bytes"] is None and complete["missing_bytes"] is None


def test_local_status_cli_forces_hugging_face_offline(monkeypatch, capsys):
    called = []
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    monkeypatch.setattr(
        download_model,
        "local_model_status",
        lambda model: called.append(model) or {"model": model, "cached": False},
    )
    monkeypatch.setattr(
        download_model,
        "model_status",
        lambda *_args: (_ for _ in ()).throw(AssertionError("network status must not run")),
    )

    assert download_model.main(["--local-status", "org/base"]) == 0
    assert called == ["org/base"]
    assert os.environ["HF_HUB_OFFLINE"] == "1"
    assert json.loads(capsys.readouterr().out)["model"] == "org/base"


def test_download_uses_exact_training_allowlist_and_default_cache(tmp_path, capsys):
    siblings = [
        SimpleNamespace(rfilename="config.json", size=10),
        SimpleNamespace(rfilename="model.safetensors", size=90),
        SimpleNamespace(rfilename="onnx/config.json", size=500),
        SimpleNamespace(rfilename="model.gguf", size=1000),
    ]
    api = SimpleNamespace(model_info=lambda model, files_metadata: SimpleNamespace(
        siblings=siblings, sha="revision", gated=False
    ))
    calls = []

    def snapshot(**kwargs):
        calls.append(kwargs)

    download_model.download_model(
        "org/base", api=api, cache_lookup=lambda *_args, **_kwargs: None, snapshot=snapshot
    )
    output = capsys.readouterr().out
    assert calls[0]["allow_patterns"] == ["config.json", "model.safetensors"]
    assert calls[0]["revision"] == "revision" and "cache_dir" not in calls[0]
    assert calls[0]["max_workers"] == 1
    assert output.endswith("DOWNLOAD DONE\n")


@pytest.mark.parametrize("named", [True, False], ids=["named-disabled", "unnamed-auto"])
def test_download_counts_reconstructed_bytes_once_when_hub_has_two_bars(capsys, named):
    siblings = [SimpleNamespace(rfilename="model.safetensors", size=100)]
    api = SimpleNamespace(model_info=lambda model, files_metadata: SimpleNamespace(
        siblings=siblings, sha="revision", gated=False
    ))
    observed = []

    def counters():
        output = capsys.readouterr().out
        observed.extend(
            int(line.split()[1].split("/")[0])
            for line in output.splitlines()
            if line.startswith("DOWNLOAD ") and line != "DOWNLOAD DONE"
        )

    def snapshot(**kwargs):
        progress = kwargs["tqdm_class"]
        transfer_kwargs = dict(
            desc="Downloading bytes",
            unit="B",
            total=100,
            initial=0,
        )
        reconstruct_kwargs = dict(
            desc="Reconstructing (incomplete total...)",
            unit="B",
            total=100,
            initial=20,
        )
        if named:
            transfer_kwargs.update(
                name="huggingface_hub.snapshot_download.transfer", disable=True
            )
            reconstruct_kwargs.update(
                name="huggingface_hub.snapshot_download", disable=True
            )
        transfer = progress(**transfer_kwargs)
        transfer.update(40)
        transfer.update(60)
        counters()
        assert observed == [0]

        reconstruct = progress(**reconstruct_kwargs)
        counters()
        reconstruct.update(30)
        counters()
        reconstruct.update(50)
        counters()

    download_model.download_model(
        "org/base", api=api, cache_lookup=lambda *_args, **_kwargs: None,
        snapshot=snapshot,
    )
    counters()
    assert observed == [0, 20, 50, 100, 100]
    assert all(done <= 100 for done in observed)
    assert 100 not in observed[:-2]


def test_download_progress_falls_back_to_older_single_unnamed_byte_bar(capsys):
    tracker = download_model._ByteProgress(10)
    tracker._emit()
    progress = tracker.tqdm_class(
        desc="Downloading model", unit="B", total=10, initial=3
    )
    progress.update(7)
    counters = [
        int(line.split()[1].split("/")[0])
        for line in capsys.readouterr().out.splitlines()
        if line.startswith("DOWNLOAD ")
    ]
    assert counters == [0, 3, 10]


# ---------------------------------------------------------------- GGUF export

def test_convert_and_quantize_commands():
    assert gguf_export.build_convert_cmd("py", "c.py", "m", "o.gguf") == [
        "py", "c.py", "m", "--outfile", "o.gguf", "--outtype", "f16"]
    assert gguf_export.build_quantize_cmd("q.exe", "a", "b", "q4_k_m") == [
        "q.exe", "a", "b", "Q4_K_M"]


def test_pinned_tag_matches_setup():
    assert gguf_export.PINNED_TAG == setup_env.LLAMA_CPP_TAG


def test_export_runs_convert_then_quantize_then_removes_f16(tmp_path):
    calls = []

    def runner(cmd, say, what):
        calls.append(cmd)
        if len(calls) == 1:  # the converter "writes" the f16 file
            Path(cmd[cmd.index("--outfile") + 1]).write_bytes(b"gguf")

    out = gguf_export.export_gguf(
        str(tmp_path / "merged"), str(tmp_path), "run1", "q4_k_m",
        python_exe="py", quantize_exe="q.exe", say=lambda m: None,
        runner=runner, converter="conv.py")
    assert out == str(tmp_path / "run1.q4_k_m.gguf")
    assert calls[0][:2] == ["py", "conv.py"]
    assert calls[1][0] == "q.exe" and calls[1][-1] == "Q4_K_M"
    assert not (tmp_path / "run1.f16.gguf").exists()


def test_export_f16_skips_quantize(tmp_path):
    calls = []
    out = gguf_export.export_gguf(
        "m", str(tmp_path), "r", "f16", python_exe="py", quantize_exe="",
        say=lambda m: None, runner=lambda c, s, w: calls.append(c), converter="c.py")
    assert out.endswith("r.f16.gguf") and len(calls) == 1


def test_export_without_quantizer_explains(tmp_path, monkeypatch):
    monkeypatch.setattr(gguf_export, "find_llama_quantize", lambda *a: "")
    with pytest.raises(gguf_export.ExportError, match="llama-quantize was not found"):
        gguf_export.export_gguf("m", str(tmp_path), "r", "q8_0", say=lambda m: None,
                                runner=lambda *a: None, converter="c.py")


def test_unknown_quant_rejected(tmp_path):
    with pytest.raises(gguf_export.ExportError):
        gguf_export.export_gguf("m", str(tmp_path), "r", "q2_k")


def _archive(members):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name, body in members.items():
            zf.writestr(name, body)
    return buf.getvalue()


def test_extract_converter_keeps_only_needed_files(tmp_path):
    data = _archive({
        "llama.cpp-b1/convert_hf_to_gguf.py": "print(1)",
        "llama.cpp-b1/gguf-py/gguf/__init__.py": "",
        "llama.cpp-b1/gguf-py/gguf/constants.py": "X=1",
        "llama.cpp-b1/conversion/__init__.py": "",
        "llama.cpp-b1/ggml/src/huge.c": "int x;",
        "llama.cpp-b1/gguf-py/tests/test_x.py": "",
        "llama.cpp-b1/../../evil.py": "bad",
    })
    script = gguf_export.extract_converter(data, str(tmp_path / "dest"))
    dest = Path(script).parent
    assert (dest / "gguf-py" / "gguf" / "constants.py").is_file()
    assert (dest / "conversion" / "__init__.py").is_file()
    assert not (dest / "ggml").exists()
    assert not (dest / "gguf-py" / "tests").exists()
    assert not (tmp_path / "evil.py").exists()


def test_extract_converter_requires_the_script(tmp_path):
    with pytest.raises(gguf_export.ExportError):
        gguf_export.extract_converter(_archive({"r/README.md": "x"}), str(tmp_path))


def test_ensure_converter_downloads_once(tmp_path):
    data = _archive({"r/convert_hf_to_gguf.py": "x", "r/gguf-py/gguf/__init__.py": "",
                     "r/conversion/__init__.py": ""})
    urls = []

    class Resp(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def opener(url, timeout):
        urls.append(url)
        return Resp(data)

    a = gguf_export.ensure_converter("b11406", root=str(tmp_path), opener=opener, say=lambda m: None)
    b = gguf_export.ensure_converter("b11406", root=str(tmp_path), opener=opener, say=lambda m: None)
    assert a == b and len(urls) == 1
    assert urls[0].endswith("/refs/tags/b11406")


def test_ensure_converter_network_failure_is_plain(tmp_path):
    def opener(url, timeout):
        raise OSError("offline")

    with pytest.raises(gguf_export.ExportError, match="internet connection"):
        gguf_export.ensure_converter("b1", root=str(tmp_path), opener=opener, say=lambda m: None)


def test_find_llama_quantize_in_data_dir(tmp_path, monkeypatch):
    exe = tmp_path / "bin" / "llama.cpp" / gguf_export._exe("llama-quantize")
    exe.parent.mkdir(parents=True)
    exe.write_bytes(b"")
    monkeypatch.setenv("LOCITIZE_DATA_DIR", str(tmp_path))
    monkeypatch.delenv("LOCITIZE_LLAMA_QUANTIZE", raising=False)
    assert gguf_export.find_llama_quantize() == str(exe)


# ------------------------------------------------------------- setup / plan

def test_finetune_feature_needs_cuda_torch_and_stack_without_docker():
    feat = setup_plan.FEATURES_BY_KEY["finetune"]
    assert {"finetune_torch_pip", "finetune_train_pip"} <= set(feat.requires)
    assert "need docker" not in feat.summary.lower()
    for key in feat.requires:
        assert "needs docker" not in setup_plan.REQUIREMENTS[key].detail.lower()


def test_torch_set_is_cuda_index_and_stack_is_pinned():
    torch_set = setup_env.PIP_SETS["finetune_torch_pip"]
    assert "https://download.pytorch.org/whl/cu128" in torch_set
    assert any(p.startswith("torch==") and "+cu128" in p for p in torch_set)
    stack = setup_env.PIP_SETS["finetune_train_pip"]
    assert all("==" in p for p in stack)
    assert not any(p.startswith("-") for p in stack)  # plain PyPI -> uv eligible
