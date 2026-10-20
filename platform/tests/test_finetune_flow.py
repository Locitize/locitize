"""The guided fine-tune flow: dataset review (finetune_dataset), the trainer
runner (finetune_jobs) and the four-step page (finetune_page).

Headless throughout: no GPU, no torch, no real trainer. The runner is exercised
with a stand-in interpreter and a faked Popen; the Qt tests use the offscreen
platform and skip cleanly where PySide6 is absent.
"""

from __future__ import annotations

import json
import os
import queue
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

import finetune_dataset as ftd
import finetune_jobs as jobs
from config import FineTuneConfig, Settings

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def chat(user, assistant, system=None):
    messages = [{"role": "system", "content": system}] if system else []
    return {"messages": messages + [{"role": "user", "content": user},
                                    {"role": "assistant", "content": assistant}]}


def _install(tmp_path: Path) -> Settings:
    """A settings tree whose fine-tune environment and data root exist on disk."""
    studio = tmp_path / "studio"
    (studio / "trainer").mkdir(parents=True)
    for script in ("train.py", "distill.py", "preflight.py", "build_dataset.py", "infer.py",
                   "download_model.py"):
        (studio / "trainer" / script).write_text("# stand-in\n", encoding="utf-8")
    python = studio / ".venv" / "Scripts" / "python.exe"
    python.parent.mkdir(parents=True)
    python.write_bytes(b"MZ")
    data = tmp_path / "data"
    data.mkdir()
    (data / "models.yaml").write_text(
        "models:\n"
        '  - id: "existing"\n'
        '    name: "Existing"\n'
        '    location: "/locitize-test/models/existing.gguf"\n'
        "    context_size: 8192\n"
        "    gpu_layers: 999\n"
        "    status: installed\n",
        encoding="utf-8",
    )
    cfg = FineTuneConfig(enabled=True, studio_dir=str(studio), python=str(python))
    return Settings(finetune=cfg, base_dir=tmp_path, data_dir=data)


# --------------------------------------------------------------------------- #
# finetune_dataset
# --------------------------------------------------------------------------- #


def test_parse_jsonl_skips_bad_lines_and_reports_them():
    rows, notes = ftd.parse_dataset(b'{"messages": []}\nnot json\n\n{"instruction": "a", "output": "b"}\n',
                                    "set.jsonl")
    assert len(rows) == 2
    assert notes == ["line 2 is not valid JSON and was skipped"]


def test_parse_json_accepts_list_and_data_wrapper():
    assert len(ftd.parse_dataset(json.dumps([chat("q", "a")]), "x.json")[0]) == 1
    assert len(ftd.parse_dataset(json.dumps({"data": [chat("q", "a")] * 2}), "x.json")[0]) == 2


def test_prompt_file_txt_ignores_blanks_and_tolerates_bom():
    prompts, malformed = ftd.parse_prompt_file(
        "\ufeff First prompt \n\nSecond prompt\r\n".encode("utf-8"), "prompts.txt"
    )
    assert prompts == ["First prompt", "Second prompt"]
    assert malformed == []


def test_prompt_file_jsonl_keeps_valid_rows_and_reports_malformed_lines():
    payload = (
        '{"prompt": "one"}\n'
        'not-json\n'
        '{"prompt": "  "}\n'
        '[]\n'
        '\n'
        '{"prompt": "two"}\n'
    ).encode()
    prompts, malformed = ftd.parse_prompt_file(payload, "prompts.jsonl")
    assert prompts == ["one", "two"]
    assert malformed == [2, 3, 4]
    assert ftd.format_line_numbers(malformed) == "3 malformed lines (2, 3, 4)"


def test_prompt_file_rejects_bad_utf8_no_valid_rows_and_oversize():
    with pytest.raises(ftd.PromptFileError, match="valid UTF-8"):
        ftd.parse_prompt_file(b"\xff", "p.txt")
    with pytest.raises(ftd.PromptFileError, match="no valid prompts.*malformed.*1"):
        ftd.parse_prompt_file(b'{"wrong": "field"}\n', "p.jsonl")
    with pytest.raises(ftd.PromptFileError, match="larger than"):
        ftd.parse_prompt_file(b"123456", "p.txt", max_bytes=5)


def test_load_prompt_file_applies_the_bound_before_reading(tmp_path):
    path = tmp_path / "large.txt"
    path.write_bytes(b"x" * 12)
    with pytest.raises(ftd.PromptFileError, match="limit"):
        ftd.load_prompt_file(path, max_bytes=10)


@pytest.mark.parametrize("payload", ['{"rows": 1}', "[]", "{bad"])
def test_parse_json_rejects_unusable_files(payload):
    with pytest.raises(ftd.DatasetError):
        ftd.parse_dataset(payload, "x.json")


def test_alpaca_rows_normalise_like_the_trainer():
    assert ftd.to_messages({"instruction": "Say hi", "input": "to Ann", "output": "Hi Ann"}) == [
        {"role": "system", "content": ftd.DEFAULT_SYSTEM_PROMPT},
        {"role": "user", "content": "Say hi\nto Ann"},
        {"role": "assistant", "content": "Hi Ann"},
    ]


@pytest.mark.parametrize("row, problem", [
    ({"text": "x"}, "not a chat row"),
    ({"messages": []}, "no messages"),
    ({"messages": [{"role": "bot", "content": "x"}]}, "unknown role"),
    (chat("q", "  "), "empty assistant message"),
    ({"messages": [{"role": "assistant", "content": "a"}]}, "no user message"),
    ({"messages": [{"role": "user", "content": "q"}]}, "does not end with an assistant reply"),
])
def test_rows_the_trainer_cannot_use_are_flagged(row, problem):
    assert ftd.review([row])[0]["problem"].startswith(problem)


def test_review_marks_later_duplicates_ignoring_case_and_spacing():
    records = ftd.review([chat("Hello", "Hi"), chat("hello ", "hi"), chat("Other", "x")])
    assert [r["duplicate"] for r in records] == [False, True, False]
    stats = ftd.summarize(records)
    assert (stats["total"], stats["usable"], stats["problems"], stats["duplicates"]) == (3, 3, 0, 1)


def test_long_rows_are_a_warning_not_a_problem():
    record = ftd.review([chat("q", "x" * (ftd.MAX_SEQ_TOKENS * ftd.CHARS_PER_TOKEN + 40))])[0]
    assert record["too_long"] and not record["problem"]


def test_curated_rows_drop_problems_duplicates_and_unticked_rows():
    records = ftd.review([chat("a", "1"), chat("a", "1"), {"bad": 1}, chat("b", "2"), chat("c", "3")])
    out = ftd.curated_rows(records, keep={3: False})
    assert [ftd.first_text(r["messages"], "user") for r in out] == ["a", "c"]
    assert len(ftd.curated_rows(records, drop_duplicates=False)) == 4


def test_curated_rows_apply_edits_and_system_prompt():
    records = ftd.review([chat("q", "old", system="s0"), chat("q2", "a2")])
    out = ftd.curated_rows(records, edits={0: ("q fixed", "new")}, system_prompt="Be kind.")
    assert out[0]["messages"] == [
        {"role": "system", "content": "Be kind."},
        {"role": "user", "content": "q fixed"},
        {"role": "assistant", "content": "new"},
    ]
    assert out[1]["messages"][0] == {"role": "system", "content": "Be kind."}


def test_editing_a_row_to_empty_drops_it_and_multi_turn_rows_ignore_edits():
    assert ftd.curated_rows(ftd.review([chat("q", "a")]), edits={0: ("q", "")}) == []
    row = {"messages": [{"role": "user", "content": "1"}, {"role": "assistant", "content": "2"},
                        {"role": "user", "content": "3"}, {"role": "assistant", "content": "4"}]}
    assert ftd.curated_rows(ftd.review([row]), edits={0: ("x", "y")}) == [row]


@pytest.mark.parametrize("rows, passes, expected", [
    (100, 3, 38),      # 300 examples / (2 x 4) per step
    (1, 1, ftd.MIN_STEPS),
    (10**6, 5, ftd.MAX_STEPS),
])
def test_recommended_steps(rows, passes, expected):
    assert ftd.recommended_steps(rows, passes, batch_size=2, grad_accum=4) == expected


def test_written_dataset_reads_back_as_messages(tmp_path):
    path = tmp_path / "out.jsonl"
    rows = ftd.curated_rows(ftd.review([{"instruction": "a", "output": "b"}]))
    ftd.write_jsonl(rows, str(path))
    again, notes = ftd.parse_dataset(path.read_bytes(), str(path))
    assert notes == [] and again == rows


# --------------------------------------------------------------------------- #
# finetune_jobs
# --------------------------------------------------------------------------- #


def test_environment_explains_what_is_missing(tmp_path):
    with pytest.raises(jobs.FineTuneUnavailable, match="not set up"):
        jobs.environment(None)
    off = Settings(finetune=FineTuneConfig(enabled=False), base_dir=tmp_path, data_dir=tmp_path)
    with pytest.raises(jobs.FineTuneUnavailable, match="turned off"):
        jobs.environment(off)
    missing = Settings(finetune=FineTuneConfig(enabled=True, studio_dir=str(tmp_path / "nope")),
                       base_dir=tmp_path, data_dir=tmp_path)
    with pytest.raises(jobs.FineTuneUnavailable, match="trainer was not found"):
        jobs.environment(missing)


def test_environment_finds_the_trainer_and_interpreter(tmp_path):
    settings = _install(tmp_path)
    python, trainer = jobs.environment(settings)
    assert python.name == "python.exe" and (trainer / "train.py").is_file()


def test_child_env_keeps_hugging_face_tokens_but_no_other_secrets(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "x")
    monkeypatch.setenv("GITHUB_TOKEN", "x")
    monkeypatch.setenv("HF_TOKEN", "hf")
    env = jobs.child_env()
    assert "OPENAI_API_KEY" not in env and "GITHUB_TOKEN" not in env
    assert env["HF_TOKEN"] == "hf" and env["PYTHONUNBUFFERED"] == "1"


def test_start_training_writes_meta_and_runs_train_py_in_the_finetune_env(tmp_path, monkeypatch):
    settings = _install(tmp_path)
    seen = {}

    class FakePopen:
        def __init__(self, argv, **kwargs):
            seen["argv"], seen["cwd"] = argv, kwargs["cwd"]
            self.pid = 1

        def poll(self):
            return None

    monkeypatch.setattr(jobs.subprocess, "Popen", FakePopen)
    job = jobs.start_training(settings, model="org/M-1B", dataset_path="d.jsonl", label="my model",
                              steps=40, learning_rate=2e-4, lora_r=16, batch_size=2, grad_accum=4,
                              quant="q4_k_m", default_system="Be kind.", dataset_rows=12,
                              revision="abc123")
    assert job.running() and job.name.startswith("my-model-")
    argv = seen["argv"]
    assert argv[0].endswith("python.exe") and argv[1].endswith("train.py")
    assert argv[argv.index("--max-steps") + 1] == "40" and argv[argv.index("--model") + 1] == "org/M-1B"
    assert argv[argv.index("--revision") + 1] == "abc123"
    meta = json.loads((job.log_path.parent / "run_meta.json").read_text(encoding="utf-8"))
    assert meta["base_model"] == "org/M-1B" and meta["dataset_rows"] == 12
    assert meta["base_model_revision"] == "abc123"
    assert job.log_path.parent.parent == Path(settings.data_dir) / "finetune" / "outputs"


def test_preflight_reads_the_json_line_and_reports_a_silent_environment(tmp_path, monkeypatch):
    settings = _install(tmp_path)
    answer = {"ok": True, "problems": [], "need_mb": 3000}

    def fake_run(argv, **kwargs):
        return subprocess.CompletedProcess(argv, 0, stdout="noise\n" + json.dumps(answer) + "\n", stderr="")

    monkeypatch.setattr(jobs.subprocess, "run", fake_run)
    assert jobs.preflight(settings, "org/M-1B")["need_mb"] == 3000
    monkeypatch.setattr(jobs.subprocess, "run",
                        lambda argv, **k: subprocess.CompletedProcess(argv, 1, stdout="", stderr="boom"))
    result = jobs.preflight(settings, "org/M-1B")
    assert not result["ok"] and "did not answer" in result["problems"][0]


def test_local_download_status_is_offline_and_uses_only_local_mode(tmp_path, monkeypatch):
    settings = _install(tmp_path)
    answer = {
        "model": "org/M-1B", "cached": False, "cache_state": "partial",
        "total_bytes": None, "missing_bytes": None, "gated": False, "error": None,
        "cache_dir": "",
        "revision": None,
    }

    def fake_run(argv, **kwargs):
        assert argv[-2:] == ["--local-status", "org/M-1B"]
        assert kwargs["timeout"] == jobs.DOWNLOAD_STATUS_TIMEOUT
        assert kwargs["env"]["HF_HUB_OFFLINE"] == "1"
        return subprocess.CompletedProcess(argv, 0, stdout=json.dumps(answer) + "\n", stderr="")

    monkeypatch.setattr(jobs.subprocess, "run", fake_run)
    assert jobs.local_download_status(settings, "org/M-1B") == answer


def test_download_status_reads_and_normalizes_the_json_line_and_logs_egress(
        tmp_path, monkeypatch):
    import egress_log

    settings = _install(tmp_path)
    answer = {
        "model": "org/M-1B", "cached": False, "total_bytes": 1500,
        "missing_bytes": 1000, "gated": False, "error": None,
        "cache_state": "missing",
        "cache_dir": "",
        "revision": "abc123",
    }
    recorded = []
    monkeypatch.setattr(
        egress_log,
        "record",
        lambda url: recorded.append((url, egress_log.current_reason())),
    )

    def fake_run(argv, **kwargs):
        assert argv[-2:] == ["--status", "org/M-1B"]
        assert kwargs["timeout"] == jobs.DOWNLOAD_STATUS_TIMEOUT
        return subprocess.CompletedProcess(argv, 0, stdout="noise\n" + json.dumps(answer) + "\n", stderr="")

    monkeypatch.setattr(jobs.subprocess, "run", fake_run)
    assert jobs.download_status(settings, "org/M-1B") == answer
    assert recorded == [
        ("https://huggingface.co/api/models/org/M-1B", "finetune-model-size")
    ]

    monkeypatch.setattr(
        jobs.subprocess, "run",
        lambda argv, **kwargs: subprocess.CompletedProcess(argv, 1, stdout="", stderr="plain failure"),
    )
    assert jobs.download_status(settings, "org/M-1B")["error"] == "plain failure"


def test_download_status_does_not_log_egress_without_an_environment(
        tmp_path, monkeypatch):
    import egress_log

    settings = Settings(
        finetune=FineTuneConfig(enabled=True, studio_dir=str(tmp_path / "missing")),
        base_dir=tmp_path,
        data_dir=tmp_path,
    )
    recorded = []
    monkeypatch.setattr(egress_log, "record", lambda url: recorded.append(url))

    result = jobs.download_status(settings, "org/M-1B")

    assert result["error"]
    assert recorded == []


def test_start_download_uses_finetune_python_and_outputs_log(tmp_path, monkeypatch):
    import egress_log

    settings = _install(tmp_path)
    seen = {}

    class FakePopen:
        pid = 11

        def __init__(self, argv, **kwargs):
            seen["argv"], seen["cwd"] = argv, kwargs["cwd"]

        def poll(self):
            return None

    monkeypatch.setattr(jobs.subprocess, "Popen", FakePopen)
    recorded = []
    monkeypatch.setattr(
        egress_log,
        "record",
        lambda url: recorded.append((url, egress_log.current_reason())),
    )
    job = jobs.start_download(settings, "org/M-1B", revision="abc123")

    assert job.kind == "download" and job.running()
    assert seen["argv"][-4:] == ["--download", "org/M-1B", "--revision", "abc123"]
    assert seen["argv"][1].endswith("download_model.py")
    assert job.log_path.name == "download.log"
    assert job.log_path.parent.parent.name == "_downloads"
    assert job.meta == {"model": "org/M-1B", "revision": "abc123"}
    assert recorded == [
        ("https://huggingface.co/org/M-1B", "finetune-model-download")
    ]


def test_progress_reads_stage_steps_losses_and_outcome():
    log = ("[STATUS] Loading base model org/M (4-bit)\n"
           "{'loss': 4.31, 'epoch': 0.1}\n 3/40 [00:01<00:10]\r 7/40 [00:02<00:09]\n"
           "{'loss': '1.07', 'epoch': 1.0}\n[STATUS] Starting training\n")
    state = jobs.progress(log)
    assert state["stage"] == "Training" and (state["current"], state["total"]) == (7, 40)
    assert state["losses"] == [4.31, 1.07] and not state["done"] and not state["failed"]
    assert jobs.progress(log + "[STATUS] DONE\n")["done"]
    failed = jobs.progress(log + "Traceback (most recent call last):\n")
    assert failed["failed"] and failed["stage"] == "Stopped with an error"
    assert jobs.progress("[DISTILL] 3/9\n")["current"] == 3


def test_download_progress_reads_bytes_done_and_errors():
    state = jobs.download_progress("DOWNLOAD 0/100\nDOWNLOAD 25/100\n")
    assert (state["current"], state["total"], state["fraction"]) == (25, 100, 0.25)
    assert not state["done"] and not state["failed"]
    assert jobs.download_progress("DOWNLOAD 100/100\nDOWNLOAD DONE\n")["done"]
    failed = jobs.download_progress("DOWNLOAD 10/100\nDOWNLOAD ERROR: token needed\n")
    assert failed["failed"] and failed["error"] == "token needed"


def test_read_new_returns_only_appended_text(tmp_path):
    log = tmp_path / "train.log"
    log.write_bytes(b"one\n")
    text, offset = jobs.read_new(log, 0)
    with open(log, "ab") as f:
        f.write(b"two\n")
    more, _ = jobs.read_new(log, offset)
    assert text == "one\n" and more == "two\n"


def test_read_new_advances_past_bytes_appended_between_eof_check_and_read(tmp_path, monkeypatch):
    log = tmp_path / "growing.log"
    log.write_bytes(b"one\n")
    real_open = open
    grew = False

    class GrowingReader:
        def __init__(self, handle):
            self.handle = handle

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.handle.close()

        def seek(self, *args):
            return self.handle.seek(*args)

        def tell(self):
            return self.handle.tell()

        def read(self, *args):
            nonlocal grew
            if not grew:
                grew = True
                with real_open(log, "ab") as writer:
                    writer.write(b"two\n")
            return self.handle.read(*args)

    def growing_open(path, mode="r", *args, **kwargs):
        handle = real_open(path, mode, *args, **kwargs)
        if Path(path) == log and mode == "rb":
            return GrowingReader(handle)
        return handle

    monkeypatch.setattr(jobs, "open", growing_open, raising=False)
    text, offset = jobs.read_new(log, 0)
    more, next_offset = jobs.read_new(log, offset)

    assert text == "one\ntwo\n"
    assert offset == log.stat().st_size
    assert more == "" and next_offset == offset


def test_history_meta_reads_only_one_byte_beyond_its_limit(tmp_path, monkeypatch):
    meta_path = tmp_path / jobs.RUN_META
    meta_path.write_bytes(b"{" + b"x" * jobs.HISTORY_META_BYTES)
    real_path_open = Path.open
    read_sizes = []

    class ReadProbe:
        def __init__(self, handle):
            self.handle = handle

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.handle.close()

        def read(self, size=-1):
            read_sizes.append(size)
            return self.handle.read(size)

    def probed_open(path, *args, **kwargs):
        handle = real_path_open(path, *args, **kwargs)
        return ReadProbe(handle) if path == meta_path else handle

    monkeypatch.setattr(Path, "open", probed_open)
    meta, error = jobs._history_meta(meta_path)

    assert meta == {} and error == "metadata is too large to display"
    assert read_sizes == [jobs.HISTORY_META_BYTES + 1]


def test_run_history_scans_status_meta_losses_and_never_guesses_running(tmp_path):
    settings = _install(tmp_path)
    root = Path(settings.data_dir) / "finetune" / "outputs"
    completed = root / "completed-run"
    completed.mkdir(parents=True)
    (completed / jobs.RUN_META).write_text(json.dumps({
        "base_model": "org/base", "dataset_path": "/synthetic/data/set.jsonl", "max_steps": 20,
    }), encoding="utf-8")
    (completed / jobs.TRAIN_LOG).write_text(
        "x" * (jobs.HISTORY_LOG_TAIL_BYTES + 200)
        + "\n{'loss': 2.5}\n{'loss': '1.25'}\n[STATUS] DONE\n",
        encoding="utf-8",
    )
    failed = root / "failed-run"
    failed.mkdir()
    (failed / jobs.TRAIN_LOG).write_text(
        "Traceback (most recent call last):\nboom\n", encoding="utf-8"
    )
    incomplete = root / "incomplete-run"
    incomplete.mkdir()
    (incomplete / jobs.TRAIN_LOG).write_text("[STATUS] Starting training\n", encoding="utf-8")
    corrupt = root / "corrupt-meta"
    corrupt.mkdir()
    (corrupt / jobs.RUN_META).write_text("{bad", encoding="utf-8")
    (root / "_distill" / "not-training").mkdir(parents=True)
    (root / "_downloads" / "not-training").mkdir(parents=True)

    rows = jobs.scan_training_runs(settings)
    by_name = {row["name"]: row for row in rows}
    assert set(by_name) == {"completed-run", "failed-run", "incomplete-run", "corrupt-meta"}
    assert by_name["completed-run"]["status"] == "completed"
    assert by_name["completed-run"]["base_model"] == "org/base"
    assert by_name["completed-run"]["dataset_path"] == "/synthetic/data/set.jsonl"
    assert by_name["completed-run"]["losses"] == [2.5, 1.25]
    assert len(by_name["completed-run"]["log_tail"]) <= jobs.HISTORY_LOG_TAIL_BYTES
    assert by_name["failed-run"]["status"] == "failed"
    assert by_name["incomplete-run"]["status"] == "incomplete"
    assert by_name["corrupt-meta"]["meta"] == {}
    assert "not valid JSON" in by_name["corrupt-meta"]["meta_error"]
    assert "running" not in {row["status"] for row in rows}


def test_run_history_reports_running_only_for_the_owned_live_job(tmp_path):
    settings = _install(tmp_path)
    run = Path(settings.data_dir) / "finetune" / "outputs" / "owned-run"
    run.mkdir(parents=True)
    log = run / jobs.TRAIN_LOG
    log.write_text("[STATUS] Starting training\n", encoding="utf-8")

    class LiveProcess:
        def poll(self):
            return None

    job = jobs.Job("train", run.name, log, LiveProcess())
    assert jobs.scan_training_runs(settings, live_job=job)[0]["status"] == "running"
    assert jobs.scan_training_runs(settings)[0]["status"] == "incomplete"


def _finished_run(settings, name="support-tone-20261019-101500"):
    run = Path(settings.data_dir) / "finetune" / "outputs" / name
    (run / "merged_model").mkdir(parents=True)
    (run / f"{name}.q4_k_m.gguf").write_bytes(b"GGUF" + b"0" * 64)
    (run / "run_meta.json").write_text(json.dumps({"base_model": "org/M-1B"}), encoding="utf-8")
    return run


def test_save_model_places_the_file_and_registers_it(tmp_path):
    settings = _install(tmp_path)
    run = _finished_run(settings)
    folder = tmp_path / "picked" / "models"
    dest, message = jobs.save_model(settings, run_dir=run, gguf_name=f"{run.name}.q4_k_m.gguf",
                                    folder=str(folder), name="Support Tone", register=True,
                                    model_id="support-tone", context_size=4096, prompt="Be kind.")
    assert dest == folder / "Support-Tone.Q4_K_M.gguf" and dest.read_bytes().startswith(b"GGUF")
    assert "added 'Support Tone'" in message
    import yaml

    rows = yaml.safe_load((Path(settings.data_dir) / "models.yaml").read_text(encoding="utf-8"))["models"]
    row = rows[-1]
    assert (row["id"], row["location"], row["context_size"]) == ("support-tone", str(dest), 4096)
    assert "org/M-1B" in row["description"]
    with pytest.raises(ValueError, match="Choose a new id or tick"):
        jobs.save_model(settings, run_dir=run, gguf_name=f"{run.name}.q4_k_m.gguf", folder=str(folder),
                        name="Support Tone", register=True, model_id="support-tone", context_size=4096,
                        prompt="")


def test_duplicate_id_is_refused_before_a_colliding_file_can_be_touched(tmp_path):
    settings = _install(tmp_path)
    run = _finished_run(settings)
    folder = tmp_path / "models"
    folder.mkdir()
    existing = folder / "Existing.Q4_K_M.gguf"
    existing.write_bytes(b"irreplaceable existing weights")

    with pytest.raises(ValueError, match="Replace existing model"):
        jobs.save_model(
            settings,
            run_dir=run,
            gguf_name=f"{run.name}.q4_k_m.gguf",
            folder=str(folder),
            name="Existing",
            register=True,
            model_id="existing",
            context_size=4096,
            prompt="",
        )

    assert existing.read_bytes() == b"irreplaceable existing weights"
    assert not (folder / "Existing-2.Q4_K_M.gguf").exists()


def test_filename_collision_is_versioned_and_existing_weights_survive(tmp_path):
    settings = _install(tmp_path)
    run = _finished_run(settings)
    folder = tmp_path / "models"
    folder.mkdir()
    existing = folder / "Support-Tone.Q4_K_M.gguf"
    existing.write_bytes(b"old model")

    dest, _message = jobs.save_model(
        settings,
        run_dir=run,
        gguf_name=f"{run.name}.q4_k_m.gguf",
        folder=str(folder),
        name="Support Tone",
        register=True,
        model_id="support-tone-v2",
        context_size=4096,
        prompt="Be kind.",
    )

    assert dest == folder / "Support-Tone-2.Q4_K_M.gguf"
    assert dest.read_bytes().startswith(b"GGUF")
    assert existing.read_bytes() == b"old model"


def test_registration_failure_removes_only_the_file_created_by_that_save(
        tmp_path, monkeypatch):
    import config

    settings = _install(tmp_path)
    run = _finished_run(settings)
    folder = tmp_path / "models"
    folder.mkdir()
    existing = folder / "Support-Tone.Q4_K_M.gguf"
    existing.write_bytes(b"keep me")
    monkeypatch.setattr(
        config,
        "append_model_entry",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            config.RegistryWriteError("registry refused")
        ),
    )

    with pytest.raises(ValueError, match="new saved file was removed"):
        jobs.save_model(
            settings,
            run_dir=run,
            gguf_name=f"{run.name}.q4_k_m.gguf",
            folder=str(folder),
            name="Support Tone",
            register=True,
            model_id="new-id",
            context_size=4096,
            prompt="",
        )

    assert existing.read_bytes() == b"keep me"
    assert not (folder / "Support-Tone-2.Q4_K_M.gguf").exists()


def test_replace_model_changes_only_registry_row_and_keeps_old_gguf(tmp_path):
    import yaml

    settings = _install(tmp_path)
    old = tmp_path / "old-existing.gguf"
    old.write_bytes(b"old weights")
    registry = Path(settings.data_dir) / "models.yaml"
    registry.write_text(
        registry.read_text(encoding="utf-8").replace(
            "/locitize-test/models/existing.gguf", str(old).replace("\\", "/")
        ),
        encoding="utf-8",
    )
    reports = Path(settings.data_dir) / "reports"
    reports.mkdir()
    ceilings = reports / "ctx_ceilings.json"
    ceilings.write_text(json.dumps({
        "existing": {"best": 8192, "verdict": "old weights"},
        "other": {"best": 4096},
    }), encoding="utf-8")
    run = _finished_run(settings)

    dest, message = jobs.save_model(
        settings,
        run_dir=run,
        gguf_name=f"{run.name}.q4_k_m.gguf",
        folder=str(tmp_path / "models"),
        name="Existing v2",
        register=True,
        model_id="existing",
        context_size=16384,
        prompt="First line\nSecond line",
        description="Custom description",
        notes="Custom notes\nTwo lines",
        gpu_layers=41,
        replace=True,
    )

    assert old.read_bytes() == b"old weights"
    assert dest.is_file() and dest != old
    assert "replaced 'Existing v2'" in message
    row = yaml.safe_load(registry.read_text(encoding="utf-8"))["models"][0]
    assert row["id"] == "existing" and row["location"] == str(dest)
    assert row["description"] == "Custom description"
    assert row["notes"] == "Custom notes\nTwo lines"
    assert row["recommended_prompt"] == "First line\nSecond line"
    assert (row["context_size"], row["gpu_layers"]) == (16384, 41)
    assert json.loads(ceilings.read_text(encoding="utf-8")) == {
        "other": {"best": 4096}
    }


def test_registry_model_entry_treats_a_missing_registry_as_no_match(tmp_path):
    settings = _install(tmp_path)
    (Path(settings.data_dir) / "models.yaml").unlink()

    assert jobs.registry_model_entry(settings, "first-model") is None


def test_registry_model_entry_still_rejects_corrupt_yaml(tmp_path):
    settings = _install(tmp_path)
    (Path(settings.data_dir) / "models.yaml").write_text(
        "models: [unterminated\n", encoding="utf-8"
    )

    with pytest.raises(ValueError, match="Could not read your model list"):
        jobs.registry_model_entry(settings, "model")


def test_save_model_refuses_a_relative_folder_and_can_skip_the_list(tmp_path):
    settings = _install(tmp_path)
    run = _finished_run(settings)
    with pytest.raises(ValueError, match="full folder path"):
        jobs.save_model(settings, run_dir=run, gguf_name=f"{run.name}.q4_k_m.gguf", folder="models",
                        name="x", register=False, model_id="x", context_size=4096, prompt="")
    dest, message = jobs.save_model(settings, run_dir=run, gguf_name=f"{run.name}.q4_k_m.gguf",
                                    folder=str(tmp_path / "plain"), name="x", register=False, model_id="x",
                                    context_size=4096, prompt="")
    assert dest.is_file() and message == f"Saved {dest}"


def test_place_file_falls_back_to_copy_when_linking_fails(tmp_path, monkeypatch):
    src = tmp_path / "m.gguf"
    src.write_bytes(b"data")

    def no_link(a, b):
        raise OSError("cross-device link")

    monkeypatch.setattr(jobs.os, "link", no_link)
    assert jobs.place_file(src, tmp_path / "other", "copy.gguf").read_bytes() == b"data"


# --------------------------------------------------------------------------- #
# The page (offscreen Qt)
# --------------------------------------------------------------------------- #


class _Window:
    """The two panel helpers FineTuneFlow borrows from the main window."""

    @staticmethod
    def _panel():
        from PySide6 import QtWidgets

        frame = QtWidgets.QFrame()
        return frame, QtWidgets.QVBoxLayout(frame)

    @staticmethod
    def _section(layout, title, hint=""):
        from PySide6 import QtWidgets

        layout.addWidget(QtWidgets.QLabel(title))


@pytest.fixture()
def flow_factory(monkeypatch):
    pytest.importorskip("PySide6")
    from PySide6 import QtWidgets

    import finetune_page

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    # No real preflight children from the page's background checks.
    monkeypatch.setattr(jobs, "preflight", lambda *a, **k: {"ok": True, "need_mb": 2048, "free_mb": 9000,
                                                            "gpu_name": "Test GPU"})
    monkeypatch.setattr(jobs, "local_download_status", lambda _settings, model: {
        "model": model, "cached": False, "cache_state": "missing",
        "total_bytes": None, "missing_bytes": None, "gated": False, "error": None,
    })
    monkeypatch.setattr(jobs, "download_status", lambda _settings, model: {
        "model": model, "cached": False, "cache_state": "missing",
        "total_bytes": 2 * 1024 ** 3, "missing_bytes": 2 * 1024 ** 3,
        "gated": False, "error": None,
    })

    def make(settings):
        controller = SimpleNamespace(_settings=settings, result_q=queue.Queue())
        return finetune_page.FineTuneFlow(_Window(), controller, None)

    yield make, app


def test_finetune_questions_use_the_desktop_plain_text_helper(
        flow_factory, monkeypatch):
    from PySide6 import QtWidgets
    import desktop
    import finetune_page

    _make, _app = flow_factory
    shown = {}

    class Box:
        def exec(self):
            return QtWidgets.QMessageBox.StandardButton.No

    def plain(parent, title, text, **kwargs):
        shown.update(parent=parent, title=title, text=text, kwargs=kwargs)
        return Box()

    monkeypatch.setattr(desktop, "plain_message_box", plain)
    monkeypatch.setattr(
        QtWidgets.QMessageBox,
        "question",
        staticmethod(lambda *_a, **_k: (_ for _ in ()).throw(
            AssertionError("static question bypassed plain rendering")
        )),
    )

    answer = finetune_page._plain_question(None, "Title", "<b>literal</b>")

    assert answer == QtWidgets.QMessageBox.StandardButton.No
    assert shown["text"] == "<b>literal</b>"
    assert shown["kwargs"]["default"] == QtWidgets.QMessageBox.StandardButton.No


def test_page_without_an_install_says_why_and_still_allows_browsing(flow_factory):
    make, _app = flow_factory
    flow = make(None)
    assert flow.unavailable.isVisibleTo(flow)
    assert "not set up" in flow.unavailable.text()
    assert flow.step == 1
    assert [b.isEnabled() for b in flow.step_buttons] == [True, False, False, False]
    for button in (flow.suggest_models, flow.download_button, flow.model_next):
        assert not button.isEnabled()
        assert "not installed" in button.toolTip()


def test_page_walks_model_then_dataset_review_then_train(tmp_path, flow_factory):
    make, app = flow_factory
    settings = _install(tmp_path)
    flow = make(settings)
    flow.custom_model.setText("org/Tiny-0.5B")
    flow.model_next.click()
    assert flow.step == 2 and flow.model == "org/Tiny-0.5B"

    rows = [chat(f"q{i}", f"a{i}") for i in range(5)] + [chat("q0", "a0"), {"bad": True}]
    flow.show_rows(rows, "file test.jsonl", "my-set")
    assert flow.review.isVisibleTo(flow) and flow.review_table.rowCount() == 7
    assert "5 rows will be used" in flow.final_count.text()
    keep = flow.review_table.item(1, 0)
    from PySide6 import QtCore

    keep.setCheckState(QtCore.Qt.CheckState.Unchecked)
    app.processEvents()
    assert "4 rows will be used" in flow.final_count.text()

    flow.dataset_next.click()
    assert flow.step == 3
    saved = Path(flow.dataset_path)
    assert saved == Path(settings.data_dir) / "finetune" / "datasets" / "my-set.jsonl"
    assert len(saved.read_text(encoding="utf-8").splitlines()) == 4
    assert flow.train_name.text() == "my-set"
    assert flow.steps.value() == ftd.recommended_steps(4, 3, 2, 4)
    assert not flow.step_buttons[3].isEnabled()


def test_page_build_and_model_change_only_run_local_download_status(
        tmp_path, flow_factory, monkeypatch):
    make, app = flow_factory
    local_calls = []
    monkeypatch.setattr(
        jobs,
        "local_download_status",
        lambda _settings, model: (
            local_calls.append(model)
            or {
                "model": model, "cached": False, "cache_state": "missing",
                "total_bytes": None, "missing_bytes": None,
                "gated": False, "error": None,
            }
        ),
    )
    monkeypatch.setattr(
        jobs,
        "download_status",
        lambda *_args: (_ for _ in ()).throw(
            AssertionError("page build and model changes must not use network status")
        ),
    )

    flow = make(_install(tmp_path))
    assert flow._pool.waitForDone(2000)
    app.processEvents()
    first = flow.model_combo.currentData()
    flow.custom_model.setText("org/Changed-1B")
    flow._on_model_choice()
    assert flow._pool.waitForDone(2000)
    app.processEvents()

    assert local_calls == [first, "org/Changed-1B"]
    assert flow.download_status_label.text() == "Not downloaded yet"


def test_download_button_states_for_cached_missing_and_running(tmp_path, flow_factory):
    make, _app = flow_factory
    flow = make(_install(tmp_path))
    model = flow.chosen_model()

    flow._on_local_download_status(model, {
        "model": model, "cached": False, "total_bytes": 3 * 1024 ** 3,
        "missing_bytes": 2 * 1024 ** 3, "gated": False, "error": None,
    }, None)
    assert flow.download_button.isEnabled()
    assert flow.download_status_label.text() == "Not downloaded yet"

    flow._on_local_download_status(model, {
        "model": model, "cached": True, "total_bytes": 3 * 1024 ** 3,
        "missing_bytes": 0, "gated": False, "error": None,
    }, None)
    assert not flow.download_button.isEnabled()
    assert flow.download_status_label.text() == "Downloaded"

    class LiveProcess:
        def poll(self):
            return None

    flow.download_job = jobs.Job("download", "run", tmp_path / "download.log", LiveProcess(),
                                 meta={"model": model})
    flow._download_model = model
    flow._refresh_download_controls()
    assert not flow.download_button.isEnabled()
    assert flow.download_stop.isEnabled()
    assert flow.download_progress.isVisibleTo(flow)

    flow.download_job = None
    flow.train_job = jobs.Job("train", "train", tmp_path / "train.log", LiveProcess())
    flow._refresh_download_controls()
    assert not flow.download_button.isEnabled()
    assert "Training is running" in flow.download_status_label.text()


def test_download_confirmation_cancel_does_not_start_a_job(tmp_path, flow_factory, monkeypatch):
    from PySide6 import QtWidgets
    import finetune_page

    make, _app = flow_factory
    flow = make(_install(tmp_path))
    model = flow.chosen_model()
    flow._on_local_download_status(model, {
        "model": model, "cached": False, "total_bytes": None,
        "missing_bytes": None, "gated": False, "error": None,
    }, None)
    remote_calls = []

    def remote_status(_settings, selected):
        remote_calls.append(selected)
        return {
            "model": selected, "cached": False, "total_bytes": 14 * 1024 ** 3,
            "missing_bytes": 14 * 1024 ** 3, "gated": False, "error": None,
        }

    checking_states = []

    def run_now(fn, done):
        checking_states.append(
            (flow.download_status_label.text(), flow.download_button.isEnabled())
        )
        done(fn(), None)

    monkeypatch.setattr(jobs, "download_status", remote_status)
    flow._run = run_now
    shown = {}

    def decline(_parent, title, text, *_args):
        shown.update(title=title, text=text)
        return QtWidgets.QMessageBox.StandardButton.No

    monkeypatch.setattr(finetune_page, "_plain_question", decline)
    monkeypatch.setattr(
        jobs, "start_download",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("download must not start")),
    )
    flow.download_button.click()

    assert remote_calls == [model]
    assert checking_states == [("Checking size on huggingface.co...", False)]
    assert shown["title"] == "Download base model?"
    assert model in shown["text"] and "14.0 GB" in shown["text"]
    assert flow.download_job is None


def test_download_size_failure_does_not_confirm_or_start(
        tmp_path, flow_factory, monkeypatch):
    import finetune_page

    make, _app = flow_factory
    flow = make(_install(tmp_path))
    model = flow.chosen_model()
    flow._on_local_download_status(model, {
        "model": model, "cached": False, "total_bytes": None,
        "missing_bytes": None, "gated": False, "error": None,
    }, None)
    monkeypatch.setattr(jobs, "download_status", lambda *_args: {
        "model": model, "cached": False, "total_bytes": None,
        "missing_bytes": None, "gated": True,
        "error": "This model is gated or unauthorized.",
    })
    monkeypatch.setattr(
        finetune_page,
        "_plain_question",
        lambda *_args: (_ for _ in ()).throw(AssertionError("must not confirm")),
    )
    monkeypatch.setattr(
        jobs,
        "start_download",
        lambda *_args: (_ for _ in ()).throw(AssertionError("must not start")),
    )
    flow._run = lambda fn, done: done(fn(), None)

    flow.download_button.click()

    assert flow.download_job is None
    assert flow.download_button.isEnabled()
    assert "gated or unauthorized" in flow.download_status_label.text()


def test_download_job_finishes_rechecks_status_and_shutdown_stops_live_job(
        tmp_path, flow_factory, monkeypatch):
    make, _app = flow_factory
    flow = make(_install(tmp_path))
    model = flow.chosen_model()
    log = tmp_path / "download.log"
    log.write_text("DOWNLOAD 100/100\nDOWNLOAD DONE\n", encoding="utf-8")

    class FinishedProcess:
        pid = 19

        def poll(self):
            return 0

    flow.download_job = jobs.Job("download", "run", log, FinishedProcess(), meta={"model": model})
    flow._download_model = model
    checked = []
    flow._request_local_download_status = (
        lambda selected, force=False: checked.append((selected, force))
    )
    assert flow._poll_download() is False
    assert flow.download_job is None and checked == [(model, True)]
    assert flow.download_progress.value() == 1000

    class LiveProcess:
        pid = 20

        def poll(self):
            return None

    live = jobs.Job("download", "live", log, LiveProcess(), meta={"model": model})
    flow.download_job = live
    stopped = []
    monkeypatch.setattr(jobs, "stop", lambda job: stopped.append(job))
    flow.shutdown()
    assert stopped == [live]


def test_failed_download_keeps_failure_visible_without_status_overwrite(
        tmp_path, flow_factory):
    make, _app = flow_factory
    flow = make(_install(tmp_path))
    model = flow.chosen_model()
    log = tmp_path / "download.log"
    log.write_text("DOWNLOAD 10/100\nDOWNLOAD ERROR: connection lost\n", encoding="utf-8")

    class FailedProcess:
        def poll(self):
            return 1

    flow.download_job = jobs.Job(
        "download", "run", log, FailedProcess(), meta={"model": model}
    )
    checked = []
    flow._request_local_download_status = lambda *_args, **_kwargs: checked.append(True)

    assert flow._poll_download() is False
    assert checked == []
    assert flow.download_status_label.text() == "connection lost (exit code 1)"
    assert flow.download_button.isEnabled()


def test_document_qa_gate_tracks_running_model_without_auto_checking(tmp_path, flow_factory):
    make, app = flow_factory
    flow = make(_install(tmp_path))
    running = {"id": None}
    flow._gc.running_model_id = lambda: running["id"]

    flow.source_combo.setCurrentIndex(1)
    app.processEvents()
    assert not flow.doc_qa.isEnabled() and not flow.doc_qa.isChecked()
    assert "Start a model" in flow.doc_qa_hint.text()
    assert not flow.doc_pairs.isEnabled()

    running["id"] = "ready-model"
    flow.source_combo.setCurrentIndex(0)
    flow.source_combo.setCurrentIndex(1)
    app.processEvents()
    assert flow.doc_qa.isEnabled() and not flow.doc_qa.isChecked()
    flow.doc_qa.setChecked(True)
    assert flow.doc_pairs.isEnabled()

    running["id"] = None
    flow._refresh_doc_qa_gate()
    assert not flow.doc_qa.isEnabled() and not flow.doc_qa.isChecked()


def test_distill_log_is_bounded_auto_opens_on_failure_and_buttons_keep_last_run(
        tmp_path, flow_factory, monkeypatch):
    import finetune_page

    make, _app = flow_factory
    settings = _install(tmp_path)
    flow = make(settings)
    run = Path(settings.data_dir) / "finetune" / "outputs" / "_distill" / "d-run"
    run.mkdir(parents=True)
    log = run / "distill.log"
    out = run / "distilled_dataset.jsonl"

    class FailedProcess:
        pid = 7

        def poll(self):
            return 2

    job = jobs.Job("distill", "d-run", log, FailedProcess(), out_path=out)
    monkeypatch.setattr(jobs, "start_distill", lambda *args, **kwargs: job)
    flow.distill_prompts.setPlainText("one")
    flow.distill_start.click()
    log.write_text(
        "x" * (finetune_page.DISTILL_LOG_MEMORY_CHARS + 100)
        + "\nTraceback (most recent call last):\nboom\n",
        encoding="utf-8",
    )
    flow._poll()

    assert flow.distill_job is None
    assert flow.distill_log_toggle.isChecked()
    assert not flow.distill_log_view.isHidden()
    assert "Traceback" in flow.distill_log_view.toPlainText()
    assert len(flow.distill_log_view.toPlainText()) <= finetune_page.DISTILL_LOG_VIEW_CHARS
    assert flow.distill_open_log.isEnabled() and flow.distill_open_folder.isEnabled()

    opened = []
    flow._open_path = lambda path: opened.append(Path(path))
    flow._open_folder = lambda path: opened.append(Path(path))
    flow.distill_open_log.click()
    flow.distill_open_folder.click()
    assert opened == [log, run]


def test_run_history_section_loads_selected_details_off_thread_contract(tmp_path, flow_factory, monkeypatch):
    make, _app = flow_factory
    flow = make(_install(tmp_path))
    run = tmp_path / "history-run"
    run.mkdir()
    scanned = [{
        "name": "history-run", "path": str(run), "status": "completed",
        "mtime": 1.0, "modified_display": "2026-10-19 10:00",
        "meta": {"max_steps": 20}, "meta_error": "", "base_model": "org/base",
        "dataset_path": "/synthetic/data/set.jsonl", "log_path": str(run / "train.log"),
        "log_tail": "{'loss': 2.0}\n[STATUS] DONE\n", "losses": [2.0, 1.0],
    }]
    monkeypatch.setattr(jobs, "scan_training_runs", lambda settings, live_job=None: scanned)
    calls = []
    flow._run = lambda fn, done: (calls.append("background"), done(fn(), None))
    flow.history_toggle.setChecked(True)

    assert calls == ["background"]
    assert flow.history_list.count() == 1
    assert "completed" in flow.history_list.item(0).text()
    assert "org/base" in flow.history_details.text()
    assert "/synthetic/data/set.jsonl" in flow.history_details.text()
    assert "max_steps" in flow.history_details.text()
    assert "2 points" in flow.history_loss_title.text()
    assert flow.history_open_folder.isEnabled()


def test_page_opens_a_finished_run_at_the_save_step_and_saves_it(tmp_path, flow_factory):
    make, _app = flow_factory
    settings = _install(tmp_path)
    run = _finished_run(settings)
    flow = make(settings)
    flow.open_run(run)
    assert flow.step == 4
    assert flow.save_name.text() == "support-tone" and flow.save_id.text() == "support-tone"
    assert flow.save_folder.text() == str(Path(settings.data_dir) / "models")
    assert "Support" not in flow.save_preview.text() and flow.save_preview.text().endswith(".Q4_K_M.gguf")
    flow.save_button.click()
    assert "added 'support-tone'" in flow.save_result.text()
    assert (Path(settings.data_dir) / "models" / "support-tone.Q4_K_M.gguf").is_file()


def test_save_step_has_deploy_fields_and_gpu_default(tmp_path, flow_factory):
    from PySide6 import QtWidgets

    make, _app = flow_factory
    settings = _install(tmp_path)
    settings.finetune.default_gpu_layers = 37
    flow = make(settings)
    flow.open_run(_finished_run(settings))

    assert flow.save_replace.isChecked() is False
    assert flow.save_gpu_layers.value() == 37
    assert isinstance(flow.save_prompt, QtWidgets.QPlainTextEdit)
    flow.save_description.setText("Description")
    flow.save_notes.setPlainText("Notes\nsecond line")
    flow.save_prompt.setPlainText("System\nsecond line")
    flow.save_button.click()

    import yaml

    row = yaml.safe_load(
        (Path(settings.data_dir) / "models.yaml").read_text(encoding="utf-8")
    )["models"][-1]
    assert row["description"] == "Description"
    assert row["notes"] == "Notes\nsecond line"
    assert row["recommended_prompt"] == "System\nsecond line"
    assert row["gpu_layers"] == 37


def test_model_advisor_runs_only_on_click_and_selection_fills_model(
        tmp_path, flow_factory, monkeypatch):
    import finetune_page
    from PySide6 import QtWidgets

    make, _app = flow_factory
    flow = make(_install(tmp_path))
    calls = []
    flow._gc.system_specs = lambda: SimpleNamespace(gpu_vram_total_mb=16384)
    monkeypatch.setattr(
        finetune_page.advisor,
        "suggest",
        lambda vram: (
            calls.append(vram)
            or ([{"id": "org/Model-7B", "params_b": 7.0, "est_gb": 7.4}], "estimated")
        ),
    )
    flow._run = lambda fn, done: done(fn(), None)
    monkeypatch.setattr(
        QtWidgets.QMessageBox,
        "question",
        lambda *_args, **_kwargs: QtWidgets.QMessageBox.StandardButton.Yes,
    )

    assert calls == [] and flow.suggestion_list.count() == 0
    flow.suggest_models.click()
    assert calls == [16.0]
    assert flow.suggestion_list.count() == 1
    flow.suggestion_list.item(0).setSelected(True)
    flow._on_suggestion_picked(flow.suggestion_list.item(0))
    assert flow.custom_model.text() == "org/Model-7B"


def test_model_advisor_names_huggingface_and_requires_confirmation(
        tmp_path, flow_factory, monkeypatch):
    import finetune_page
    from PySide6 import QtWidgets

    make, _app = flow_factory
    flow = make(_install(tmp_path))
    calls = []
    monkeypatch.setattr(
        finetune_page.advisor,
        "suggest",
        lambda _vram: calls.append(True),
    )
    monkeypatch.setattr(
        QtWidgets.QMessageBox,
        "question",
        lambda *_args, **_kwargs: QtWidgets.QMessageBox.StandardButton.No,
    )

    assert "huggingface.co" in flow.suggest_models.toolTip()
    flow.suggest_models.click()
    assert calls == []


def test_save_step_duplicate_id_message_points_to_replace_without_writing(
        tmp_path, flow_factory):
    make, _app = flow_factory
    settings = _install(tmp_path)
    flow = make(settings)
    flow.open_run(_finished_run(settings))
    flow.save_id.setText("existing")
    before = (Path(settings.data_dir) / "models.yaml").read_bytes()

    flow.save_button.click()

    assert "new id" in flow.save_result.text()
    assert "Replace existing model" in flow.save_result.text()
    assert (Path(settings.data_dir) / "models.yaml").read_bytes() == before
    assert not (Path(settings.data_dir) / "models").exists()


def test_replace_refuses_the_running_model_before_confirmation(tmp_path, flow_factory, monkeypatch):
    import finetune_page

    make, _app = flow_factory
    settings = _install(tmp_path)
    flow = make(settings)
    flow.open_run(_finished_run(settings))
    flow.save_id.setText("existing")
    flow.save_replace.setChecked(True)
    flow._gc.running_model_id = lambda: "existing"
    monkeypatch.setattr(
        finetune_page,
        "_plain_question",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("confirmation must not open for a running model")
        ),
    )

    flow.save_button.click()

    assert "Stop model 'existing'" in flow.save_result.text()
    assert not (Path(settings.data_dir) / "models").exists()


def test_replace_confirmation_cancel_writes_nothing_and_lists_changes(
        tmp_path, flow_factory, monkeypatch):
    from PySide6 import QtWidgets
    import finetune_page

    make, _app = flow_factory
    settings = _install(tmp_path)
    flow = make(settings)
    flow.open_run(_finished_run(settings))
    flow.save_id.setText("existing")
    flow.save_name.setText("Replacement")
    flow.save_replace.setChecked(True)
    flow._gc.running_model_id = lambda: None
    before = (Path(settings.data_dir) / "models.yaml").read_bytes()
    shown = {}

    def decline(_parent, title, text, *_args):
        shown.update(title=title, text=text)
        return QtWidgets.QMessageBox.StandardButton.No

    monkeypatch.setattr(finetune_page, "_plain_question", decline)
    flow.save_button.click()

    assert shown["title"] == "Replace existing model?"
    assert "'existing'" in shown["text"]
    assert "Old path:" in shown["text"] and "New path:" in shown["text"]
    assert "name:" in shown["text"] and "location:" in shown["text"]
    assert (Path(settings.data_dir) / "models.yaml").read_bytes() == before
    assert not (Path(settings.data_dir) / "models").exists()


def test_replace_confirmation_accepts_and_keeps_the_old_file(
        tmp_path, flow_factory, monkeypatch):
    import yaml
    from PySide6 import QtWidgets
    import finetune_page

    make, _app = flow_factory
    settings = _install(tmp_path)
    old = tmp_path / "old.gguf"
    old.write_bytes(b"old")
    registry = Path(settings.data_dir) / "models.yaml"
    registry.write_text(
        registry.read_text(encoding="utf-8").replace(
            "/locitize-test/models/existing.gguf", str(old).replace("\\", "/")
        ),
        encoding="utf-8",
    )
    flow = make(settings)
    flow.open_run(_finished_run(settings))
    flow.save_id.setText("existing")
    flow.save_name.setText("Replacement")
    flow.save_replace.setChecked(True)
    flow._gc.running_model_id = lambda: None
    monkeypatch.setattr(
        finetune_page,
        "_plain_question",
        lambda *_args, **_kwargs: QtWidgets.QMessageBox.StandardButton.Yes,
    )

    flow.save_button.click()

    assert "replaced 'Replacement'" in flow.save_result.text()
    assert old.read_bytes() == b"old"
    row = yaml.safe_load(registry.read_text(encoding="utf-8"))["models"][0]
    assert row["id"] == "existing"
    assert row["location"] != str(old) and Path(row["location"]).is_file()


def test_desktop_fine_tune_page_hosts_the_flow_and_no_browser_studio(flow_factory):
    pytest.importorskip("PySide6")
    import desktop
    from tests.test_finetune_ui import _ViewController

    window = desktop.MainWindow(_ViewController(), health="OK")
    assert window._ft_flow.step == 1
    assert not hasattr(window, "_ft_start_btn")
    assert window._ft_save_btn.isEnabled() is False
