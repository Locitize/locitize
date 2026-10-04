"""Tests for the first-run setup planner and executor helpers (M15).

The planner is pure, so it is tested exhaustively and without mocks. The executor
is tested only where it is pure or where a fake can stand in for the machine: no
test here creates a venv, runs pip, opens a socket, or calls winget.
"""

from __future__ import annotations

import zipfile

import pytest

import setup_env
import setup_plan


# --------------------------------------------------------------------------- #
# Feature closure
# --------------------------------------------------------------------------- #


def test_core_is_always_selected_even_when_absent():
    assert "core" in setup_plan.expand_selection(())
    assert "core" in setup_plan.expand_selection(("vision",))


def test_second_eye_pulls_in_vision_and_voice_out():
    expanded = setup_plan.expand_selection(("second_eye",))
    assert "vision" in expanded
    assert "voice_out" in expanded


def test_expansion_returns_catalogue_order_not_input_order():
    expanded = setup_plan.expand_selection(("secure_proxy", "voice_in"))
    order = [f.key for f in setup_plan.FEATURES]
    assert list(expanded) == [k for k in order if k in expanded]


def test_unknown_feature_keys_are_dropped_not_raised():
    # A stale saved selection must never stop the wizard from drawing.
    assert setup_plan.expand_selection(("no_such_feature",)) == ("core",)


def test_expansion_is_idempotent():
    once = setup_plan.expand_selection(("second_eye",))
    assert setup_plan.expand_selection(once) == once


# --------------------------------------------------------------------------- #
# Requirement resolution
# --------------------------------------------------------------------------- #


def test_shared_requirement_is_deduplicated_across_features():
    pairs = setup_plan.requirements_for(("vision", "second_eye"))
    keys = [req.key for req, _ in pairs]
    assert len(keys) == len(set(keys))


def test_wanted_by_records_every_requesting_feature():
    pairs = dict(
        (req.key, wanted) for req, wanted in setup_plan.requirements_for(("second_eye",))
    )
    # voice_out is implied by second_eye, so kokoro is wanted by voice_out.
    assert "voice_out" in pairs["kokoro_pip"]


def test_steps_are_ordered_venv_before_pip():
    plan = setup_plan.plan_for(("core",))
    keys = [s.key for s in plan.steps]
    assert keys.index("platform_venv") < keys.index("base_pip")
    assert keys.index("platform_venv") < keys.index("gui_pip")


def test_every_requirement_named_by_a_feature_exists():
    for feature in setup_plan.FEATURES:
        for key in feature.requires:
            assert key in setup_plan.REQUIREMENTS, f"{feature.key} -> {key}"


def test_every_implied_feature_exists():
    for feature in setup_plan.FEATURES:
        for key in feature.implies:
            assert key in setup_plan.FEATURES_BY_KEY


# --------------------------------------------------------------------------- #
# Plan and sizing
# --------------------------------------------------------------------------- #


def test_satisfied_step_costs_nothing():
    plan = setup_plan.plan_for(("core",), {"llama_cpp": True})
    step = next(s for s in plan.steps if s.key == "llama_cpp")
    assert step.satisfied
    assert step.size_mb == 0


def test_total_excludes_already_present_items():
    bare = setup_plan.plan_for(("core",))
    partial = setup_plan.plan_for(("core",), {"llama_cpp": True})
    llama_size = setup_plan.REQUIREMENTS["llama_cpp"].size_mb
    assert bare.total_mb - partial.total_mb == llama_size


def test_missing_state_key_is_treated_as_not_satisfied():
    # Pessimistic by design: a wrong True would claim a broken install is done.
    plan = setup_plan.plan_for(("core",), {})
    assert all(not s.satisfied for s in plan.steps)


def test_nothing_to_do_when_every_requirement_is_present():
    plan = setup_plan.plan_for(("core",))
    everything = {s.key: True for s in plan.steps}
    assert setup_plan.plan_for(("core",), everything).nothing_to_do


def test_summarize_states_the_real_total():
    plan = setup_plan.plan_for(("core",), {})
    assert setup_plan.format_size(plan.total_mb) in setup_plan.summarize(plan)


def test_summarize_when_nothing_to_do_makes_no_download_claim():
    plan = setup_plan.plan_for(("core",), {s.key: True for s in setup_plan.plan_for(("core",)).steps})
    assert "already installed" in setup_plan.summarize(plan)


@pytest.mark.parametrize(
    "mb,expected",
    [(0, "nothing to download"), (15, "15 MB"), (1024, "1.0 GB"), (2600, "2.5 GB")],
)
def test_format_size(mb, expected):
    assert setup_plan.format_size(mb) == expected


# --------------------------------------------------------------------------- #
# Asset scoring (pure)
# --------------------------------------------------------------------------- #


def test_cuda_build_wins_on_an_nvidia_machine():
    names = ["llama-b1-bin-win-cpu-x64.zip", "llama-b1-bin-win-cuda-12.4-x64.zip"]
    assert "cuda" in setup_env.choose_asset(names, want_cuda=True)


def test_cuda_build_is_never_chosen_without_a_gpu():
    names = ["llama-b1-bin-win-cuda-12.4-x64.zip"]
    assert setup_env.choose_asset(names, want_cuda=False) == ""


def test_cpu_build_is_an_acceptable_fallback_on_a_gpu_machine():
    names = ["llama-b1-bin-win-cpu-x64.zip"]
    assert setup_env.choose_asset(names, want_cuda=True) == names[0]


def test_foreign_vendor_and_arch_builds_are_rejected():
    for name in (
        "llama-b1-bin-win-hip-x64.zip",
        "llama-b1-bin-win-vulkan-x64.zip",
        "llama-b1-bin-win-sycl-x64.zip",
        "llama-b1-bin-win-cpu-arm64.zip",
        "llama-b1-bin-ubuntu-x64.zip",
        "llama-b1-bin-win-cpu-x64.tar.gz",
    ):
        assert setup_env.score_asset(name, want_cuda=True) == 0, name


def test_choose_asset_returns_empty_when_nothing_fits():
    assert setup_env.choose_asset(["llama-b1-bin-ubuntu-x64.zip"], want_cuda=True) == ""


def test_avx2_preferred_among_cpu_builds():
    names = ["llama-b1-bin-win-cpu-x64.zip", "llama-b1-bin-win-cpu-avx2-x64.zip"]
    assert "avx2" in setup_env.choose_asset(names, want_cuda=False)


# --------------------------------------------------------------------------- #
# Release asset parsing
# --------------------------------------------------------------------------- #


def test_release_asset_extracts_sha_from_publisher_digest():
    asset = setup_env.ReleaseAsset("a.zip", "u", 1, "sha256:" + "ab" * 32)
    assert asset.sha256 == "ab" * 32


def test_release_asset_without_digest_reports_no_sha():
    assert setup_env.ReleaseAsset("a.zip", "u").sha256 == ""


def test_release_asset_ignores_a_non_sha256_digest():
    assert setup_env.ReleaseAsset("a.zip", "u", 1, "md5:abc").sha256 == ""


# --------------------------------------------------------------------------- #
# Archive safety
# --------------------------------------------------------------------------- #


def test_unpack_refuses_a_member_that_escapes_the_destination(tmp_path):
    archive = tmp_path / "evil.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("../escaped.txt", "no")
    result = setup_env.unpack_zip(archive, tmp_path / "out")
    assert not result.ok
    assert "escapes destination" in result.message
    assert not (tmp_path / "escaped.txt").exists()


def test_unpack_extracts_a_normal_archive(tmp_path):
    archive = tmp_path / "ok.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("bin/llama-server.exe", "stub")
    result = setup_env.unpack_zip(archive, tmp_path / "out")
    assert result.ok
    assert (tmp_path / "out" / "bin" / "llama-server.exe").is_file()


def test_unpack_reports_a_corrupt_archive_without_raising(tmp_path):
    bad = tmp_path / "bad.zip"
    bad.write_bytes(b"not a zip")
    result = setup_env.unpack_zip(bad, tmp_path / "out")
    assert not result.ok


# --------------------------------------------------------------------------- #
# Executor guards
# --------------------------------------------------------------------------- #


def test_create_venv_is_idempotent_on_an_existing_venv(tmp_path, monkeypatch):
    venv = tmp_path / ".venv"
    exe = setup_env.venv_python_path(tmp_path)
    exe.parent.mkdir(parents=True)
    exe.write_text("stub")
    result = setup_env.create_venv(venv)
    assert result.ok and result.skipped


def test_pip_install_refuses_a_missing_interpreter(tmp_path):
    result = setup_env.pip_install(tmp_path / "nope.exe", ("PyYAML",))
    assert not result.ok
    assert "not found" in result.message


def test_write_settings_paths_is_a_noop_with_no_updates(tmp_path):
    result = setup_env.write_settings_paths(tmp_path / "python.exe", tmp_path, {})
    assert result.ok and result.skipped


def test_write_settings_paths_refuses_when_settings_missing(tmp_path):
    exe = tmp_path / "python.exe"
    exe.write_text("stub")
    result = setup_env.write_settings_paths(exe, tmp_path, {"llama_cpp": "x"})
    assert not result.ok
    assert "no settings.yaml" in result.message


def test_modules_present_is_false_for_a_missing_interpreter(tmp_path):
    assert not setup_env.modules_present(tmp_path / "nope.exe", ("sys",))


def test_modules_present_is_true_for_an_empty_module_list(tmp_path):
    assert setup_env.modules_present(tmp_path / "nope.exe", ())


def test_pip_sets_cover_every_pip_requirement():
    for key, req in setup_plan.REQUIREMENTS.items():
        if req.kind == setup_plan.KIND_PIP:
            assert key in setup_env.PIP_SETS, key


def test_kokoro_pip_set_keeps_the_cpu_index_and_the_exact_pin():
    packages = setup_env.PIP_SETS["kokoro_pip"]
    assert "--extra-index-url" in packages
    assert any(p.startswith("torch==") and p.endswith("+cpu") for p in packages)


def test_kokoro_cuda_set_pins_the_cu128_index_first_and_carries_kokoro():
    """The GPU set REPLACES the CPU torch, so cu128 must be the primary index
    (an extra index would let the resolver keep the +cpu wheel) and the set
    must carry kokoro itself: the wizard skips the CPU step when this one is
    coming, so nothing else installs the package."""
    packages = setup_env.PIP_SETS["kokoro_pip_cuda"]
    assert packages[:2] == ("--index-url", "https://download.pytorch.org/whl/cu128")
    assert "https://pypi.org/simple" in packages
    assert any(p.startswith("torch==") and p.endswith("+cu128") for p in packages)
    assert any(p.startswith("kokoro") for p in packages)


def test_gpu_voice_is_opt_in_and_implies_the_voice_itself():
    feature = next(f for f in setup_plan.FEATURES if f.key == "voice_out_gpu")
    assert not feature.default_on
    assert "voice_out" in feature.implies
    expanded = setup_plan.expand_selection(("voice_out_gpu",))
    assert "voice_out" in expanded
    keys = [req.key for req, _ in setup_plan.requirements_for(expanded)]
    # The CPU step is planned before the GPU one: the chain that runs them is
    # ordered by this list, and the GPU step overwrites what the CPU step did.
    assert keys.index("kokoro_pip") < keys.index("kokoro_pip_cuda")
    assert keys.index("kokoro_weights") > keys.index("kokoro_pip_cuda")


def test_torch_sees_cuda_is_false_for_a_missing_interpreter(tmp_path):
    assert not setup_env.torch_sees_cuda(tmp_path / "nope.exe")


def test_release_hosts_do_not_leak_into_the_model_allowlist():
    import modelhub

    for host in setup_env.RELEASE_HOSTS:
        assert host not in modelhub.DEFAULT_ALLOWED_HOSTS


# --------------------------------------------------------------------------- #
# M15.11: local model discovery (no shipped model names)
# --------------------------------------------------------------------------- #


def test_scan_finds_ggufs_and_skips_projectors_and_small_files(tmp_path):
    big = b"G" * (setup_env.DISCOVERY_MIN_BYTES + 1)
    (tmp_path / "chat-model.Q4_K_M.gguf").write_bytes(big)
    (tmp_path / "mmproj-F16.gguf").write_bytes(big)          # projector: skip
    (tmp_path / "tokenizer-fixture.gguf").write_bytes(b"x")  # tiny: skip
    found = setup_env.find_local_models([tmp_path])
    assert [f["name"] for f in found] == ["chat-model.Q4_K_M.gguf"]


def test_scan_dedupes_hardlinked_copies(tmp_path):
    import os as _os

    big = b"G" * (setup_env.DISCOVERY_MIN_BYTES + 1)
    original = tmp_path / "a" / "model.gguf"
    original.parent.mkdir()
    original.write_bytes(big)
    linked = tmp_path / "b" / "model.gguf"
    linked.parent.mkdir()
    _os.link(original, linked)
    found = setup_env.find_local_models([tmp_path])
    assert len(found) == 1  # one model, however many names it has


def test_scan_keeps_same_name_different_size_as_distinct(tmp_path):
    # Two revisions of "the same" model differ by bytes; both are real.
    (tmp_path / "a").mkdir(); (tmp_path / "b").mkdir()
    (tmp_path / "a" / "m.gguf").write_bytes(b"G" * (setup_env.DISCOVERY_MIN_BYTES + 1))
    (tmp_path / "b" / "m.gguf").write_bytes(b"G" * (setup_env.DISCOVERY_MIN_BYTES + 2))
    assert len(setup_env.find_local_models([tmp_path])) == 2


def test_place_hardlinks_on_same_volume_without_copying(tmp_path):
    import os as _os

    src = tmp_path / "store" / "model.gguf"
    src.parent.mkdir()
    src.write_bytes(b"G" * 1024)
    models = tmp_path / "models"
    placed = setup_env.place_into_models_dir(src, models)
    assert placed == str(models / "model.gguf")
    assert _os.stat(placed).st_ino == src.stat().st_ino  # same file, no copy


def test_place_never_clobbers_a_different_file_with_the_same_name(tmp_path):
    src = tmp_path / "store" / "model.gguf"
    src.parent.mkdir()
    src.write_bytes(b"G" * 1024)
    models = tmp_path / "models"
    models.mkdir()
    (models / "model.gguf").write_bytes(b"different")
    placed = setup_env.place_into_models_dir(src, models)
    assert placed == str(src)  # falls back to the original, clobbers nothing
    assert (models / "model.gguf").read_bytes() == b"different"


# --------------------------------------------------------------------------- #
# M17.4: update check version parsing (pure)
# --------------------------------------------------------------------------- #


def test_installed_build_parses_new_format(tmp_path, monkeypatch):
    import setup_env
    exe = tmp_path / "llama-server.exe"
    exe.write_text("stub")
    monkeypatch.setattr(setup_env, "_run",
                        lambda *a, **k: (0, "version: 0.3.0-dev (build 10684, commit abc)"))
    assert setup_env.installed_llama_build(exe) == 10684


def test_installed_build_parses_old_format(tmp_path, monkeypatch):
    import setup_env
    exe = tmp_path / "llama-server.exe"
    exe.write_text("stub")
    monkeypatch.setattr(setup_env, "_run",
                        lambda *a, **k: (0, "version: 10037 (56d6e9dde)"))
    assert setup_env.installed_llama_build(exe) == 10037


def test_installed_build_missing_binary_is_none(tmp_path):
    import setup_env
    assert setup_env.installed_llama_build(tmp_path / "nope.exe") is None


def test_check_update_reports_available(tmp_path, monkeypatch):
    import setup_env
    exe = tmp_path / "llama-server.exe"
    exe.write_text("stub")
    monkeypatch.setattr(setup_env, "installed_llama_build", lambda p: 10000)
    monkeypatch.setattr(setup_env, "latest_llama_build", lambda cuda: (10687, ""))
    r = setup_env.check_llama_update(exe, want_cuda=True)
    assert r["ok"] and r["update_available"] and "10687" in r["detail"]
