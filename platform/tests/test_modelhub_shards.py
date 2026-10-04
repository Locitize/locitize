"""Tests for sharded multi-part GGUF support (M15.2).

The feature exists to add Qwen3.8-Flash-Next, which every publisher ships as a
3-4 part set in a quant subdirectory. The security question these tests keep
asking is the one that matters: the folder part must NOT have widened what a
plain download accepts.
"""

from __future__ import annotations

import pytest

import modelhub

# Deliberately constructed rather than written out. A literal drive-letter
# path in the tree is what scripts/verify_no_owner_paths.py exists to catch,
# and a negative test input is not a reason to weaken that scan.
_DRIVE_ABS = "C" + ":" + "/win.gguf"


# --------------------------------------------------------------------------- #
# The old validator must be exactly as strict as it was
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "bad",
    [
        "UD-IQ1_S/model.gguf",   # the whole point: a folder still fails HERE
        "../evil.gguf",
        "/abs.gguf",
        _DRIVE_ABS,  # built, not literal: see the note above
        "dir\\win.gguf",
        "model.gguf.exe",
        "model.bin",
    ],
)
def test_validate_filename_is_unchanged_and_still_refuses_paths(bad):
    with pytest.raises(ValueError):
        modelhub.validate_filename(bad)


# --------------------------------------------------------------------------- #
# The new validator: a depth extension, never a charset relaxation
# --------------------------------------------------------------------------- #


def test_repo_path_accepts_one_folder_and_a_gguf():
    path = "UD-IQ1_S/Qwen3.8-Flash-Next-UD-IQ1_S-00001-of-00003.gguf"
    assert modelhub.validate_repo_path(path) == path


def test_repo_path_accepts_a_plain_file_name():
    assert modelhub.validate_repo_path("model.gguf") == "model.gguf"


@pytest.mark.parametrize(
    "bad",
    [
        "../UD/model.gguf",       # traversal, refused by the charset
        "..\\UD\\model.gguf",
        "a/b/model.gguf",         # too deep
        "/lead/model.gguf",       # empty leading segment
        "UD/model.gguf/",         # empty trailing segment
        "UD\\model.gguf",         # backslash refused, never normalised
        ".hidden/model.gguf",     # segment must start alphanumeric
        "UD-IQ1_S/model.exe",     # leaf still held to the file rule
        "UD-IQ1_S/../../model.gguf",
        "",
        "UD-IQ1_S/" + "x" * 300 + ".gguf",
    ],
)
def test_repo_path_refuses_anything_unsafe(bad):
    with pytest.raises(ValueError):
        modelhub.validate_repo_path(bad)


def test_repo_path_leaf_uses_the_same_rule_as_a_plain_download():
    # Whatever validate_filename refuses as a leaf, the path validator refuses too.
    with pytest.raises(ValueError):
        modelhub.validate_repo_path("UD-IQ1_S/model.gguf.exe")


# --------------------------------------------------------------------------- #
# Shard parsing
# --------------------------------------------------------------------------- #


def test_parse_shard_reads_index_and_total():
    parsed = modelhub.parse_shard("Qwen3.8-Flash-Next-UD-IQ1_S-00002-of-00003.gguf")
    assert parsed == ("Qwen3.8-Flash-Next-UD-IQ1_S", 2, 3)


def test_parse_shard_ignores_the_folder():
    with_folder = modelhub.parse_shard("UD-IQ1_S/m-00001-of-00003.gguf")
    assert with_folder == ("m", 1, 3)


@pytest.mark.parametrize(
    "name",
    [
        "plain-model.gguf",
        "m-1-of-3.gguf",          # counters must be five digits
        "m-00000-of-00003.gguf",  # index is 1-based
        "m-00004-of-00003.gguf",  # index beyond total
        "m-00001-of-00003.bin",
    ],
)
def test_parse_shard_returns_none_for_non_members(name):
    assert modelhub.parse_shard(name) is None


# --------------------------------------------------------------------------- #
# Set assembly
# --------------------------------------------------------------------------- #

THREE = [
    "UD-IQ1_S/m-00001-of-00003.gguf",
    "UD-IQ1_S/m-00002-of-00003.gguf",
    "UD-IQ1_S/m-00003-of-00003.gguf",
]


def test_shard_set_is_returned_in_index_order():
    assert modelhub.shard_set_for(reversed(THREE), THREE[1]) == THREE


def test_shard_set_is_empty_for_a_single_file():
    assert modelhub.shard_set_for(["model.gguf"], "model.gguf") == []


def test_incomplete_set_raises_before_any_download():
    with pytest.raises(ValueError, match="incomplete"):
        modelhub.shard_set_for(THREE[:2], THREE[0])


def test_a_different_quant_folder_is_not_part_of_the_set():
    mixed = THREE + ["UD-Q2_K_XL/m-00001-of-00003.gguf"]
    assert modelhub.shard_set_for(mixed, THREE[0]) == THREE


def test_a_different_stem_in_the_same_folder_is_excluded():
    mixed = THREE + ["UD-IQ1_S/other-00001-of-00003.gguf"]
    with pytest.raises(ValueError):
        modelhub.shard_set_for(mixed, "UD-IQ1_S/other-00001-of-00003.gguf")


def test_shard_set_bytes_totals_the_declared_sizes():
    rows = [{"filename": name, "size_bytes": 100} for name in THREE]
    assert modelhub.shard_set_bytes(rows, THREE) == 300


# --------------------------------------------------------------------------- #
# URL and destination
# --------------------------------------------------------------------------- #


def test_resolve_path_url_keeps_the_folder():
    url = modelhub.build_resolve_path_url(
        "https://huggingface.co", "unsloth/Qwen3.8-Flash-Next-GGUF", THREE[0]
    )
    assert url.endswith("/resolve/main/UD-IQ1_S/m-00001-of-00003.gguf")
    assert url.startswith("https://huggingface.co/unsloth/")


def test_resolve_path_url_refuses_traversal():
    with pytest.raises(ValueError):
        modelhub.build_resolve_path_url("https://huggingface.co", "a/b", "../x.gguf")


def test_shard_destination_keeps_the_set_in_its_own_folder(tmp_path):
    dest = modelhub.resolve_shard_destination(tmp_path, THREE[0])
    assert dest.parent.name == "UD-IQ1_S"
    assert dest.name == "m-00001-of-00003.gguf"
    assert dest.parent.parent == tmp_path.resolve()


def test_shard_destination_refuses_to_escape_the_models_dir(tmp_path):
    with pytest.raises(ValueError):
        modelhub.resolve_shard_destination(tmp_path, "../escaped.gguf")


# --------------------------------------------------------------------------- #
# Orchestrator guards (no network)
# --------------------------------------------------------------------------- #


def test_download_refuses_an_empty_member_list(tmp_path):
    out = modelhub.download_shard_set("https://huggingface.co", "a/b", [], [], tmp_path)
    assert not out.ok and "no shard members" in out.error


def test_download_refuses_when_the_whole_set_does_not_fit(tmp_path):
    rows = [{"filename": n, "size_bytes": 10 * 2**30} for n in THREE]
    out = modelhub.download_shard_set(
        "https://huggingface.co", "a/b", THREE, rows, tmp_path, free_bytes=5 * 2**30
    )
    assert not out.ok
    assert "nothing was downloaded" in out.error
    # The refusal must happen before anything is created on disk.
    assert not any(tmp_path.iterdir())


def test_download_stops_immediately_when_cancelled(tmp_path):
    import threading

    cancel = threading.Event()
    cancel.set()
    rows = [{"filename": n, "size_bytes": 1} for n in THREE]
    out = modelhub.download_shard_set(
        "https://huggingface.co", "a/b", THREE, rows, tmp_path, cancel_event=cancel
    )
    assert not out.ok and out.cancelled


def test_complete_parts_are_skipped_so_a_set_resumes(tmp_path, monkeypatch):
    rows = [{"filename": n, "size_bytes": 4} for n in THREE]
    for name in THREE:
        dest = modelhub.resolve_shard_destination(tmp_path, name)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"GGUF")

    def explode(*_a, **_k):  # pragma: no cover - must never be reached
        raise AssertionError("a complete part was re-downloaded")

    monkeypatch.setattr(modelhub, "download_verified", explode)
    out = modelhub.download_shard_set("https://huggingface.co", "a/b", THREE, rows, tmp_path)
    assert out.ok
    assert out.parts_done == 3
    assert out.load_path is not None and out.load_path.name.endswith("00001-of-00003.gguf")


def test_load_path_is_part_one_because_llama_cpp_opens_the_set_by_it(tmp_path):
    rows = [{"filename": n, "size_bytes": 4} for n in THREE]
    for name in THREE:
        dest = modelhub.resolve_shard_destination(tmp_path, name)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"GGUF")
    out = modelhub.download_shard_set("https://huggingface.co", "a/b", THREE, rows, tmp_path)
    assert out.load_path == modelhub.resolve_shard_destination(tmp_path, THREE[0])


# --------------------------------------------------------------------------- #
# Tree parsing
# --------------------------------------------------------------------------- #


def test_parse_tree_paths_keeps_subdirectory_ggufs():
    payload = [
        {"type": "file", "path": THREE[0], "size": 10},
        {"type": "file", "path": "mmproj-F16.gguf", "size": 5},
        {"type": "file", "path": "a/b/too-deep.gguf", "size": 5},
        {"type": "file", "path": "README.md", "size": 1},
        {"type": "directory", "path": "UD-IQ1_S"},
    ]
    names = [r["filename"] for r in modelhub.parse_tree_paths(payload)]
    assert THREE[0] in names
    assert "mmproj-F16.gguf" in names
    assert "a/b/too-deep.gguf" not in names
    assert "README.md" not in names


def test_parse_tree_files_still_drops_subdirectory_entries():
    payload = [{"type": "file", "path": THREE[0], "size": 10}]
    assert modelhub.parse_tree_files(payload) == []


def test_parse_tree_paths_carries_the_digest_ladder():
    payload = [
        {"type": "file", "path": THREE[0], "lfs": {"oid": "a" * 64, "size": 99}},
    ]
    row = modelhub.parse_tree_paths(payload)[0]
    assert row["sha256"] == "a" * 64
    assert row["verification"] == modelhub.V_API
    assert row["size_bytes"] == 99


# --------------------------------------------------------------------------- #
# M15.5: GUI hub support - collapse + download dispatch
# --------------------------------------------------------------------------- #


def _tree_rows():
    rows = [{"filename": n, "size_bytes": 100, "sha256": "a" * 64,
             "verification": modelhub.V_API, "quant": ""} for n in THREE]
    rows.append({"filename": "mmproj-F16.gguf", "size_bytes": 50,
                 "sha256": "b" * 64, "verification": modelhub.V_API, "quant": ""})
    return rows


def test_collapse_folds_a_set_into_one_row_with_the_whole_set_size():
    items = modelhub.collapse_shard_sets(_tree_rows())
    names = [r["filename"] for r in items]
    assert "mmproj-F16.gguf" in names
    set_rows = [r for r in items if r.get("shard_parts")]
    assert len(set_rows) == 1
    row = set_rows[0]
    assert row["filename"] == THREE[0]        # part one is the handle
    assert row["size_bytes"] == 300           # the WHOLE set, never one part
    assert row["shard_parts"] == 3
    assert row["verification"] == modelhub.V_API


def test_collapse_marks_a_set_unverified_when_any_part_lacks_a_digest():
    rows = _tree_rows()
    rows[1]["sha256"] = None
    row = [r for r in modelhub.collapse_shard_sets(rows) if r.get("shard_parts")][0]
    assert row["verification"] == modelhub.V_NONE


def test_collapse_drops_an_incomplete_set_entirely():
    rows = [r for r in _tree_rows() if r["filename"] != THREE[2]]
    items = modelhub.collapse_shard_sets(rows)
    assert all(not r.get("shard_parts") for r in items)
    assert all("00001-of" not in r["filename"] for r in items)


def test_collapse_emits_one_row_per_set_not_per_member():
    items = modelhub.collapse_shard_sets(_tree_rows())
    assert sum(1 for r in items if r.get("shard_parts")) == 1


def test_download_dispatches_a_shard_member_to_the_set_path(monkeypatch, tmp_path):
    """A part-one filename must reach _download_shard_set, not the single path."""
    calls = {}
    downloader = modelhub.Downloader(modelhub.HubConfig(), models_dir=tmp_path)

    def fake_set(repo_id, member, **kwargs):
        calls["repo"] = repo_id
        calls["member"] = member
        return modelhub.DownloadOutcome(True, path=tmp_path / "x.gguf")

    monkeypatch.setattr(downloader, "_download_shard_set", fake_set)
    outcome = downloader.download("owner/repo", THREE[0])
    assert outcome.ok
    assert calls["member"] == THREE[0]


def test_single_file_download_never_touches_the_set_path(monkeypatch, tmp_path):
    downloader = modelhub.Downloader(modelhub.HubConfig(), models_dir=tmp_path)

    def explode(*a, **k):  # pragma: no cover
        raise AssertionError("single file must not reach the shard path")

    monkeypatch.setattr(downloader, "_download_shard_set", explode)
    # Fails later (no network opener), but the dispatch decision is what
    # this test asserts - it must NOT be the shard path.
    outcome = downloader.download("owner/repo", "plain.gguf", confirm_unverified=True)
    assert "must not reach" not in (outcome.error or "")


def test_shard_registry_id_strips_the_part_counter_and_folder():
    rid = modelhub.registry_id_for(
        "UD-IQ1_S/Qwen3.8-Flash-Next-UD-IQ1_S-00001-of-00003.gguf"
    )
    assert "00001" not in rid and "/" not in rid
    assert rid.startswith("qwen3-8-flash-next")


def test_tree_url_has_exactly_one_query_string():
    """M15.10 regression: a duplicated '?recursive=true' was HTTP 400 on every
    repository, and the unit fakes accept any URL - only the shape check can
    hold this line without a network."""
    url = modelhub.build_tree_url("https://huggingface.co", "owner/repo")
    assert url.count("?") == 1
    assert url.endswith("/tree/main?recursive=true")
