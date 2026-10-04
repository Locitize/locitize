"""PROTECTED checksummed model-weight downloader - OWNER RUN ONLY (thin CLI).

WHAT THIS IS NOW
----------------
As of M14.14.7 the verified-download machinery lives in `modelhub.py` and is
shared with the Models page's Get models section. This script is the thin command
line over it, and it stays for one specific reason: it is the ONLY place in the
whole product where a raw download URL is accepted. The GUI never accepts a typed
URL - it builds every URL from a validated repository id and file name - so this
script is the deliberate manual escape hatch for a file that is not on
HuggingFace.

The import direction is one way and load-bearing: `scripts/fetch_model.py`
imports `modelhub`, never the reverse. No GUI code path can reach this file's
argument handling, so running it stays a conscious act by the owner.

It is a PROTECTED action: network egress plus a multi-gigabyte disk write.

SAFETY CONTRACT (unchanged, now enforced inside modelhub.download_verified)
--------------------------------------------------------------------------
- https only, and the host must be on the BUILT-IN default allowlist
  (modelhub.DEFAULT_ALLOWED_HOSTS) - re-checked on every redirect hop. Note this
  script does not read the user's settings.yaml models_hub block, so a narrowed
  allowlist there does not narrow this CLI; it is an owner-run tool by design.
- Stream to a `.partial` file, hashing while writing.
- Rename into place ONLY when the computed sha256 matches the expected one.
- On mismatch: delete the partial, print both digests, exit non-zero. A wrong or
  corrupted download is never left where the registry could load it.
- Refuse to overwrite an existing destination. No silent clobber.
- The finished file must begin with the ASCII magic `GGUF`, which is what catches
  an HTML error or sign-in page saved under a .gguf name.

This script does NOT hardcode a URL or a hash. The exact file and its checksum
must be confirmed against the live listing at download time, because a guessed
hash is worse than no hash at all.

USAGE (owner, from Codebase/platform)
-------------------------------------
Manual form - you supply the URL and the digest yourself:

  python scripts/fetch_model.py \
      --url    <confirmed direct https download URL> \
      --sha256 <confirmed expected sha256 of that file> \
      --dest   <your models directory>/<model>.gguf

Convenience form - LOCITIZE resolves the URL and the checksum from HuggingFace:

  python scripts/fetch_model.py \
      --repo unsloth/Qwen3-4B-Instruct-2507-GGUF \
      --file Qwen3-4B-Instruct-2507-Q4_K_M.gguf \
      --dest <your models directory>/Qwen3-4B-Instruct-2507-Q4_K_M.gguf

If HuggingFace publishes no checksum for the file, the convenience form REFUSES
unless you also pass --i-accept-no-checksum, which records in your own command
line that you were told LOCITIZE cannot verify the file.

Afterwards, add the saved path as a `location:` in models.yaml (the Models page's
Register button does this for you) and restart LOCITIZE.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# The platform directory is this script's parent's parent; add it so `modelhub`
# imports the same way it does for every other entry point in the tree.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import modelhub  # noqa: E402 - deliberate: the sys.path line above must run first


def _print_progress(progress: modelhub.Progress) -> None:
    """One line per progress sample, throttled to twice a second by the core."""
    total = f" of {progress.bytes_total / 1e9:.2f} GB" if progress.bytes_total else ""
    print(
        f"  {progress.phase}: {progress.bytes_done / 1e9:.2f} GB{total} "
        f"({progress.rate_bps / 1e6:.1f} MB/s)"
    )


def _report(outcome: modelhub.DownloadOutcome) -> int:
    """Turn an outcome into owner-readable output and a shell exit code."""
    if outcome.ok:
        rung = modelhub.VERIFICATION_LABELS.get(outcome.verification, "")
        print(f"sha256 OK ({outcome.sha256}); saved to {outcome.path}")
        print(f"  verification: {outcome.verification} - {rung}")
        print(
            "Next: add this path as a model 'location:' in models.yaml "
            "(or use the Models page's Register button) and restart LOCITIZE."
        )
        return 0
    print(f"FAIL: {outcome.error}")
    return 1


def main(argv: list[str] | None = None) -> int:
    """Parse arguments and run exactly one of the two supported forms."""
    parser = argparse.ArgumentParser(
        prog="fetch_model",
        description=(
            "PROTECTED owner-run checksummed model downloader. The only raw-URL "
            "download entry point in LOCITIZE."
        ),
    )
    parser.add_argument("--url", help="confirmed direct https download URL")
    parser.add_argument("--sha256", help="confirmed expected sha256 of the file")
    parser.add_argument(
        "--dest",
        required=True,
        help=(
            "destination gguf path on disk. With --repo/--file only the FOLDER "
            "part is used: the file always keeps its published name, so the "
            "digest and the file name can never disagree"
        ),
    )
    parser.add_argument("--repo", help="HuggingFace repository id, owner/name")
    parser.add_argument("--file", dest="filename", help="file name inside --repo")
    parser.add_argument(
        "--i-accept-no-checksum",
        action="store_true",
        help=(
            "proceed when HuggingFace publishes no checksum; the file is saved "
            "and its computed digest recorded, but LOCITIZE cannot verify it"
        ),
    )
    parser.add_argument(
        "--shard-set",
        dest="shard_set",
        action="store_true",
        help=(
            "treat --file as one member of a multi-part GGUF set (M15.2): resolve "
            "every sibling part from the repository tree, refuse to start unless "
            "the WHOLE set fits on disk, fetch each part through the same verified "
            "path, and print the part-one path to register. Required for large MoE "
            "models such as Qwen3.8-Flash-Next, which no publisher ships as a "
            "single file"
        ),
    )
    args = parser.parse_args(argv)

    dest = Path(args.dest)

    if args.repo and args.filename and args.shard_set:
        return _fetch_shard_set(args, dest)
    if args.repo and args.filename:
        return _fetch_by_repo(args, dest)
    if args.url:
        if not args.sha256:
            print(
                "REFUSING: --url requires --sha256. LOCITIZE will not download a "
                "file from an arbitrary URL without a digest to check it against."
            )
            return 1
        # The manual form's URL is still validated against the same host
        # allowlist: this is an escape hatch for an unusual HOST-approved
        # source, not a way to disable the transport rules.
        # V_OPERATOR, not V_API: the digest came from this command line, so the
        # printed provenance must say "the digest you supplied". Claiming
        # HuggingFace verified it would overstate what LOCITIZE actually knows.
        outcome = modelhub.download_verified(
            args.url,
            dest,
            args.sha256,
            verification=modelhub.V_OPERATOR,
            progress=_print_progress,
        )
        return _report(outcome)
    print("REFUSING: pass either --url + --sha256, or --repo + --file.")
    return 1


def _fetch_by_repo(args: argparse.Namespace, dest: Path) -> int:
    """Resolve the file through the same checksum ladder the GUI uses."""
    downloader = modelhub.Downloader(modelhub.HubConfig())
    listing = downloader.list_files(args.repo)
    if not listing["ok"]:
        print(f"FAIL: {listing['reason']}")
        return 1
    row = next(
        (r for r in listing["items"] if r["filename"] == args.filename), None
    )
    if row is None:
        available = ", ".join(r["filename"] for r in listing["items"][:8]) or "none"
        print(
            f"FAIL: '{args.filename}' is not a downloadable single-file .gguf in "
            f"{args.repo}. Available: {available}"
        )
        return 1
    if row["verification"] == modelhub.V_NONE and not args.i_accept_no_checksum:
        print("REFUSING: " + modelhub.UNVERIFIED_CONSENT_TEXT)
        print("Re-run with --i-accept-no-checksum if you want it anyway.")
        return 1
    outcome = downloader.download(
        args.repo,
        args.filename,
        expected_sha256=row["sha256"],
        verification=row["verification"],
        size_bytes=row["size_bytes"],
        confirm_unverified=bool(args.i_accept_no_checksum),
        models_dir=dest.parent,
        progress=_print_progress,
    )
    return _report(outcome)


def _fetch_shard_set(args: argparse.Namespace, dest: Path) -> int:
    """Fetch a multi-part GGUF set (M15.2).

    Uses the recursive tree because shard sets live in a quant subdirectory, which
    the single-file listing deliberately excludes. Every part still goes through
    download_verified with its own digest, so the safety contract is per-part and
    identical to a single-file fetch; what is added is a whole-set fit check up
    front and a resume that starts at the first missing part.
    """
    import shutil

    downloader = modelhub.Downloader(modelhub.HubConfig())
    # build_tree_url is already recursive (M15.10; a duplicated query string
    # here was HTTP 400 on every repository).
    url = modelhub.build_tree_url(modelhub.DEFAULT_API_BASE, args.repo)
    try:
        payload = downloader._get_json(url)  # noqa: SLF001 - same package, one CLI
    except Exception as exc:  # noqa: BLE001 - any network failure is one message
        print(f"FAIL: could not read the repository tree: {exc}")
        return 1

    rows = modelhub.parse_tree_paths(payload)
    names = [r["filename"] for r in rows]
    try:
        members = modelhub.shard_set_for(names, args.filename)
    except ValueError as exc:
        print(f"FAIL: {exc}")
        return 1
    if not members:
        print(
            f"FAIL: '{args.filename}' is not a shard member. Drop --shard-set for "
            f"a single file, or name the '-00001-of-000NN.gguf' part."
        )
        return 1

    total = modelhub.shard_set_bytes(rows, members)
    unverified = [n for n in members if not next(
        (r["sha256"] for r in rows if r["filename"] == n), None
    )]
    if unverified and not args.i_accept_no_checksum:
        print(f"REFUSING: {len(unverified)} of {len(members)} parts publish no checksum.")
        print("Re-run with --i-accept-no-checksum if you want it anyway.")
        return 1

    free = shutil.disk_usage(dest.parent if dest.parent.exists() else Path.cwd()).free
    print(f"shard set: {len(members)} parts, {total / 2**30:.1f} GB")
    print(f"free space: {free / 2**30:.1f} GB")

    outcome = modelhub.download_shard_set(
        modelhub.DEFAULT_API_BASE,
        args.repo,
        members,
        rows,
        dest.parent,
        confirm_unverified=bool(args.i_accept_no_checksum),
        progress=_print_progress,
        free_bytes=free,
    )
    if not outcome.ok:
        print(f"FAIL: {outcome.error}")
        return 1
    print(f"OK: {outcome.parts_done}/{outcome.parts_total} parts, "
          f"{outcome.bytes_written / 2**30:.1f} GB written")
    print(f"Register this path (llama.cpp finds the siblings): {outcome.load_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
