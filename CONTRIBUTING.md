# Contributing

locitize is small on dependencies and heavy on discipline. The test suite
and its gates are the contract; if they pass, your change fits.

## Setup

```
cd platform
pip install -r requirements.txt
python -m pytest -q          # expect: all green in under a minute
```

## The rules the gates enforce (so you do not fight them)

- **No machine-specific paths in the tree.** `scripts/verify_no_owner_paths.py`
  refuses drive-letter absolute paths, user directories, and machine names -
  in code, tests, and docs alike. Derive paths from the environment or the
  checkout.
- **The install tree is never written at runtime.** Logs, downloads, user
  config all go to the data root. The write-fence tests check this.
- **Honest degradation, never a traceback.** A missing binary is a health
  finding with a remedy. A failed optional step narrows a feature.
- **Measured, not assumed.** Performance claims in code comments and
  changelog entries carry the number that was measured and where.
- **Plain ASCII in code and comments.** No emojis, no smart quotes.
- **models.yaml is written only through `config.py`'s writers** - it is a
  user-owned, comment-carrying file; nothing else may rewrite it.

## UI changes

Render your change before and after with the screenshot harnesses:

```
python scripts/ui_screenshots.py out/        # fake data: first-launch view
python scripts/ui_screenshot_real.py out/    # your registry: loaded view
```

Both have caught real bugs the other could not. Attach the before/after to
your PR.

## Commit style

One logical change per commit, with a message that says what broke or what
was missing, what changed, and how it was verified. The existing history is
the template.
