# LOCITIZE production ops (install / health / boot / GPU)

Keep listeners on `127.0.0.1`.

## Canonical roots

Paths are machine-local. Prefer environment variables; never commit a drive-letter path.

| Role | Resolve as | Notes |
|------|------------|-------|
| **DEV_ROOT** | `$env:LOCITIZE_DEV_ROOT` or git checkout root | Source of truth |
| **Alias** | `$env:SystemDrive\locitize` | Optional directory **junction** -> DEV_ROOT (same files) |
| **Code (platform)** | `<DEV_ROOT>\platform\` | `launcher.py`, scripts, templates |
| **DATA_ROOT** | `<DEV_ROOT>\platform\locitize-data\` | Portable data (bins, models, logs, OWUI store). Override with `LOCITIZE_DATA_DIR` |
| **BIN_ROOT** | `<DATA_ROOT>\bin\` | `llama.cpp`, `whisper.cpp`, etc. |
| **Venv** | `<DEV_ROOT>\.venv\` | Resolved by scripts relative to script location (also `.webui-venv` for OWUI) |
| **Portal** | sibling `agent-portal` checkout (default listen `:4200`) | Optional; probed by stack health |

A common layout is a checkout anywhere on disk with an optional system-drive junction `locitize` pointing at it. Scripts do not hard-code any layout.

Do not hand-edit a "mystery" venv. Scripts under `platform\scripts\` always resolve python as:

1. `<DEV_ROOT>\.venv\Scripts\python.exe`
2. `<platform>\.venv\Scripts\python.exe`
3. `<platform>\runtime\python.exe` (packaged)
4. `python` on PATH

## Single install / update path

`````powershell
cd $env:LOCITIZE_DEV_ROOT   # or the junction alias, same tree
.\platform\scripts\install_or_update.ps1
```

What it does:

1. Prints resolved DEV_ROOT / DATA_ROOT / python
2. `git pull --ff-only` (skip with `-NoPull`)
3. `pip install -r platform\requirements.txt` into the resolved venv (skip with `-SkipDeps`)
4. Optional `-Restart` -> `stop.ps1` then `start.ps1` (desktop)
5. Runs stack health (`health.ps1`) unless `-SkipHealth`

Aliases:

```powershell
.\platform\scripts\update.ps1   # same entry as install_or_update.ps1
```

## Health (one command)

```powershell
.\platform\scripts\health.ps1
# or:
python platform\launcher.py --stack-health
```

Probes (PASS/FAIL): llama `:8080`, router `:8093`, Open WebUI `:8096`, whisper `:8091`, kokoro `:8092`, Agent Portal `:4200`.

## Boot order

1. llama-server (`:8080`)
2. router (`:8093`)
3. Open WebUI (`:8096`)
4. whisper (`:8091`) / kokoro (`:8092`) as needed
5. Agent Portal (`:4200`) — daily driver

`start.ps1` defaults to `--desktop`. Use `-Terminal` for the interactive menu.

## GPU switch (llama.cpp build)

```powershell
.\platform\scripts\gpu_switch.ps1 -Target cuda   # dry-run default
.\platform\scripts\gpu_switch.ps1 -Target cpu -Apply
```

Details and safety notes live with `gpu_switch_lib.py`. Keep the chat model's VRAM clear of optional sidecars (they run on CPU).

## Tailscale Serve (phone / remote on tailnet)

Tailnet only — **no Funnel / public**. Locals stay on `127.0.0.1`.

**Serve layout:** Locitize/OWUI keeps root; Agent Portal has its own port.

Idempotent script (matches live map; do not fight Serve unless applying this exact map):

```powershell
.\platform\scripts\tailscale_serve.ps1
# status only:  .\platform\scripts\tailscale_serve.ps1 -StatusOnly
# clear+apply:  .\platform\scripts\tailscale_serve.ps1 -Reset
```

| URL | Backend |
|-----|---------|
| `https://<magicdns>/` | Open WebUI `127.0.0.1:8096` (root `:443`) |
| `https://<magicdns>:4443/` | Open WebUI `127.0.0.1:8096` (alt Serve URL; WEBUI_URL uses root) |
| `https://<magicdns>:8443/` | Agent Portal `127.0.0.1:4200` (**portal only** — do **not** put Portal on `:443`) |

Example: `https://<host>.<tailnet>.ts.net/` and `:4443` → OWUI; `:8443` → Portal only.

### Open WebUI env on every boot

`platform/webui.py` → `_build_openwebui_env` (launcher `_start_openwebui` → `build_openwebui_spec`) sets:

| Env | Value |
|-----|-------|
| `WEBUI_URL` | Root HTTPS `https://<magicdns>` (no port). `LOCITIZE_WEBUI_URL` overrides it; without Tailscale it falls back to `http://127.0.0.1:<openwebui port>`. `:4443` is alt Serve only, not WEBUI_URL |
| `WEBUI_AUTH` | `False` (single-user default; Open WebUI listens on loopback only) |
| `ENABLE_LOGIN_FORM` | `False` |
| `DATA_DIR` | `<data root>/webui-data` (portable: `platform/locitize-data/webui-data`) |
| `OPENAI_API_BASE_URL(S)` | Router `http://127.0.0.1:8093/v1` when `router.enabled`, else llama.cpp `:8080/v1` |
| `WEBUI_SESSION_COOKIE_SECURE` / `WEBUI_AUTH_COOKIE_SECURE` | `True` (Serve HTTPS) |
| `WEBUI_*_COOKIE_SAME_SITE` | `lax` |
| `FORWARDED_ALLOW_IPS` | `*` (trust Tailscale Serve X-Forwarded-*) |

**Auth-off signin:** with `WEBUI_AUTH=False`, Open WebUI authenticates as `admin@localhost` / password literal `admin`. The row in `webui.db` must hold a bcrypt hash of the literal string `admin` (not another password). Reset the hash if sign-in fails after auth-off.

**Auto-update:** every _start_openwebui runs python -m pip install -U open-webui in the launcher .webui-venv (junction-safe). Fail-soft: log and start previous install on error.

**Single process:** start path kills duplicate `open-webui` PIDs so only one listener owns `:8096`.

Do **not** run `tailscale serve reset` unless intentionally remapping. Path-based `/portal` is not used (SPA absolute assets).

