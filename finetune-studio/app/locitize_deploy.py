"""Deploy a finished GGUF to LOCITIZE: copy the file into LOCITIZE's models
directory and register (or update) its entry in the live models.yaml.

Writes the yaml entry as a textual block insert/replace rather than a full
PyYAML round-trip, because models.yaml is a hand-maintained, heavily
commented file (see its own header) -- a full re-serialize would strip every
comment. This mirrors exactly what a by-hand deployment does.
"""
import os
import re
import shutil


def _data_root():
    """Resolve LOCITIZE's data root the way the platform does, from the
    environment only - this studio runs in its own venv and must ship no
    machine-specific path. LOCITIZE launches the studio with LOCITIZE_DATA_DIR
    set (a path, never a secret); absent that, the standard install default is
    %LOCALAPPDATA%\LOCITIZE. Both fields below are user-editable in the UI, so
    a non-standard layout is always overridable by hand."""
    explicit = (os.environ.get("LOCITIZE_DATA_DIR")
                or os.environ.get("LOCITIZE_DATA_DIR") or "").strip()
    if explicit:
        return explicit
    local = (os.environ.get("LOCALAPPDATA") or "").strip()
    if local:
        return os.path.join(local, "LOCITIZE")
    return os.path.join(os.path.expanduser("~"), ".locitize")


DEFAULT_LOCITIZE_MODELS_YAML = os.path.join(_data_root(), "models.yaml")
DEFAULT_MODELS_DIR = os.path.join(_data_root(), "models")


def deploy_gguf(gguf_src_path, model_filename, models_dir=DEFAULT_MODELS_DIR):
    """Copy the GGUF into LOCITIZE's model store. Returns the destination path."""
    os.makedirs(models_dir, exist_ok=True)
    dest = os.path.join(models_dir, model_filename)
    shutil.copy2(gguf_src_path, dest)
    return dest


def _yaml_str(value):
    """Quote a filesystem path: backslashes become forward slashes (matches
    the style already used throughout models.yaml) and quotes are escaped."""
    escaped = value.replace("\\", "/").replace('"', '\\"')
    return f'"{escaped}"'


def _yaml_text(value):
    """Quote free text: escape backslashes and quotes, but don't touch slashes."""
    escaped = value.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def build_entry_block(
    model_id,
    name,
    description,
    location,
    context_size,
    gpu_layers,
    recommended_prompt,
    quantization,
    vram_estimate_mb,
    notes,
    server_args=None,
):
    server_args = server_args or ["--parallel", "1"]
    args_str = ", ".join(_yaml_str(a) for a in server_args)
    lines = [
        f"  - id: {model_id}",
        f"    name: {_yaml_text(name)}",
        f"    description: {_yaml_text(description)}",
        f"    location: {_yaml_str(location)}",
        f"    context_size: {context_size}",
        f"    gpu_layers: {gpu_layers}",
        f"    recommended_prompt: {_yaml_text(recommended_prompt)}",
        "    benchmark_score: null",
        f"    notes: {_yaml_text(notes)}",
        "    status: installed",
        f"    quantization: {_yaml_text(quantization)}",
        f"    vram_estimate_mb: {vram_estimate_mb}",
        f"    server_args: [{args_str}]",
    ]
    return "\n".join(lines) + "\n"


def upsert_model_entry(yaml_path, model_id, entry_block):
    """Insert entry_block as a new list item, or replace the existing block
    for model_id if one is already present. Returns True if it was an update,
    False if it was a fresh insert."""
    with open(yaml_path, "r", encoding="utf-8") as f:
        text = f.read()

    pattern = re.compile(
        rf"(?m)^  - id: {re.escape(model_id)}\n(?:(?!^  - id: ).*\n)*"
    )
    match = pattern.search(text)
    if match:
        new_text = text[: match.start()] + entry_block + "\n" + text[match.end() :]
        with open(yaml_path, "w", encoding="utf-8") as f:
            f.write(new_text)
        return True

    new_text = text.rstrip("\n") + "\n\n" + entry_block
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(new_text)
    return False


def validate_yaml(yaml_path):
    """Best-effort validation via PyYAML (read-only, never used for writing).
    Returns (ok, message)."""
    try:
        import yaml

        with open(yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return True, f"valid, {len(data.get('models', []))} models"
    except Exception as e:  # noqa: BLE001 -- surfacing any parse error to the UI
        return False, str(e)
