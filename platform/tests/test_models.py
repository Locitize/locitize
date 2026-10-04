"""Model-registry tests: the live models.yaml and start-spec construction.

Loads whatever registry config.py resolves to, asserts every row in it is
well-formed, and asserts build_start_spec() is correct and guards not-launchable
models.

Why some assertions here are conditional (M14 config architecture, DEC-M14-4):
the registry LOCITIZE ships is now a template with ZERO rows, on purpose - a row is
a claim that a specific file exists on this machine, and shipping rows for files
a new user does not have would be fabricated data. The spec models this file
used to demand unconditionally (qwen3-14b, the 27B, and so on) are therefore the
MAINTAINER'S OWN registrations, present in his data root and in nobody else's.

Asserting them unconditionally made the suite pass on one machine and fail on a
clean clone - the precise "works on my machine" failure Milestone 14 exists to
eliminate, and it would have broken AC-M14-1 for any other installer. So the
machine-independent invariants below run everywhere and always, while the
maintainer-registry checks announce themselves as skipped when those models are
not registered. Nothing is weakened on a machine that HAS them: the same
assertions still run there. The spec behaviour they cover (for example D-M4-2's
extended cold-load window) is additionally pinned by synthetic-fixture tests in
this file, which need no particular machine at all.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from config import Config, Model, ModelRegistryData, Settings
from models import ModelRegistry

PLATFORM_DIR = Path(__file__).resolve().parent.parent

# The models the maintainer has registered locally, in the order M4/M5 recorded.
# Absent from a fresh install by design; see the module docstring.
MAINTAINER_SPEC_IDS = [
    "qwen3-14b",
    "deepseek-r1-distill-14b",
    "qwen3-6-27b",
    "qwen2-5-vl",
]


def _live_registry_ids() -> list[str]:
    """Model ids in whatever registry this machine resolves to (possibly none)."""
    _settings, data, _issues = Config.load(PLATFORM_DIR)
    return [m.id for m in data.models]


def requires_registered(model_id: str):
    """Skip a maintainer-registry assertion when that model is not on this box.

    A skip here is the honest outcome, not a hidden failure: the test is making a
    claim about a specific .gguf on a specific machine, and on a machine without
    it the claim has no meaning. pytest reports the skip and its reason.
    """
    return pytest.mark.skipif(
        model_id not in _live_registry_ids(),
        reason=(
            f"'{model_id}' is not registered on this machine - the shipped registry "
            "is empty by design (M14), so this maintainer-registry check does not apply"
        ),
    )


VALID_STATUSES = {"installed", "future", "disabled"}


def assert_row_is_well_formed(model: Model) -> None:
    """The registry row invariants, in one place so both tests below use them.

    Shared deliberately: the live-registry test that applies these runs over
    zero rows on a clean install, so the rules themselves need a caller that is
    never empty. Keeping one function means the two callers cannot drift apart.
    """
    assert model.name and model.description
    assert model.context_size > 0
    assert model.gpu_layers >= -1
    assert model.status in VALID_STATUSES


def test_every_row_in_the_live_registry_is_well_formed():
    """Machine-independent invariant: whatever is registered must be valid.

    This holds on a fresh install and on a machine with fifty owner-added
    models. Note what it does NOT do on a fresh install: with zero rows the loop
    body never runs, so this test alone proves nothing there - which is why the
    two synthetic tests below carry the same invariants on every machine
    (Review M-7).
    """
    _settings, data, _issues = Config.load(PLATFORM_DIR)
    for m in data.models:
        assert_row_is_well_formed(m)


def test_row_invariants_hold_for_a_synthetic_registry_on_any_machine():
    """The same invariants, applied to rows that exist everywhere.

    A constructed registry has no dependence on what any particular person has
    downloaded, so this is the coverage a clean install actually gets. It also
    proves the shared checker accepts a legitimate row - without that, the
    rejection test below could be passing because the checker rejects
    everything.
    """
    data = ModelRegistryData(
        version=1,
        models=[
            Model(
                id="ready", name="Ready", description="a normal installed row",
                location="/locitize-test/models/ready.gguf",
                context_size=8192, gpu_layers=-1, status="installed",
            ),
            Model(
                id="planned", name="Planned", description="a not-yet-downloaded row",
                location="/locitize-test/models/planned.gguf",
                context_size=32768, gpu_layers=0, status="future",
            ),
            Model(
                id="off", name="Off", description="a row the owner turned off",
                location="/locitize-test/models/off.gguf",
                context_size=4096, gpu_layers=20, status="disabled",
            ),
        ],
    )
    assert len(data.models) == 3, "this test must never run over an empty registry"
    for model in data.models:
        assert_row_is_well_formed(model)


@pytest.mark.parametrize(
    "field, value, expected_warning",
    [
        ("status", "installled", "status"),          # a typo in the status word
        ("context_size", 0, "context_size"),         # a window of nothing
        ("context_size", -1, "context_size"),        # ... or of less than nothing
        ("gpu_layers", -2, "gpu_layers"),            # below the "all layers" value
    ],
)
def test_a_malformed_row_is_reported_by_the_config_validator(
    field, value, expected_warning
):
    """A bad row must be REPORTED, not silently accepted (Review M-7).

    The invariants above say what a good row looks like; this says what happens
    to a bad one. The validator is reached directly rather than through a file
    on disk because the point is the rule, not YAML parsing - and because a test
    that writes a registry file has to pick a machine to write it on.

    `_validate` is private to config.py, and using it here is a deliberate
    exception: it is the single place these invariants are enforced for real, so
    testing anything else would be testing a copy of the rule instead of the
    rule.
    """
    from config import _validate  # noqa: PLC0415 - see the docstring above

    row = Model(
        id="probe", name="Probe", description="d",
        location="/locitize-test/models/probe.gguf",
        context_size=8192, gpu_layers=-1, status="installed",
    )
    setattr(row, field, value)
    issues: list = []
    _validate(Settings(), ModelRegistryData(version=1, models=[row]), issues)
    messages = [i.message for i in issues if "probe" in i.message]
    assert any(expected_warning in m for m in messages), issues


@requires_registered("qwen3-14b")
def test_maintainer_registry_still_carries_every_spec_model_in_order():
    """On the maintainer's machine the M4/M5 spec models are all present, in order.

    The owner has since added further models of their own (abliterated variants,
    their own fine-tunes), which is exactly what the registry is for, so this
    asserts the spec models are present IN FILE ORDER rather than freezing the
    exact list - a frozen list turns every legitimate owner registration into a
    red suite.
    """
    ids = _live_registry_ids()
    for spec_id in MAINTAINER_SPEC_IDS:
        assert spec_id in ids, f"spec model '{spec_id}' is missing from models.yaml"
    positions = [ids.index(spec_id) for spec_id in MAINTAINER_SPEC_IDS]
    assert positions == sorted(positions), "spec models are no longer in file order"


def test_installed_split_honours_status_on_any_machine():
    """The installed/future split itself is machine-independent, so prove it here.

    The check above needs the maintainer's registry; this one uses a synthetic
    one, so the RULE (status 'future' is never launchable) stays covered on a
    fresh install where that registry does not exist.
    """
    data = ModelRegistryData(
        version=1,
        models=[
            Model(
                id="ready", name="Ready", description="d",
                location="/locitize-test/models/ready.gguf",
                context_size=8192, gpu_layers=-1, status="installed",
            ),
            Model(
                id="later", name="Later", description="d",
                location="/locitize-test/models/later.gguf",
                context_size=8192, gpu_layers=-1, status="future",
            ),
        ],
    )
    installed_ids = {m.id for m in ModelRegistry(data, Settings()).installed()}
    assert installed_ids == {"ready"}


def _registry_with_llama_path() -> ModelRegistry:
    settings = Settings()
    settings.paths.llama_cpp = "llama-server.exe"
    data = ModelRegistryData(
        version=1,
        models=[
            Model(
                id="qwen3-14b",
                name="Qwen3 14B",
                description="d",
                location="/locitize-test/models/qwen3-14b.gguf",
                context_size=32768,
                gpu_layers=-1,
                status="installed",
            )
        ],
    )
    return ModelRegistry(data, settings)


def test_build_start_spec_is_data_driven():
    """The start spec embeds the configured binary, model path, port, and flags."""
    registry = _registry_with_llama_path()
    spec = registry.build_start_spec("qwen3-14b")
    assert spec.command[0] == "llama-server.exe"
    assert "/locitize-test/models/qwen3-14b.gguf" in spec.command
    assert "127.0.0.1" in spec.command  # loopback host, never 0.0.0.0
    assert spec.port == 8080
    # The spec carries a bare health_path (not a full URL); the loopback host is
    # composed at probe time so a crafted path cannot move the probe (SEC-1).
    assert spec.health_path == "/health"


def test_build_start_spec_uses_global_ready_timeout_by_default():
    """D-M4-2: a model with no per-model ready_timeout_s inherits the global one."""
    registry = _registry_with_llama_path()
    spec = registry.build_start_spec("qwen3-14b")
    # Settings() default services.ready_timeout_s is 60.0.
    assert spec.ready_timeout_s == 60.0
    # D-M4-3: the llama.cpp child log is appended, not truncated.
    assert spec.append_log is True


def test_build_start_spec_honors_per_model_ready_timeout():
    """D-M4-2: an explicit per-model ready_timeout_s overrides the global default."""
    settings = Settings()
    settings.paths.llama_cpp = "llama-server.exe"
    data = ModelRegistryData(
        models=[
            Model(
                id="big",
                name="Big",
                description="d",
                location="/locitize-test/models/big.gguf",
                context_size=10000,
                gpu_layers=60,
                status="installed",
                ready_timeout_s=240,
            )
        ],
    )
    registry = ModelRegistry(data, settings)
    spec = registry.build_start_spec("big")
    assert spec.ready_timeout_s == 240.0


@requires_registered("qwen3-6-27b")
def test_maintainer_27b_has_extended_ready_timeout():
    """D-M4-2: the maintainer's 27B row carries the longer cold-load window.

    The RULE this depends on - a per-model ready_timeout_s is honoured by
    build_start_spec - is proved machine-independently by the synthetic-fixture
    test just above, which is why skipping this one on a fresh install loses no
    coverage of LOCITIZE's behaviour.
    """
    _settings, data, _issues = Config.load(PLATFORM_DIR)
    registry = ModelRegistry(data, _settings)
    model = registry.get("qwen3-6-27b")
    assert model is not None
    assert model.ready_timeout_s == 240


def test_invalid_per_model_ready_timeout_warns_and_falls_back():
    """D-M4-2: a non-positive/invalid ready_timeout_s is reported and ignored."""
    from config import _build_model

    issues: list = []
    model = _build_model(
        {"id": "m", "context_size": 8192, "gpu_layers": -1, "ready_timeout_s": 0},
        0,
        issues,
    )
    assert model is not None
    assert model.ready_timeout_s is None  # dropped -> falls back to global default
    assert any("ready_timeout_s" in i.message for i in issues)


def test_build_start_spec_applies_ctx_and_gpu_overrides():
    """--smoke-start overrides for ctx_size/gpu_layers flow into the argv."""
    registry = _registry_with_llama_path()
    spec = registry.build_start_spec("qwen3-14b", ctx_size=4096, gpu_layers=10)
    # The overridden values replace the model-row defaults in the argv pairs.
    assert spec.command[spec.command.index("--ctx-size") + 1] == "4096"
    assert spec.command[spec.command.index("--n-gpu-layers") + 1] == "10"


def test_build_start_spec_guards_missing_location():
    """A model with no location cannot build a start spec (honest guard)."""
    settings = Settings()
    settings.paths.llama_cpp = "llama-server.exe"
    data = ModelRegistryData(
        models=[
            Model(
                id="x",
                name="X",
                description="d",
                location="",  # unset
                context_size=8192,
                gpu_layers=-1,
                status="installed",
            )
        ]
    )
    registry = ModelRegistry(data, settings)
    try:
        registry.build_start_spec("x")
        raise AssertionError("expected ValueError for missing location")
    except ValueError:
        pass


def test_build_start_spec_guards_missing_llama_path():
    """No configured llama.cpp path -> a clear ValueError, not a broken launch."""
    data = ModelRegistryData(
        models=[
            Model(
                id="x",
                name="X",
                description="d",
                location="/locitize-test/m.gguf",
                context_size=8192,
                gpu_layers=-1,
                status="installed",
            )
        ]
    )
    registry = ModelRegistry(data, Settings())  # no llama path
    try:
        registry.build_start_spec("x")
        raise AssertionError("expected ValueError for missing llama path")
    except ValueError:
        pass


# --------------------------------------------------------------------------- #
# Reasoning / thinking control (owner request 2026-09-02)
# --------------------------------------------------------------------------- #


def _registry_with_reasoning(reasoning: dict | None) -> ModelRegistry:
    """The standard one-model registry with a `reasoning` mapping attached."""
    settings = Settings()
    settings.paths.llama_cpp = "llama-server.exe"
    data = ModelRegistryData(
        version=1,
        models=[
            Model(
                id="qwen3-14b",
                name="Qwen3 14B",
                description="d",
                location="/locitize-test/models/qwen3-14b.gguf",
                context_size=32768,
                gpu_layers=-1,
                status="installed",
                reasoning=reasoning,
            )
        ],
    )
    return ModelRegistry(data, settings)


def test_build_start_spec_appends_no_reasoning_flags_by_default():
    """A row with no `reasoning:` must produce a byte-identical argv to before.

    This is the additive contract: every existing models.yaml row keeps starting
    exactly as it did, with llama-server left on its own defaults ('auto'
    detection, the template's own effort, budget -1).
    """
    spec = _registry_with_reasoning(None).build_start_spec("qwen3-14b")
    assert not [a for a in spec.command if str(a).startswith("--reasoning")]


def test_build_start_spec_appends_effort_and_budget_in_order():
    """Flags are appended in a deterministic, testable order with real values."""
    spec = _registry_with_reasoning(
        {"enabled": True, "effort": "low", "budget": 2048}
    ).build_start_spec("qwen3-14b")
    command = [str(a) for a in spec.command]
    tail = command[command.index("--reasoning") :]
    assert tail == [
        "--reasoning",
        "on",
        "--reasoning-effort",
        "low",
        "--reasoning-budget",
        "2048",
    ]


def test_build_start_spec_renders_enabled_as_on_off_not_python_bools():
    """`--reasoning False` is not a value llama-server accepts; 'off' is."""
    spec = _registry_with_reasoning({"enabled": False}).build_start_spec("qwen3-14b")
    command = [str(a) for a in spec.command]
    assert command[command.index("--reasoning") + 1] == "off"
    assert "False" not in command


def test_build_start_spec_reasoning_override_wins_over_the_row():
    """The keyword override lets a benchmark sweep vary effort without a write."""
    registry = _registry_with_reasoning({"effort": "xhigh"})
    spec = registry.build_start_spec("qwen3-14b", reasoning={"effort": "low"})
    command = [str(a) for a in spec.command]
    assert command[command.index("--reasoning-effort") + 1] == "low"
    # Passing None forces reasoning OFF for one scenario, exactly as spec_config
    # does, rather than falling back to the row.
    bare = registry.build_start_spec("qwen3-14b", reasoning=None)
    assert not [a for a in bare.command if str(a).startswith("--reasoning")]


# --------------------------------------------------------------------------- #
# The per-launch fit margin (owner rule 2026-09-03, "do not affect my tok/s")
# --------------------------------------------------------------------------- #


def test_fit_request_carries_the_measured_margin_and_logs_it():
    import gpu_ledger

    asked: list = []

    def fake_budget(binary):
        asked.append(binary)
        return gpu_ledger.FitBudget(
            engine_total_mib=16302, engine_free_mib=14923,
            adapter_used_mib=1700, adapter_total_mib=16303,
        )

    registry = _registry_with_llama_path()
    registry._fit_budget_fn = fake_budget
    spec = registry.build_start_spec("qwen3-14b")  # row gpu_layers -1 = fit
    assert asked == ["llama-server.exe"]  # measured with the binary that will load
    cmd = spec.command
    assert cmd[cmd.index("--n-gpu-layers") + 1] == "-1"
    # 14923 - (16303 - 1700) + 1024
    assert cmd[cmd.index("--fit-target") + 1] == "1344"


def test_explicit_layer_count_takes_no_reading_and_no_margin():
    registry = _registry_with_llama_path()

    def never(binary):
        raise AssertionError("an explicit count must not measure the card")

    registry._fit_budget_fn = never
    spec = registry.build_start_spec("qwen3-14b", gpu_layers=10)
    assert "--fit-target" not in spec.command


def test_unmeasurable_margin_sends_none_so_the_engine_keeps_its_default():
    registry = _registry_with_llama_path()
    registry._fit_budget_fn = lambda binary: None
    spec = registry.build_start_spec("qwen3-14b")
    assert spec.command[spec.command.index("--n-gpu-layers") + 1] == "-1"
    assert "--fit-target" not in spec.command


def test_a_fake_binary_path_measures_nothing_by_default():
    """The default reading needs the real server binary; a test path is not
    one, so the default registry builds a spec without touching a GPU."""
    registry = _registry_with_llama_path()
    spec = registry.build_start_spec("qwen3-14b")
    assert "--fit-target" not in spec.command
