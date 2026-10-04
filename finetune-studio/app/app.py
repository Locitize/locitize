import json
import os
import re
import subprocess
import time
from datetime import datetime

import streamlit as st

import dataset_builder
import document_dataset
import locitize_deploy
import model_advisor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# M18.6 (integration fix): when LOCITIZE launches the studio it passes the data
# root via LOCITIZE_DATA_DIR, and the PLATFORM's fine-tune discovery scans
# <data root>/finetune/outputs (finetune.outputs_root - a trained model must
# survive a re-clone of this checkout, DEC-M14-9). The studio therefore writes
# datasets and runs THERE when the env var is set, so every finished run shows
# up on LOCITIZE's Fine-tune page automatically. Standalone use (no env var)
# keeps the studio-local folders exactly as before.
_DATA_ROOT = (os.environ.get("LOCITIZE_DATA_DIR")
              or os.environ.get("LOCITIZE_DATA_DIR") or "").strip()
if _DATA_ROOT:
    DATASETS_DIR = os.path.join(_DATA_ROOT, "finetune", "datasets")
    OUTPUTS_DIR = os.path.join(_DATA_ROOT, "finetune", "outputs")
else:
    DATASETS_DIR = os.path.join(ROOT, "datasets")
    OUTPUTS_DIR = os.path.join(ROOT, "outputs")
DISTILL_DIR = os.path.join(OUTPUTS_DIR, "_distill")
DOCKER_IMAGE = "llm-finetune-trainer"

BASE_MODELS = [
    "unsloth/Qwen2.5-1.5B-Instruct",
    "unsloth/Qwen2.5-7B-Instruct",
    "unsloth/Meta-Llama-3.1-8B-Instruct",
    "unsloth/Phi-3.5-mini-instruct",
]

os.makedirs(DATASETS_DIR, exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)
os.makedirs(DISTILL_DIR, exist_ok=True)

st.set_page_config(page_title="Model Fine-Tuning Studio", layout="wide")

st.markdown(
    """
    <style>
    /* Matches LoCiTiZe (web/index.html, dark theme): the studio is shown inside its page. */
    :root { --bg: #171717; --panel: #212121; --panel-2: #262626; --hover: #2f2f2f;
            --border: rgba(255,255,255,.08); --text: #ececec; --muted: #a3a3a3; --accent: #6ea8ff; }
    html, body, .stApp, [data-testid="stAppViewContainer"], [data-testid="stHeader"] { background: var(--bg) !important; }
    .stApp { font-family: Inter, ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto, sans-serif; color: var(--text); }
    .stApp p, .stApp li, .stApp label { color: var(--text); }
    .block-container { padding-top: 1.4rem; max-width: 1100px; }
    h1 { font-size: 1.35rem !important; font-weight: 600 !important; letter-spacing: 0 !important; padding: 0 0 .2rem !important; }
    h2, h3 { font-size: 1rem !important; font-weight: 600 !important; }
    [data-testid="stCaptionContainer"], .stApp small { color: var(--muted) !important; font-size: .84rem; }
    section[data-testid="stSidebar"] { background: var(--panel) !important; border-right: 1px solid var(--border); }
    section[data-testid="stSidebar"] h2 { font-size: .95rem !important; margin-top: 1rem; }
    div[data-testid="stExpander"] details { border: 1px solid var(--border) !important; border-radius: 10px !important; background: var(--panel); }
    div[data-testid="stExpander"] summary { font-size: .88rem; }
    .stButton > button, .stDownloadButton > button, [data-testid="stFileUploader"] button {
        border-radius: 9px !important; border: 1px solid var(--border) !important; background: var(--panel-2) !important;
        color: var(--text) !important; font-size: .86rem !important; }
    .stButton > button:hover { background: var(--hover) !important; }
    .stButton > button[kind="primary"] { background: var(--text) !important; color: var(--bg) !important; border-color: var(--text) !important; }
    .stButton > button[kind="primary"] p, .stButton > button[kind="primary"] span { color: var(--bg) !important; }
    .stButton > button:disabled { opacity: .45; }
    [data-testid="stFileUploaderDropzone"], .stTextInput input, .stNumberInput input, .stTextArea textarea,
    div[data-baseweb="select"] > div { background: var(--panel) !important; border-color: var(--border) !important; border-radius: 9px !important; }
    .stTabs [data-baseweb="tab"] { font-size: .86rem; }
    .stTabs [aria-selected="true"] { color: var(--text) !important; }
    .stTabs [data-baseweb="tab-highlight"] { background: var(--accent) !important; }
    .studio-card { background: var(--panel); border: 1px solid var(--border); border-radius: 12px;
        padding: 1rem 1.25rem; margin-bottom: 1rem; font-size: .9rem; }
    .studio-pill { display: inline-block; padding: .15rem .65rem; border-radius: 999px; font-size: .74rem;
        font-weight: 600; margin-bottom: .5rem; border: 1px solid var(--border); }
    .pill-idle { color: var(--muted); }
    .pill-running { color: var(--accent); border-color: rgba(110,168,255,.4); }
    .pill-done { color: #3fb950; border-color: rgba(63,185,80,.4); }
    .pill-error { color: #f85149; border-color: rgba(248,81,73,.4); }
    #MainMenu, footer, [data-testid="stToolbar"], [data-testid="stDecoration"] { display: none !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

# Inside LoCiTiZe (?embed=true) its own page already carries the name and the explanation.
if st.query_params.get("inlocitize") != "1":
    st.title("Model Fine-Tuning Studio")
    st.caption("Upload a dataset, pick a base model, and get a fine-tuned model (LoRA, merged weights and a GGUF file), all on your own GPU.")

if "run_name" not in st.session_state:
    st.session_state.run_name = None
if "log_path" not in st.session_state:
    st.session_state.log_path = None
if "process" not in st.session_state:
    st.session_state.process = None
if "base_model" not in st.session_state:
    st.session_state.base_model = None
if "default_system" not in st.session_state:
    st.session_state.default_system = ""
if "chat_result" not in st.session_state:
    st.session_state.chat_result = None
if "built_dataset_path" not in st.session_state:
    st.session_state.built_dataset_path = None
if "distill_process" not in st.session_state:
    st.session_state.distill_process = None
if "distill_log_path" not in st.session_state:
    st.session_state.distill_log_path = None
if "distill_run_name" not in st.session_state:
    st.session_state.distill_run_name = None


def run_infer(model_path, system, prompt, extra_mounts=None):
    cmd = ["docker", "run", "--rm", "--gpus", "all", "-v", "hf-cache:/root/.cache/huggingface"]
    for host_path, container_path in (extra_mounts or {}).items():
        cmd += ["-v", f"{host_path}:{container_path}:ro"]
    cmd += [
        DOCKER_IMAGE, "/workspace/infer.py",
        "--model-path", model_path,
        "--system", system,
        "--prompt", prompt,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if proc.returncode != 0:
        return f"[error] {proc.stderr[-800:]}"
    for line in proc.stdout.splitlines():
        if line.startswith("[RESULT]"):
            return json.loads(line[len("[RESULT]"):])["reply"]
    return f"[error] no result in output:\n{proc.stdout[-800:]}"


def parse_progress(log_text):
    """Pull the last tqdm-style `current/total [` step counter out of the log."""
    matches = re.findall(r"(\d+)/(\d+)\s*\[", log_text)
    if not matches:
        return None
    current, total = matches[-1]
    return int(current), int(total)


def parse_distill_progress(log_text):
    matches = re.findall(r"\[DISTILL\]\s*(\d+)/(\d+)", log_text)
    if not matches:
        return None
    current, total = matches[-1]
    return int(current), int(total)


def parse_losses(log_text):
    """Pull every `'loss': <float>` value the trainer logged, in order."""
    return [float(v) for v in re.findall(r"'loss':\s*([0-9.]+)", log_text)]


def run_status(run_dir):
    log_path = os.path.join(run_dir, "train.log")
    if not os.path.exists(log_path):
        return "No log", "pill-idle"
    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()
    if "[STATUS] DONE" in text:
        return "Done", "pill-done"
    if "Traceback (most recent call last)" in text:
        return "Error", "pill-error"
    return "Incomplete", "pill-idle"


def load_run_meta(run_dir):
    meta_path = os.path.join(run_dir, "run_meta.json")
    if not os.path.exists(meta_path):
        return {}
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def stage_label(log_text):
    if "[STATUS] DONE" in log_text:
        return "Done", "pill-done"
    if "quantize" in log_text.lower() and "gguf_writer" in log_text.lower():
        return "Quantizing GGUF", "pill-running"
    if "gguf_writer" in log_text.lower():
        return "Exporting GGUF", "pill-running"
    if "Merging weights" in log_text or "Merging LoRA" in log_text:
        return "Merging weights", "pill-running"
    if "[STATUS] Training complete" in log_text:
        return "Saving adapter", "pill-running"
    if "it/s]" in log_text or "'loss':" in log_text:
        return "Training", "pill-running"
    return "Starting up", "pill-running"


with st.expander("Distill a dataset from a teacher model (optional)"):
    st.caption(
        "Have a bigger model generate answers to a list of prompts, and save the "
        "(prompt, answer) pairs as a training dataset -- knowledge distillation. "
        "Runs on your GPU like training does; the teacher model loads once and answers every prompt."
    )
    distill_is_running = (
        st.session_state.distill_process is not None
        and st.session_state.distill_process.poll() is None
    )

    teacher_model = st.selectbox(
        "Teacher model", BASE_MODELS, index=1, key="teacher_model",
        help="Pick a bigger/better model to generate the training answers.",
    )
    distill_system = st.text_area(
        "System prompt", value="You are a helpful assistant.", height=70, key="distill_system"
    )
    prompts_text = st.text_area(
        "Prompts (one per line)",
        height=140,
        placeholder="How do I reset my password?\nWhat's your return policy?\n...",
        key="distill_prompts_text",
    )
    prompts_file = st.file_uploader(
        "...or upload prompts (.txt, one per line, or .jsonl with a \"prompt\" field)",
        type=["txt", "jsonl"],
        key="distill_prompts_file",
    )
    dc1, dc2 = st.columns(2)
    distill_max_tokens = dc1.slider("Max new tokens", 32, 512, 200, step=16, key="distill_max_tokens")
    distill_temp = dc2.slider("Temperature", 0.0, 1.5, 0.7, step=0.1, key="distill_temp")
    distill_run_name_input = st.text_input(
        "Distillation run name", placeholder="e.g. handbook-distill-v1", key="distill_run_name_input"
    )

    dstart_col, dstop_col = st.columns(2)
    has_prompts = bool(prompts_text.strip()) or prompts_file is not None
    distill_start = dstart_col.button(
        "▶ Start distillation", type="primary", disabled=(not has_prompts) or distill_is_running,
        use_container_width=True, key="distill_start_btn",
    )
    distill_stop = dstop_col.button(
        "■ Stop", disabled=not distill_is_running, use_container_width=True, key="distill_stop_btn"
    )

    if distill_stop and st.session_state.distill_process is not None:
        st.session_state.distill_process.terminate()
        st.warning("Distillation stopped.")

    if distill_start:
        timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
        safe_name = re.sub(r"[^a-zA-Z0-9_-]+", "-", distill_run_name_input.strip()).strip("-")
        drun_name = f"{safe_name}-{timestamp}" if safe_name else f"distill-{timestamp}"

        if prompts_file is not None:
            ext = "jsonl" if prompts_file.name.endswith(".jsonl") else "txt"
            prompts_path = os.path.join(DATASETS_DIR, f"{drun_name}-prompts.{ext}")
            with open(prompts_path, "wb") as f:
                f.write(prompts_file.getbuffer())
        else:
            prompts_path = os.path.join(DATASETS_DIR, f"{drun_name}-prompts.txt")
            with open(prompts_path, "w", encoding="utf-8") as f:
                f.write(prompts_text)

        drun_dir = os.path.join(DISTILL_DIR, drun_name)
        os.makedirs(drun_dir, exist_ok=True)
        dlog_path = os.path.join(drun_dir, "distill.log")
        doutput_path = os.path.join(drun_dir, "distilled_dataset.jsonl")

        dcmd = [
            "docker", "run", "--rm", "--gpus", "all",
            "-v", f"{DATASETS_DIR}:/data",
            "-v", f"{DISTILL_DIR}:/out",
            "-v", "hf-cache:/root/.cache/huggingface",
            DOCKER_IMAGE, "/workspace/distill.py",
            "--teacher-model", teacher_model,
            "--system", distill_system,
            "--prompts-file", f"/data/{os.path.basename(prompts_path)}",
            "--output", f"/out/{drun_name}/distilled_dataset.jsonl",
            "--max-new-tokens", str(distill_max_tokens),
            "--temperature", str(distill_temp),
        ]

        dlog_file = open(dlog_path, "w", encoding="utf-8")
        dprocess = subprocess.Popen(dcmd, stdout=dlog_file, stderr=subprocess.STDOUT)

        st.session_state.distill_process = dprocess
        st.session_state.distill_log_path = dlog_path
        st.session_state.distill_run_name = drun_name
        st.rerun()

    if st.session_state.distill_log_path and os.path.exists(st.session_state.distill_log_path):
        with open(st.session_state.distill_log_path, "r", encoding="utf-8", errors="ignore") as f:
            dlog_text = f.read()

        if "[STATUS] DONE" in dlog_text:
            dprogress = parse_distill_progress(dlog_text)
            st.markdown('<span class="studio-pill pill-done">Done</span>', unsafe_allow_html=True)
            if dprogress:
                st.caption(f"Generated {dprogress[1]} examples.")
            drun_dir = os.path.join(DISTILL_DIR, st.session_state.distill_run_name)
            doutput_path = os.path.join(drun_dir, "distilled_dataset.jsonl")
            if os.path.exists(doutput_path):
                if st.button("Use this dataset for training", key="use_distilled"):
                    st.session_state.built_dataset_path = doutput_path
                    st.success(f"Set as training dataset: `{doutput_path}`")
        elif distill_is_running:
            dprogress = parse_distill_progress(dlog_text)
            st.markdown('<span class="studio-pill pill-running">Distilling</span>', unsafe_allow_html=True)
            if dprogress:
                current, total = dprogress
                st.progress(min(current / total, 1.0) if total else 0.0, text=f"{current}/{total}")
        elif st.session_state.distill_process is not None and st.session_state.distill_process.poll() not in (None, 0):
            st.error(f"Distillation exited with code {st.session_state.distill_process.poll()}. See log below.")

        with st.expander("Distillation log", expanded=False):
            st.code(dlog_text[-4000:] or "(waiting for output...)", language="text")

        if distill_is_running:
            time.sleep(2)
            st.rerun()

with st.sidebar:
    st.header("Dataset")
    use_built = st.session_state.built_dataset_path is not None
    if use_built:
        st.info(f"Built dataset ready: `{os.path.basename(st.session_state.built_dataset_path)}`")
        use_built = st.checkbox("Use this built dataset", value=True)
    uploaded = None
    if not use_built:
        uploaded = st.file_uploader(
            "Chat-format dataset (.jsonl or .json)",
            type=["jsonl", "json"],
            help='Each row: {"messages": [{"role": "system", ...}, {"role": "user", ...}, {"role": "assistant", ...}]} '
            'or Alpaca-style {"instruction", "input", "output"}',
        )

    with st.expander("Build dataset from your documents"):
        st.caption(
            "Upload anything - notes, manuals, exports (.txt .md .csv .docx .pdf). "
            "LoCiTiZe chunks the text and builds chat-format training rows. "
            "If a model is running in LoCiTiZe, it can also write Q&A pairs "
            "about your content - entirely on this machine."
        )
        doc_files = st.file_uploader(
            "Documents", type=["txt", "md", "markdown", "csv", "docx", "pdf"],
            accept_multiple_files=True, key="doc_files",
        )
        qa_available = document_dataset.local_model_available()
        make_qa = st.checkbox(
            "Also generate Q&A pairs with my running LoCiTiZe model",
            value=qa_available, disabled=not qa_available,
            help="Needs a model running in LoCiTiZe (port 8080). The document "
                 "never leaves this machine.",
        )
        qa_per_chunk = st.slider("Q&A pairs per chunk", 1, 6, 3, disabled=not make_qa)
        doc_dataset_name = st.text_input("Dataset name", placeholder="e.g. my-handbook", key="doc_ds_name")
        if st.button("Build dataset from documents", disabled=not doc_files):
            all_rows, notes = [], []
            chunks_by_doc = []
            for f in doc_files:
                try:
                    text = document_dataset.extract_text(f.name, f.getvalue())
                except document_dataset.ExtractionError as exc:
                    notes.append(f"skipped {f.name}: {exc}")
                    continue
                chunks = document_dataset.chunk_text(text)
                if not chunks:
                    notes.append(f"skipped {f.name}: no usable text found")
                    continue
                chunks_by_doc.append((f.name, chunks))
                all_rows.extend(document_dataset.structural_rows(chunks, f.name))
            qa_fail = 0
            if make_qa and chunks_by_doc:
                bar = st.progress(0.0, text="asking your local model to write Q&A pairs...")
                total = sum(len(c) for _n, c in chunks_by_doc)
                done_count = [0]
                for name, chunks in chunks_by_doc:
                    def tick(i, n):
                        bar.progress(min(1.0, (done_count[0] + i) / max(1, total)))
                    rows, fails = document_dataset.generate_qa_rows(
                        chunks, pairs_per_chunk=qa_per_chunk, progress=tick
                    )
                    done_count[0] += len(chunks)
                    qa_fail += fails
                    all_rows.extend(rows)
                bar.empty()
            if not all_rows:
                st.error("no training rows could be built" + ("; " + "; ".join(notes) if notes else ""))
            else:
                import random as _random
                _random.Random(42).shuffle(all_rows)
                safe = re.sub(r"[^a-zA-Z0-9_-]+", "-", (doc_dataset_name or "documents").strip()).strip("-")
                out_path = os.path.join(DATASETS_DIR, f"{safe}.jsonl")
                dataset_builder.write_jsonl(all_rows, out_path)
                st.session_state.built_dataset_path = out_path
                message = f"Built {len(all_rows)} rows from {len(chunks_by_doc)} document(s) -> `{out_path}`"
                if qa_fail:
                    message += f" ({qa_fail} chunk(s) skipped by the Q&A generator)"
                for n in notes:
                    st.warning(n)
                st.success(message)
                st.rerun()

    with st.expander("Build dataset from a word table instead"):
        st.caption(
            "Upload a reviewed word table (markdown: | # | Word | Meaning | Example sentence |, "
            "example cell format 'Sheng sentence — English translation'). Generates several "
            "phrasing variants per word, oversamples an identity/persona dataset alongside it, "
            "shuffles, and writes a combined training file."
        )
        word_table_file = st.file_uploader("Word table (.md)", type=["md"], key="word_table")
        identity_file = st.file_uploader(
            "Identity/persona examples (.jsonl, optional)", type=["jsonl"], key="identity_file"
        )
        templates_per_word = st.slider(
            "Phrasing templates per word", 1, 16, 16,
            help="More templates = more training rows per word, up to 16 available.",
        )
        identity_oversample = st.slider(
            "Identity oversample multiplier", 0, 30, 8,
            help="How many times to repeat the identity dataset. Aim for identity rows to land "
            "around 5-10% of the total -- higher ratios have destabilized training runs.",
        )
        built_name = st.text_input("Dataset name", placeholder="e.g. my-assistant-v1")
        if st.button("Build combined dataset", disabled=word_table_file is None):
            entries = dataset_builder.parse_word_table(word_table_file.getvalue().decode("utf-8"))
            vocab_rows = dataset_builder.build_vocab_rows(entries, templates_per_word)
            identity_rows = (
                dataset_builder.load_jsonl_rows(identity_file.getvalue())
                if identity_file is not None
                else []
            )
            combined = dataset_builder.combine_and_shuffle(vocab_rows, identity_rows, identity_oversample)
            safe = re.sub(r"[^a-zA-Z0-9_-]+", "-", (built_name or "combined").strip()).strip("-")
            out_path = os.path.join(DATASETS_DIR, f"{safe}.jsonl")
            dataset_builder.write_jsonl(combined, out_path)
            st.session_state.built_dataset_path = out_path
            identity_pct = (
                100 * len(identity_rows) * identity_oversample / len(combined) if combined else 0
            )
            st.success(
                f"Built {len(entries)} words -> {len(vocab_rows)} vocab rows + "
                f"{len(identity_rows) * identity_oversample} identity rows "
                f"({identity_pct:.1f}% of total) = {len(combined)} rows -> `{out_path}`"
            )

    st.header("Base model")
    # M18.6: suggestions come live from huggingface.co, filtered to what THIS
    # GPU can QLoRA-train (a user-initiated request - nothing phones home on
    # its own). The known-good defaults remain as the offline fallback, and a
    # hand-typed model id always wins.
    if "advisor_choices" not in st.session_state:
        st.session_state.advisor_choices = list(BASE_MODELS)
        st.session_state.advisor_note = ""
    if st.button("Suggest models that fit my GPU", key="advise_btn"):
        with st.spinner("reading your GPU and asking huggingface.co..."):
            vram_gb = model_advisor.detect_vram_gb()
            suggestions, note = model_advisor.suggest(vram_gb)
        if suggestions:
            st.session_state.advisor_choices = [
                f"{c['id']}  ({c['params_b']:g}B, ~{c['est_gb']} GB to train)"
                for c in suggestions
            ]
        st.session_state.advisor_note = note
    if st.session_state.advisor_note:
        st.caption(st.session_state.advisor_note)
    picked = st.selectbox("Model", st.session_state.advisor_choices, index=0)
    custom_model = st.text_input(
        "Or any Hugging Face model id",
        placeholder="e.g. org/Model-7B-Instruct",
        help="Overrides the pick above when filled in.",
    )
    model_name = (custom_model.strip() or picked.split("  (")[0]).strip()

    custom_run_name = st.text_input(
        "Model name (optional)",
        placeholder="e.g. support-tone-v1",
        help="Used as the output folder name and GGUF filename. Leave blank to auto-generate a timestamped name.",
    )

    with st.expander("Training settings"):
        max_steps = st.slider("Max steps", 10, 500, 60, step=10)
        learning_rate = st.select_slider(
            "Learning rate", options=[1e-5, 5e-5, 1e-4, 2e-4, 3e-4, 5e-4], value=2e-4
        )
        lora_r = st.select_slider("LoRA rank (r)", options=[4, 8, 16, 32, 64], value=16)
        batch_size = st.select_slider("Batch size", options=[1, 2, 4, 8], value=2)
        grad_accum = st.select_slider("Gradient accumulation", options=[1, 2, 4, 8], value=4)
        gguf_quant = st.selectbox("GGUF quantization", ["f16", "q8_0", "q4_k_m"], index=2)

    with st.expander("Run history"):
        past_runs = sorted(
            (d for d in os.listdir(OUTPUTS_DIR) if os.path.isdir(os.path.join(OUTPUTS_DIR, d))),
            key=lambda d: os.path.getmtime(os.path.join(OUTPUTS_DIR, d)),
            reverse=True,
        )
        if not past_runs:
            st.caption("No runs yet.")
        for run in past_runs[:12]:
            run_dir = os.path.join(OUTPUTS_DIR, run)
            label, pill_class = run_status(run_dir)
            rc1, rc2 = st.columns([3, 1])
            rc1.markdown(
                f'<div style="font-size:0.85rem; padding-top:0.35rem;">{run}<br>'
                f'<span class="studio-pill {pill_class}">{label}</span></div>',
                unsafe_allow_html=True,
            )
            if rc2.button("Load", key=f"load-{run}"):
                meta = load_run_meta(run_dir)
                log_path = os.path.join(run_dir, "train.log")
                st.session_state.run_name = run
                st.session_state.log_path = log_path if os.path.exists(log_path) else None
                st.session_state.process = None
                st.session_state.base_model = meta.get("base_model", BASE_MODELS[0])
                st.session_state.default_system = meta.get("default_system", "")
                st.session_state.chat_result = None
                if not meta:
                    st.session_state.pop("_run_meta_missing_warning", None)
                    st.session_state["_run_meta_missing_warning"] = run
                st.rerun()
        if st.session_state.get("_run_meta_missing_warning") == st.session_state.run_name:
            st.caption(
                "No saved metadata for this run (trained before run history existed) -- "
                "base model defaulted, system prompt left blank."
            )

is_running = st.session_state.process is not None and st.session_state.process.poll() is None

start_disabled = (uploaded is None and not use_built) or is_running
col1, col2 = st.columns([1, 1])
start = col1.button("▶ Start training", type="primary", disabled=start_disabled, use_container_width=True, key="train_start")
stop = col2.button("■ Stop", disabled=not is_running, use_container_width=True, key="train_stop")

if stop and st.session_state.process is not None:
    st.session_state.process.terminate()
    st.warning("Training stopped.")

if start:
    timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    safe_name = re.sub(r"[^a-zA-Z0-9_-]+", "-", custom_run_name.strip()).strip("-")
    run_name = f"{safe_name}-{timestamp}" if safe_name else f"run-{timestamp}"

    default_system = ""
    if use_built:
        dataset_path = st.session_state.built_dataset_path
    else:
        dataset_path = os.path.join(DATASETS_DIR, f"{run_name}-{uploaded.name}")
        with open(dataset_path, "wb") as f:
            f.write(uploaded.getbuffer())
        try:
            first_line = uploaded.getvalue().decode("utf-8").splitlines()[0]
            first_row = json.loads(first_line) if dataset_path.endswith(".jsonl") else json.loads(uploaded.getvalue())[0]
            default_system = next(
                (m["content"] for m in first_row["messages"] if m["role"] == "system"), ""
            )
        except Exception:
            pass

    run_out_dir = os.path.join(OUTPUTS_DIR, run_name)
    os.makedirs(run_out_dir, exist_ok=True)
    log_path = os.path.join(run_out_dir, "train.log")

    with open(os.path.join(run_out_dir, "run_meta.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "base_model": model_name,
                "default_system": default_system,
                "dataset_path": dataset_path,
                "max_steps": max_steps,
                "learning_rate": learning_rate,
                "lora_r": lora_r,
                "batch_size": batch_size,
                "grad_accum": grad_accum,
                "gguf_quant": gguf_quant,
            },
            f,
        )

    cmd = [
        "docker", "run", "--rm", "--gpus", "all",
        "-v", f"{DATASETS_DIR}:/data",
        "-v", f"{OUTPUTS_DIR}:/out",
        "-v", "hf-cache:/root/.cache/huggingface",
        DOCKER_IMAGE, "/workspace/train.py",
        "--dataset", f"/data/{os.path.basename(dataset_path)}",
        "--model", model_name,
        "--output-dir", "/out",
        "--run-name", run_name,
        "--max-steps", str(max_steps),
        "--learning-rate", str(learning_rate),
        "--lora-r", str(lora_r),
        "--batch-size", str(batch_size),
        "--grad-accum", str(grad_accum),
        "--gguf-quant", gguf_quant,
    ]

    log_file = open(log_path, "w", encoding="utf-8")
    process = subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT)

    st.session_state.run_name = run_name
    st.session_state.log_path = log_path
    st.session_state.process = process
    st.session_state.base_model = model_name
    st.session_state.default_system = default_system
    st.session_state.chat_result = None
    st.rerun()

st.markdown("### Progress")

if st.session_state.log_path and os.path.exists(st.session_state.log_path):
    with open(st.session_state.log_path, "r", encoding="utf-8", errors="ignore") as f:
        log_text = f.read()

    label, pill_class = stage_label(log_text)
    st.markdown(f'<span class="studio-pill {pill_class}">{label}</span>', unsafe_allow_html=True)

    progress = parse_progress(log_text)
    if progress and "DONE" not in log_text:
        current, total = progress
        st.progress(min(current / total, 1.0) if total else 0.0, text=f"{current}/{total}")

    losses = parse_losses(log_text)
    if losses:
        st.markdown("**Training loss**")
        st.line_chart(losses, height=220)

    with st.expander("Full log", expanded=not is_running and "DONE" not in log_text):
        st.code(log_text[-6000:] or "(waiting for output...)", language="text")

    if "DONE" in log_text:
        st.markdown(f'<div class="studio-card"><b>Training complete:</b> {st.session_state.run_name}</div>', unsafe_allow_html=True)
        run_dir = os.path.join(OUTPUTS_DIR, st.session_state.run_name)
        lora_dir = os.path.join(run_dir, "lora_adapter")
        merged_dir = os.path.join(run_dir, "merged_model")
        gguf_files = [f for f in os.listdir(run_dir) if f.endswith(".gguf")] if os.path.isdir(run_dir) else []

        tab_artifacts, tab_compare, tab_deploy = st.tabs(["Artifacts", "Compare", "Deploy to LoCiTiZe"])

        with tab_artifacts:
            st.write(f"**LoRA adapter:** `{lora_dir}`")
            st.write(f"**Merged model:** `{merged_dir}`")
            for g in gguf_files:
                st.write(f"**GGUF:** `{os.path.join(run_dir, g)}`")
            if st.button("Open output folder"):
                os.startfile(run_dir)

        with tab_compare:
            st.caption("Sends the same prompt to the original base model and to your fine-tuned merged model.")
            system_prompt = st.text_area("System prompt", value=st.session_state.default_system, height=80)
            user_prompt = st.text_input("User message", placeholder="Ask something your dataset should have taught it...")
            if st.button("Compare replies", disabled=not user_prompt):
                with st.spinner("Running base model..."):
                    before = run_infer(st.session_state.base_model, system_prompt, user_prompt)
                with st.spinner("Running fine-tuned model..."):
                    after = run_infer(
                        "/finetuned/merged_model", system_prompt, user_prompt,
                        extra_mounts={merged_dir: "/finetuned/merged_model"},
                    )
                st.session_state.chat_result = {"before": before, "after": after}

            if st.session_state.chat_result:
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown("**Before (base model)**")
                    st.markdown(f'<div class="studio-card">{st.session_state.chat_result["before"]}</div>', unsafe_allow_html=True)
                with c2:
                    st.markdown("**After (fine-tuned)**")
                    st.markdown(f'<div class="studio-card">{st.session_state.chat_result["after"]}</div>', unsafe_allow_html=True)

        with tab_deploy:
            if not gguf_files:
                st.info("No GGUF file found for this run.")
            else:
                st.caption(
                    "Copies the GGUF into LoCiTiZe's model store and registers (or updates) its entry "
                    "in models.yaml -- the same steps a manual deployment would take."
                )
                locitize_yaml_path = st.text_input(
                    "LoCiTiZe models.yaml path", value=locitize_deploy.DEFAULT_LOCITIZE_MODELS_YAML
                )
                deploy_gguf_name = st.selectbox("GGUF to deploy", gguf_files)
                d_id = st.text_input("Model id (registry key)", placeholder="e.g. my-assistant")
                d_name = st.text_input("Display name", placeholder="e.g. My Assistant")
                d_desc = st.text_input("Description", placeholder="One line describing this model")
                d_ctx = st.number_input("Context size", value=8192, step=1024)
                d_gpu_layers = st.number_input("GPU layers (999 = full offload)", value=999)
                d_prompt = st.text_input("Recommended system prompt", value=st.session_state.default_system)
                d_notes = st.text_area(
                    "Notes", placeholder="Training recipe, known issues, comparisons to prior versions..."
                )
                if st.button("Deploy", type="primary", disabled=not (d_id and d_name)):
                    dest_filename = f"{d_name}.{gguf_quant.upper()}.gguf".replace(" ", "")
                    src = os.path.join(run_dir, deploy_gguf_name)
                    dest_path = locitize_deploy.deploy_gguf(src, dest_filename)
                    # M18.6: the fine-tuned distinction is AUTOMATIC, not left to
                    # a free-text field the user may leave blank. Every deployed
                    # row carries its training provenance in description+notes,
                    # so the Models page always shows what this model is.
                    run_meta = load_run_meta(run_dir)
                    base_used = run_meta.get("base_model", "") if isinstance(run_meta, dict) else ""
                    provenance = "Fine-tuned in the LoCiTiZe studio"
                    if base_used:
                        provenance += f" (base: {base_used})"
                    provenance += f", run {os.path.basename(run_dir)}."
                    final_desc = d_desc.strip() or provenance
                    if provenance not in final_desc:
                        final_desc = f"{provenance} {final_desc}".strip()
                    final_notes = f"{provenance} {d_notes.strip()}".strip()
                    entry = locitize_deploy.build_entry_block(
                        model_id=d_id.strip(),
                        name=d_name.strip(),
                        description=final_desc,
                        location=dest_path,
                        context_size=int(d_ctx),
                        gpu_layers=int(d_gpu_layers),
                        recommended_prompt=d_prompt.strip(),
                        quantization=gguf_quant.upper(),
                        vram_estimate_mb=2000,
                        notes=final_notes,
                    )
                    was_update = locitize_deploy.upsert_model_entry(locitize_yaml_path, d_id.strip(), entry)
                    ok, msg = locitize_deploy.validate_yaml(locitize_yaml_path)
                    if ok:
                        action = "Updated" if was_update else "Registered"
                        st.success(f"{action} '{d_id}' in LoCiTiZe. GGUF at `{dest_path}`. Registry: {msg}.")
                    else:
                        st.error(f"Deployed the file, but the registry may be malformed: {msg}")

    elif is_running:
        time.sleep(2)
        st.rerun()
    elif st.session_state.process is not None and st.session_state.process.poll() not in (None, 0):
        st.error(f"Training process exited with code {st.session_state.process.poll()}. See log above.")
else:
    st.markdown(
        '<div class="studio-card">Upload a dataset in the sidebar and click <b>Start training</b> to begin.</div>',
        unsafe_allow_html=True,
    )
