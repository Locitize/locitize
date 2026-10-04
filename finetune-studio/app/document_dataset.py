"""Any document -> training-ready dataset (M18.6).

Turns the files a user actually has - notes, manuals, exports - into chat-format
training rows the trainer accepts, so "fine-tune on my data" starts from a PDF
or a folder of markdown, not from hand-written JSONL.

Two row modes, layered:

1. STRUCTURAL (always available, no model needed): each document is chunked and
   turned into continuation and recitation exercises - the honest SFT framing of
   "learn this text" that needs no external intelligence to build.
2. GENERATED Q&A (optional): if LOCITIZE is serving a model on this machine, the
   chunks are sent to that local OpenAI-compatible endpoint and the model writes
   question/answer pairs about them. The user's own model, on their own machine,
   preparing their own data - nothing leaves localhost.

Extraction is deliberately dependency-light: txt/md/csv are native, .docx is
parsed from its XML with the stdlib (a docx IS a zip), .pdf uses pypdf when the
studio venv has it and says exactly what to install when it does not.

Pure functions over bytes/strings; the Streamlit app owns all UI.
"""

import csv
import io
import json
import re
import urllib.request
import zipfile

CHUNK_TARGET_CHARS = 1600
CHUNK_OVERLAP_CHARS = 200
DEFAULT_SYSTEM = "You are a helpful assistant."


class ExtractionError(Exception):
    """A document could not be turned into text; message says what to do."""


# --------------------------------------------------------------------------- #
# Extraction
# --------------------------------------------------------------------------- #


def extract_text(filename, data):
    """Bytes of a supported document -> plain text. Raises ExtractionError."""
    name = (filename or "").lower()
    if name.endswith((".txt", ".md", ".markdown")):
        return _decode(data)
    if name.endswith(".csv"):
        return _csv_to_text(data)
    if name.endswith(".docx"):
        return _docx_to_text(data)
    if name.endswith(".pdf"):
        return _pdf_to_text(data)
    raise ExtractionError(
        f"'{filename}': unsupported type. Supported: .txt .md .csv .docx .pdf "
        f"(or upload ready-made .jsonl through the dataset uploader instead)."
    )


def _decode(data):
    for encoding in ("utf-8", "utf-16", "cp1252"):
        try:
            return data.decode(encoding)
        except (UnicodeDecodeError, AttributeError):
            continue
    raise ExtractionError("could not decode the file as text (tried utf-8/utf-16/cp1252)")


def _csv_to_text(data):
    """Each row becomes one 'header: value' line-group - keeps tabular meaning."""
    text = _decode(data)
    reader = csv.reader(io.StringIO(text))
    rows = [r for r in reader if any(cell.strip() for cell in r)]
    if not rows:
        return ""
    header = rows[0]
    out = []
    for row in rows[1:]:
        pairs = [
            f"{(header[i] if i < len(header) else f'col{i}').strip()}: {cell.strip()}"
            for i, cell in enumerate(row)
            if cell.strip()
        ]
        if pairs:
            out.append(". ".join(pairs) + ".")
    return "\n\n".join(out) if out else "\n".join(",".join(r) for r in rows)


def _docx_to_text(data):
    """A .docx is a zip; paragraphs live in word/document.xml as <w:p>/<w:t>."""
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            xml = zf.read("word/document.xml").decode("utf-8", errors="replace")
    except (zipfile.BadZipFile, KeyError) as exc:
        raise ExtractionError(f"not a readable .docx: {exc}")
    paragraphs = []
    for para in re.findall(r"<w:p[ >].*?</w:p>", xml, flags=re.S):
        runs = re.findall(r"<w:t[^>]*>(.*?)</w:t>", para, flags=re.S)
        text = "".join(runs)
        text = (
            text.replace("&amp;", "&").replace("&lt;", "<")
            .replace("&gt;", ">").replace("&quot;", '"').replace("&apos;", "'")
        )
        if text.strip():
            paragraphs.append(text.strip())
    return "\n\n".join(paragraphs)


def _pdf_to_text(data):
    try:
        from pypdf import PdfReader
    except ImportError:
        raise ExtractionError(
            "PDF support needs the 'pypdf' package in the studio environment: "
            "finetune-studio\\.venv\\Scripts\\pip install pypdf"
        )
    try:
        reader = PdfReader(io.BytesIO(data))
        pages = [(page.extract_text() or "") for page in reader.pages]
    except Exception as exc:  # noqa: BLE001 - pypdf raises many types
        raise ExtractionError(f"could not read the PDF: {exc}")
    text = "\n\n".join(p.strip() for p in pages if p.strip())
    if not text:
        raise ExtractionError(
            "the PDF contains no extractable text (likely a scan; OCR it first)"
        )
    return text


# --------------------------------------------------------------------------- #
# Chunking + structural rows
# --------------------------------------------------------------------------- #


def chunk_text(text, target=CHUNK_TARGET_CHARS, overlap=CHUNK_OVERLAP_CHARS):
    """Paragraph-aware chunks of roughly `target` chars with a little overlap."""
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text or "") if p.strip()]
    chunks, current = [], ""
    for para in paragraphs:
        if current and len(current) + len(para) + 2 > target:
            chunks.append(current)
            current = current[-overlap:] if overlap else ""
        current = (current + "\n\n" + para).strip() if current else para
    if current.strip():
        chunks.append(current.strip())
    # A pathological single huge paragraph still gets split.
    out = []
    for chunk in chunks:
        while len(chunk) > target * 2:
            out.append(chunk[: target * 2])
            chunk = chunk[target * 2 - overlap :]
        out.append(chunk)
    return [c for c in out if len(c) >= 200]  # drop fragments too small to teach


def _row(user, assistant, system=DEFAULT_SYSTEM):
    return {
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ]
    }


def structural_rows(chunks, source_name, system=DEFAULT_SYSTEM):
    """Continuation + recitation exercises per chunk - no model required."""
    rows = []
    for chunk in chunks:
        cut = max(200, int(len(chunk) * 0.4))
        head, tail = chunk[:cut].strip(), chunk[cut:].strip()
        if head and tail:
            rows.append(
                _row(
                    f"Continue this passage from {source_name}:\n\n{head}",
                    tail,
                    system,
                )
            )
        opening = " ".join(chunk.split()[:8])
        rows.append(
            _row(
                f"Recite the part of {source_name} that begins: \"{opening}...\"",
                chunk,
                system,
            )
        )
    return rows


# --------------------------------------------------------------------------- #
# Generated Q&A via the user's running local model
# --------------------------------------------------------------------------- #

_QA_PROMPT = (
    "Read the passage below and write {n} question-and-answer pairs about it. "
    "Questions must be answerable from the passage alone; answers must be "
    "complete sentences. Reply ONLY with a JSON array like "
    '[{{"question": "...", "answer": "..."}}].\n\nPASSAGE:\n{chunk}'
)


def local_model_available(endpoint="http://127.0.0.1:8080", opener=None):
    """True when LOCITIZE (or any OpenAI-compatible server) answers locally."""
    call = opener or urllib.request.urlopen
    try:
        with call(endpoint.rstrip("/") + "/v1/models", timeout=4) as resp:
            return resp.status == 200
    except Exception:  # noqa: BLE001 - probe: any failure means "not available"
        return False


def generate_qa_rows(
    chunks,
    endpoint="http://127.0.0.1:8080",
    pairs_per_chunk=3,
    system=DEFAULT_SYSTEM,
    opener=None,
    progress=None,
    timeout=180,
):
    """Ask the LOCAL model to write Q&A pairs for each chunk.

    Returns (rows, failures). A chunk whose reply cannot be parsed is counted
    and skipped, never fabricated. Localhost only by intent: the endpoint is the
    user's own machine, so the document never leaves it.
    """
    call = opener or urllib.request.urlopen
    rows, failures = [], 0
    for index, chunk in enumerate(chunks):
        if progress:
            progress(index, len(chunks))
        body = json.dumps(
            {
                "model": "local",
                "messages": [
                    {
                        "role": "user",
                        "content": _QA_PROMPT.format(n=pairs_per_chunk, chunk=chunk),
                    }
                ],
                "temperature": 0.3,
            }
        ).encode("utf-8")
        request = urllib.request.Request(
            endpoint.rstrip("/") + "/v1/chat/completions",
            data=body,
            headers={"Content-Type": "application/json"},
        )
        try:
            with call(request, timeout=timeout) as resp:
                reply = json.loads(resp.read())
            content = reply["choices"][0]["message"]["content"]
            pairs = _parse_pairs(content)
        except Exception:  # noqa: BLE001 - one bad chunk must not kill the build
            pairs = []
        if not pairs:
            failures += 1
            continue
        for pair in pairs:
            rows.append(_row(pair["question"], pair["answer"], system))
    if progress:
        progress(len(chunks), len(chunks))
    return rows, failures


def _parse_pairs(content):
    """Pull a [{'question','answer'}...] array out of a model reply, tolerantly."""
    match = re.search(r"\[.*\]", content or "", flags=re.S)
    if not match:
        return []
    try:
        parsed = json.loads(match.group(0))
    except json.JSONDecodeError:
        return []
    pairs = []
    for item in parsed if isinstance(parsed, list) else []:
        if not isinstance(item, dict):
            continue
        question = str(item.get("question", "")).strip()
        answer = str(item.get("answer", "")).strip()
        if question and answer:
            pairs.append({"question": question, "answer": answer})
    return pairs
