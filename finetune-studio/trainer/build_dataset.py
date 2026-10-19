"""Build a training dataset from documents or a word table (command line).

The desktop Fine-tune page runs this in the fine-tune environment, which has the
PDF reader the desktop's own environment does not. It writes the rows to --out as
JSONL and prints one JSON summary line ({"rows", "notes"}) on stdout; progress
lines start with [BUILD].

  python build_dataset.py documents --out x.jsonl [--qa-pairs 3] file1.pdf file2.md
  python build_dataset.py wordtable --out x.jsonl --table words.md
                          [--identity persona.jsonl] [--templates 16] [--oversample 8]
"""

import argparse
import json
import os
import random
import sys

import dataset_builder
import document_dataset


def build_documents(paths, qa_pairs):
    rows, notes, chunks_by_doc = [], [], []
    for path in paths:
        name = os.path.basename(path)
        try:
            with open(path, "rb") as f:
                text = document_dataset.extract_text(name, f.read())
        except (OSError, document_dataset.ExtractionError) as exc:
            notes.append(f"skipped {name}: {exc}")
            continue
        chunks = document_dataset.chunk_text(text)
        if not chunks:
            notes.append(f"skipped {name}: no usable text found")
            continue
        chunks_by_doc.append((name, chunks))
        rows.extend(document_dataset.structural_rows(chunks, name))
    if qa_pairs and chunks_by_doc:
        if not document_dataset.local_model_available():
            notes.append("no model is running in LoCiTiZe, so no question and answer pairs were written")
        else:
            total = sum(len(c) for _n, c in chunks_by_doc)
            done = 0
            for _name, chunks in chunks_by_doc:
                def tick(i, _n, base=done):
                    print(f"[BUILD] {base + i}/{total}", flush=True)
                qa_rows, fails = document_dataset.generate_qa_rows(chunks, pairs_per_chunk=qa_pairs, progress=tick)
                done += len(chunks)
                rows.extend(qa_rows)
                if fails:
                    notes.append(f"{fails} chunk(s) skipped by the question writer")
    random.Random(42).shuffle(rows)
    return rows, notes


def build_word_table(table_path, identity_path, templates, oversample):
    with open(table_path, "r", encoding="utf-8") as f:
        entries = dataset_builder.parse_word_table(f.read())
    vocab_rows = dataset_builder.build_vocab_rows(entries, templates)
    identity_rows = dataset_builder.load_jsonl_rows(identity_path) if identity_path else []
    rows = dataset_builder.combine_and_shuffle(vocab_rows, identity_rows, oversample)
    notes = [f"{len(entries)} words -> {len(vocab_rows)} rows, plus "
             f"{len(identity_rows) * oversample} identity rows"]
    return rows, notes


def main(argv=None):
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="kind", required=True)
    docs = sub.add_parser("documents")
    docs.add_argument("--out", required=True)
    docs.add_argument("--qa-pairs", type=int, default=0)
    docs.add_argument("files", nargs="+")
    table = sub.add_parser("wordtable")
    table.add_argument("--out", required=True)
    table.add_argument("--table", required=True)
    table.add_argument("--identity", default="")
    table.add_argument("--templates", type=int, default=16)
    table.add_argument("--oversample", type=int, default=8)
    args = parser.parse_args(argv)

    if args.kind == "documents":
        rows, notes = build_documents(args.files, args.qa_pairs)
    else:
        rows, notes = build_word_table(args.table, args.identity, args.templates, args.oversample)
    dataset_builder.write_jsonl(rows, args.out)
    print(json.dumps({"rows": len(rows), "notes": notes}), flush=True)
    return 0 if rows else 1


if __name__ == "__main__":
    sys.exit(main())
