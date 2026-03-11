#!/usr/bin/env python3
"""
Phase 1 corpus builder
- scans a source folder recursively
- supports PDF and Microsoft Word files (.docx)
- extracts raw text
- creates source inventory
- performs basic extraction quality assessment
- writes normalized text outputs
- creates manifests and reports

Usage:
    python phase1_corpus_builder.py --source "/path/to/files" --output "./prepared_corpus_phase1"

Notes:
- PDF extraction uses PyMuPDF (`fitz`) if installed, then falls back to `pypdf`.
- Word extraction uses `python-docx` if installed.
- This phase does NOT do concept extraction or ontology building yet.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import hashlib
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional, Any

SUPPORTED_EXTENSIONS = {".pdf", ".docx"}
HEADER_FOOTER_MAX_CANDIDATES = 3
MIN_TEXT_FOR_GOOD_EXTRACTION = 500
MANUAL_REVIEW_MIN_TEXT = 80


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def slugify(text: str, max_len: int = 80) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^\w\s-]", "", text, flags=re.UNICODE)
    text = re.sub(r"[-\s]+", "_", text)
    text = text.strip("_")
    return text[:max_len] or "untitled"


def sha1_file(path: Path) -> str:
    h = hashlib.sha1()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def safe_json_dump(obj: Any, path: Path) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def write_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


@dataclass
class SourceRecord:
    document_id: str
    source_path: str
    filename: str
    extension: str
    file_size_bytes: int
    sha1: str
    source_kind: str
    title_guess: str
    extraction_status: str
    extraction_quality: str
    extraction_notes: str
    raw_char_count: int
    normalized_char_count: int
    page_count: Optional[int]
    paragraph_count: Optional[int]
    language_guess: str
    manual_review_required: bool
    output_text_path: Optional[str]


def discover_files(source_dir: Path) -> list[Path]:
    files = []
    for path in source_dir.rglob("*"):
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            files.append(path)
    return sorted(files)


def classify_file(path: Path) -> str:
    name = path.stem.lower()
    if "patent" in name:
        return "patent"
    if any(k in name for k in ["paper", "study", "journal", "review", "article"]):
        return "research_paper"
    if any(k in name for k in ["book", "chapter", "textbook"]):
        return "book"
    return "unknown"


def extract_pdf(path: Path) -> tuple[str, dict]:
    meta = {"page_count": None, "engine": None, "notes": []}
    # Try PyMuPDF first
    try:
        import fitz  # type: ignore
        doc = fitz.open(path)
        meta["page_count"] = doc.page_count
        meta["engine"] = "pymupdf"
        text_parts = []
        for page in doc:
            try:
                text_parts.append(page.get_text("text"))
            except Exception as e:
                meta["notes"].append(f"page_extract_error:{e}")
        return "\n\n".join(text_parts), meta
    except Exception as e:
        meta["notes"].append(f"pymupdf_failed:{e}")

    # Fall back to pypdf
    try:
        from pypdf import PdfReader  # type: ignore
        reader = PdfReader(str(path))
        meta["page_count"] = len(reader.pages)
        meta["engine"] = "pypdf"
        text_parts = []
        for page in reader.pages:
            try:
                text_parts.append(page.extract_text() or "")
            except Exception as e:
                meta["notes"].append(f"page_extract_error:{e}")
        return "\n\n".join(text_parts), meta
    except Exception as e:
        meta["notes"].append(f"pypdf_failed:{e}")
        raise RuntimeError("; ".join(meta["notes"]))


def extract_docx(path: Path) -> tuple[str, dict]:
    meta = {"page_count": None, "engine": None, "notes": []}
    try:
        from docx import Document  # type: ignore
        doc = Document(str(path))
        meta["engine"] = "python-docx"
        paras = [p.text for p in doc.paragraphs]
        text = "\n".join(paras)
        return text, meta
    except Exception as e:
        meta["notes"].append(f"python-docx_failed:{e}")
        raise RuntimeError("; ".join(meta["notes"]))


def extract_text(path: Path) -> tuple[str, dict]:
    ext = path.suffix.lower()
    if ext == ".pdf":
        return extract_pdf(path)
    if ext == ".docx":
        return extract_docx(path)
    raise ValueError(f"Unsupported file type: {ext}")


def detect_repeated_edge_lines(lines: list[str]) -> set[str]:
    candidates = []
    head = [ln.strip() for ln in lines[:HEADER_FOOTER_MAX_CANDIDATES] if ln.strip()]
    tail = [ln.strip() for ln in lines[-HEADER_FOOTER_MAX_CANDIDATES:] if ln.strip()]
    candidates.extend(head)
    candidates.extend(tail)

    counts = {}
    for c in candidates:
        if len(c) < 3:
            continue
        counts[c] = counts.get(c, 0) + 1

    return {line for line, count in counts.items() if count > 1}


def normalize_text(text: str) -> tuple[str, dict]:
    notes = []
    original = text

    # Normalize line endings
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Remove nulls and odd spacing
    text = text.replace("\x00", "")
    text = re.sub(r"[ \t]+", " ", text)

    # Repair hyphenation across line breaks, e.g. "mecha-\nnical"
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)

    # Convert multiple single line breaks into paragraphs more cleanly:
    # collapse line wraps inside paragraphs, preserve blank-line paragraph breaks
    chunks = re.split(r"\n\s*\n", text)
    cleaned_chunks = []
    for chunk in chunks:
        lines = [ln.strip() for ln in chunk.split("\n") if ln.strip()]
        if not lines:
            continue
        repeated = detect_repeated_edge_lines(lines)
        if repeated:
            filtered = [ln for ln in lines if ln not in repeated]
            if filtered:
                lines = filtered
                notes.append(f"removed_repeated_edge_lines:{len(repeated)}")
        paragraph = " ".join(lines)
        paragraph = re.sub(r"\s+", " ", paragraph).strip()
        if paragraph:
            cleaned_chunks.append(paragraph)

    text = "\n\n".join(cleaned_chunks)

    # Remove isolated page-number-only paragraphs
    before = text
    text = re.sub(r"(?m)^\s*\d+\s*$", "", text)
    if text != before:
        notes.append("removed_numeric_page_lines")

    # Final cleanup
    text = re.sub(r"\n{3,}", "\n\n", text).strip()

    return text, {
        "original_char_count": len(original),
        "normalized_char_count": len(text),
        "normalization_notes": notes,
    }


def guess_title(path: Path, text: str) -> str:
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if lines:
        first = lines[0]
        if 5 <= len(first) <= 180:
            return first
    return path.stem.replace("_", " ").replace("-", " ").strip()


def guess_language(text: str) -> str:
    if not text.strip():
        return "unknown"
    arabic = len(re.findall(r"[\u0600-\u06FF]", text))
    latin = len(re.findall(r"[A-Za-z]", text))
    if arabic > latin * 1.5:
        return "ar"
    if latin > 0:
        return "en_or_latin"
    return "unknown"


def assess_quality(raw_text: str, normalized_text: str, extract_meta: dict) -> tuple[str, str, bool]:
    notes = list(extract_meta.get("notes", []))
    raw_len = len(raw_text.strip())
    norm_len = len(normalized_text.strip())

    if raw_len == 0 or norm_len == 0:
        return "manual_review_required", "no_text_extracted", True

    single_char_words = re.findall(r"\b\w\b", normalized_text)
    single_char_ratio = len(single_char_words) / max(1, len(normalized_text.split()))
    weird_replacement_chars = normalized_text.count("�")

    if norm_len < MANUAL_REVIEW_MIN_TEXT:
        notes.append("very_short_text")
        return "manual_review_required", "; ".join(notes), True

    if weird_replacement_chars > 10 or single_char_ratio > 0.18:
        notes.append("garbled_text_signals")
        return "poor", "; ".join(notes), True

    if norm_len >= MIN_TEXT_FOR_GOOD_EXTRACTION:
        return "good", "; ".join(notes), False

    notes.append("low_text_volume")
    return "usable_with_cleanup", "; ".join(notes), False


def build_document_id(path: Path) -> str:
    digest = sha1_file(path)[:12]
    stem = slugify(path.stem, 36)
    return f"{stem}_{digest}"


def write_markdown_text(
    out_path: Path,
    *,
    title: str,
    source_path: Path,
    document_id: str,
    source_kind: str,
    normalized_text: str,
    page_count: Optional[int],
    language_guess: str,
) -> None:
    front_matter = [
        "---",
        f'title: "{title.replace(chr(34), chr(39))}"',
        f'document_id: "{document_id}"',
        f'source_path: "{str(source_path)}"',
        f'source_kind: "{source_kind}"',
        f'page_count: {page_count if page_count is not None else "null"}',
        f'language_guess: "{language_guess}"',
        "---",
        "",
        f"# {title}",
        "",
    ]
    out_path.write_text("\n".join(front_matter) + normalized_text + "\n", encoding="utf-8")


def ensure_dirs(output_dir: Path) -> dict[str, Path]:
    dirs = {
        "root": output_dir,
        "metadata": output_dir / "metadata",
        "reports": output_dir / "reports",
        "normalized": output_dir / "normalized_text",
        "manual_review": output_dir / "manual_review",
    }
    for p in dirs.values():
        p.mkdir(parents=True, exist_ok=True)
    return dirs


def write_reports(records: list[SourceRecord], dirs: dict[str, Path]) -> None:
    inv_rows = [asdict(r) for r in records]
    safe_json_dump(inv_rows, dirs["metadata"] / "inventory.json")
    write_csv(inv_rows, dirs["metadata"] / "inventory.csv")

    quality_rows = [
        {
            "document_id": r.document_id,
            "filename": r.filename,
            "extraction_quality": r.extraction_quality,
            "manual_review_required": r.manual_review_required,
            "extraction_notes": r.extraction_notes,
            "raw_char_count": r.raw_char_count,
            "normalized_char_count": r.normalized_char_count,
        }
        for r in records
    ]
    safe_json_dump(quality_rows, dirs["reports"] / "extraction_quality_report.json")
    write_csv(quality_rows, dirs["reports"] / "extraction_quality_report.csv")

    md = [
        "# Extraction Quality Report",
        "",
        f"Generated: {utc_now()}",
        "",
        f"Total files processed: {len(records)}",
        "",
        "## Summary",
        "",
    ]
    counts = {}
    for r in records:
        counts[r.extraction_quality] = counts.get(r.extraction_quality, 0) + 1
    for k, v in sorted(counts.items()):
        md.append(f"- {k}: {v}")
    md.extend(["", "## Per file", ""])
    for r in records:
        md.extend([
            f"### {r.filename}",
            f"- document_id: `{r.document_id}`",
            f"- source_kind: {r.source_kind}",
            f"- extraction_status: {r.extraction_status}",
            f"- extraction_quality: {r.extraction_quality}",
            f"- manual_review_required: {r.manual_review_required}",
            f"- raw_char_count: {r.raw_char_count}",
            f"- normalized_char_count: {r.normalized_char_count}",
            f"- page_count: {r.page_count}",
            f"- notes: {r.extraction_notes or 'none'}",
            "",
        ])
    (dirs["reports"] / "extraction_quality_report.md").write_text("\n".join(md), encoding="utf-8")

    manual = [asdict(r) for r in records if r.manual_review_required]
    safe_json_dump(manual, dirs["manual_review"] / "manual_review_queue.json")
    manual_md = ["# Manual Review Queue", "", f"Items: {len(manual)}", ""]
    for r in manual:
        manual_md.extend([
            f"## {r['filename']}",
            f"- document_id: `{r['document_id']}`",
            f"- extraction_quality: {r['extraction_quality']}",
            f"- reason: {r['extraction_notes'] or 'review needed'}",
            "",
        ])
    (dirs["manual_review"] / "manual_review_queue.md").write_text("\n".join(manual_md), encoding="utf-8")

    phase1_md = [
        "# Phase 1 Corpus Builder Summary",
        "",
        f"Generated: {utc_now()}",
        "",
        "## Outputs",
        "",
        "- `metadata/inventory.json`",
        "- `metadata/inventory.csv`",
        "- `reports/extraction_quality_report.md`",
        "- `reports/extraction_quality_report.json`",
        "- `manual_review/manual_review_queue.md`",
        "- `manual_review/manual_review_queue.json`",
        "- `normalized_text/*.md`",
        "",
        "## Counts",
        "",
    ]
    kinds = {}
    for r in records:
        kinds[r.source_kind] = kinds.get(r.source_kind, 0) + 1
    for k, v in sorted(kinds.items()):
        phase1_md.append(f"- {k}: {v}")
    (dirs["reports"] / "phase1_summary.md").write_text("\n".join(phase1_md), encoding="utf-8")


def process_one(path: Path, dirs: dict[str, Path]) -> SourceRecord:
    document_id = build_document_id(path)
    source_kind = classify_file(path)

    try:
        raw_text, meta = extract_text(path)
        normalized_text, norm_meta = normalize_text(raw_text)
        title = guess_title(path, raw_text)
        language_guess = guess_language(normalized_text)

        extraction_quality, notes, manual_review = assess_quality(raw_text, normalized_text, meta)
        all_notes = "; ".join([n for n in [notes] if n])

        out_name = f"{document_id}.md"
        out_path = dirs["normalized"] / out_name
        write_markdown_text(
            out_path,
            title=title,
            source_path=path,
            document_id=document_id,
            source_kind=source_kind,
            normalized_text=normalized_text,
            page_count=meta.get("page_count"),
            language_guess=language_guess,
        )

        paragraph_count = len([p for p in normalized_text.split("\n\n") if p.strip()])

        return SourceRecord(
            document_id=document_id,
            source_path=str(path),
            filename=path.name,
            extension=path.suffix.lower(),
            file_size_bytes=path.stat().st_size,
            sha1=sha1_file(path),
            source_kind=source_kind,
            title_guess=title,
            extraction_status="success",
            extraction_quality=extraction_quality,
            extraction_notes=all_notes,
            raw_char_count=len(raw_text),
            normalized_char_count=norm_meta["normalized_char_count"],
            page_count=meta.get("page_count"),
            paragraph_count=paragraph_count,
            language_guess=language_guess,
            manual_review_required=manual_review,
            output_text_path=str(out_path),
        )
    except Exception as e:
        return SourceRecord(
            document_id=document_id,
            source_path=str(path),
            filename=path.name,
            extension=path.suffix.lower(),
            file_size_bytes=path.stat().st_size,
            sha1=sha1_file(path),
            source_kind=source_kind,
            title_guess=path.stem,
            extraction_status="failed",
            extraction_quality="manual_review_required",
            extraction_notes=str(e),
            raw_char_count=0,
            normalized_char_count=0,
            page_count=None,
            paragraph_count=None,
            language_guess="unknown",
            manual_review_required=True,
            output_text_path=None,
        )


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 1 corpus builder with PDF + DOCX support")
    parser.add_argument("--source", required=True, help="Source folder containing PDFs and DOCX files")
    parser.add_argument("--output", required=True, help="Output folder for prepared phase 1 corpus")
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    source_dir = Path(args.source).expanduser().resolve()
    output_dir = Path(args.output).expanduser().resolve()

    if not source_dir.exists() or not source_dir.is_dir():
        print(f"Source folder does not exist or is not a directory: {source_dir}", file=sys.stderr)
        return 2

    dirs = ensure_dirs(output_dir)
    files = discover_files(source_dir)
    if not files:
        print("No supported files found (.pdf, .docx).", file=sys.stderr)
        return 1

    source_listing = [{"path": str(p), "filename": p.name, "extension": p.suffix.lower()} for p in files]
    safe_json_dump(source_listing, dirs["metadata"] / "source_file_listing.json")
    write_csv(source_listing, dirs["metadata"] / "source_file_listing.csv")

    records = [process_one(path, dirs) for path in files]
    write_reports(records, dirs)

    print(f"Processed {len(records)} files into: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
