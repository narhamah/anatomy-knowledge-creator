#!/usr/bin/env python3
"""
Phase 3 Concept Hub Builder

Builds vector-store-ready concept hub documents and upload manifests by combining:
- Phase 1 normalized corpus
- Phase 2 concept-document map
- Phase 2.5 refined ontology outputs

Outputs:
- concept hub markdown files
- source copies for upload
- chunk/upload manifests
- reports

Usage:
    python phase3_concept_hub_builder.py \
        --phase1 "./prepared_corpus_phase1" \
        --phase2 "./prepared_corpus_phase2" \
        --phase2_5 "./prepared_corpus_phase2_5" \
        --output "./prepared_corpus_final"
"""

from __future__ import annotations

import argparse
from collections import defaultdict, Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from utils import (
    utc_now, safe_json_dump, write_csv, slugify, strip_frontmatter, load_json, make_dirs,
)


@dataclass
class SourceDoc:
    document_id: str
    title: str
    source_kind: str
    path: Path
    body: str
    paragraphs: list[str]


def load_source_docs(phase1_dir: Path) -> dict[str, SourceDoc]:
    normalized_dir = phase1_dir / "normalized_text"
    docs = {}
    for path in sorted(normalized_dir.glob("*.md")):
        text = path.read_text(encoding="utf-8")
        meta, body = strip_frontmatter(text)
        paragraphs = [p.strip() for p in body.split("\n\n") if p.strip()]
        doc = SourceDoc(
            document_id=meta.get("document_id", path.stem),
            title=meta.get("title", path.stem),
            source_kind=meta.get("source_kind", "unknown"),
            path=path,
            body=body,
            paragraphs=paragraphs,
        )
        docs[doc.document_id] = doc
    return docs


def find_relevant_paragraphs(doc: SourceDoc, aliases: list[str], limit: int = 8) -> list[str]:
    found = []
    seen = set()
    lowered_aliases = [a.lower() for a in aliases if a]
    for para in doc.paragraphs:
        low = para.lower()
        if any(a in low for a in lowered_aliases):
            key = para[:200]
            if key not in seen:
                seen.add(key)
                found.append(para)
        if len(found) >= limit:
            break
    return found


def infer_chunk_policy(source_kind: str, text_len: int) -> tuple[str, int, int]:
    if source_kind == "book":
        return "BOOK_LARGE", 1200, 200
    if source_kind == "research_paper":
        return "PAPER_MEDIUM", 800, 120
    if source_kind == "patent":
        if text_len < 2500:
            return "PATENT_CLAIMS", 400, 80
        return "PATENT_DESCRIPTION", 900, 150
    if text_len > 5000:
        return "GENERAL_LARGE", 1000, 150
    return "GENERAL_MEDIUM", 800, 100


def build_concept_hub(
    concept: dict,
    source_docs: dict[str, SourceDoc],
    alias_lookup: dict[str, list[str]],
    refined_domains_by_id: dict[str, dict],
) -> tuple[str, dict]:
    concept_id = concept["concept_id"]
    label = concept["preferred_label"]
    aliases = concept.get("aliases", [])
    source_ids = concept.get("source_concept_ids", [])
    source_doc_ids = concept.get("source_documents", [])
    domain_id = concept.get("domain_id")
    domain = refined_domains_by_id.get(domain_id, {})
    all_aliases = [label] + aliases + alias_lookup.get(label, [])
    all_aliases = list(dict.fromkeys([a for a in all_aliases if a]))

    selected_docs = []
    evidence_sections = []
    doc_kind_counter = Counter()

    for doc_id in source_doc_ids:
        doc = source_docs.get(doc_id)
        if not doc:
            continue
        selected_docs.append(doc)
        doc_kind_counter[doc.source_kind] += 1
        paras = find_relevant_paragraphs(doc, all_aliases, limit=5)
        if not paras and doc.paragraphs:
            paras = doc.paragraphs[:2]
        evidence_sections.append({
            "document_id": doc.document_id,
            "title": doc.title,
            "source_kind": doc.source_kind,
            "source_path": str(doc.path),
            "selected_paragraphs": paras,
        })

    top_kind = doc_kind_counter.most_common(1)[0][0] if doc_kind_counter else "unknown"
    text_for_chunk = "\n\n".join(
        para for item in evidence_sections for para in item["selected_paragraphs"]
    )
    policy_name, chunk_size, overlap = infer_chunk_policy(top_kind, len(text_for_chunk))

    frontmatter = [
        "---",
        f'title: "Concept Hub - {label.replace(chr(34), chr(39))}"',
        f'document_type: "concept_hub"',
        f'concept_id: "{concept_id}"',
        f'preferred_label: "{label.replace(chr(34), chr(39))}"',
        f'domain_id: "{domain_id or ""}"',
        f'chunk_policy: "{policy_name}"',
        f'chunk_size_tokens: {chunk_size}',
        f'chunk_overlap_tokens: {overlap}',
        "---",
        "",
        f"# Concept: {label}",
        "",
        "## Definition",
        concept.get("definition", "") or "No definition provided.",
        "",
        "## Domain",
        domain.get("label", domain_id or "unassigned"),
        "",
        "## Concept Type",
        concept.get("concept_type", "unknown"),
        "",
        "## Aliases",
        ", ".join(all_aliases) if all_aliases else "none",
        "",
        "## Parent Concepts",
        ", ".join(concept.get("parent_concepts", [])) if concept.get("parent_concepts") else "none",
        "",
        "## Child Concepts",
        ", ".join(concept.get("child_concepts", [])) if concept.get("child_concepts") else "none",
        "",
        "## Related Concepts",
        ", ".join(concept.get("related_concepts", [])) if concept.get("related_concepts") else "none",
        "",
        "## Source Concept IDs",
        ", ".join(source_ids) if source_ids else "none",
        "",
        "## Evidence from Source Documents",
        "",
    ]

    for item in evidence_sections:
        frontmatter.extend([
            f"### {item['title']}",
            f"- document_id: `{item['document_id']}`",
            f"- source_kind: {item['source_kind']}",
            f"- source_path: `{item['source_path']}`",
            "",
        ])
        for para in item["selected_paragraphs"]:
            frontmatter.extend([para, ""])
    md = "\n".join(frontmatter).rstrip() + "\n"

    manifest_row = {
        "output_type": "concept_hub",
        "concept_id": concept_id,
        "preferred_label": label,
        "domain_id": domain_id,
        "domain_label": domain.get("label", ""),
        "source_document_count": len(selected_docs),
        "source_documents": " | ".join([d.document_id for d in selected_docs]),
        "output_filename": f"concept_{slugify(label, 80)}.md",
        "recommended_chunk_policy": policy_name,
        "recommended_chunk_size_tokens": chunk_size,
        "recommended_chunk_overlap_tokens": overlap,
    }
    return md, manifest_row


def copy_source_docs(source_docs: dict[str, SourceDoc], out_sources: Path) -> list[dict]:
    rows = []
    for doc in source_docs.values():
        target_name = f"source_{slugify(doc.title, 70)}__{doc.document_id}.md"
        target = out_sources / target_name
        target.write_text(doc.path.read_text(encoding="utf-8"), encoding="utf-8")
        policy_name, chunk_size, overlap = infer_chunk_policy(doc.source_kind, len(doc.body))
        rows.append({
            "output_type": "source_document",
            "document_id": doc.document_id,
            "title": doc.title,
            "source_kind": doc.source_kind,
            "output_filename": target_name,
            "recommended_chunk_policy": policy_name,
            "recommended_chunk_size_tokens": chunk_size,
            "recommended_chunk_overlap_tokens": overlap,
            "original_path": str(doc.path),
        })
    return rows


def build_reports(
    dirs: dict[str, Path],
    refined_domains: list[dict],
    refined_core_concepts: list[dict],
    concept_rows: list[dict],
    source_rows: list[dict],
) -> None:
    domain_counts = Counter(r["domain_label"] or r["domain_id"] or "unassigned" for r in concept_rows)
    lines = [
        "# Phase 3 Summary",
        "",
        f"Generated: {utc_now()}",
        "",
        f"- refined_domains: {len(refined_domains)}",
        f"- refined_core_concepts: {len(refined_core_concepts)}",
        f"- concept_hub_files: {len(concept_rows)}",
        f"- source_document_files: {len(source_rows)}",
        "",
        "## Concept hubs by domain",
        "",
    ]
    for domain, count in domain_counts.most_common():
        lines.append(f"- {domain}: {count}")
    (dirs["reports"] / "phase3_summary.md").write_text("\n".join(lines), encoding="utf-8")

    upload = [
        "# OpenAI Upload Strategy",
        "",
        "## Recommended vector stores",
        "",
    ]
    for domain in refined_domains:
        upload.extend([
            f"### {domain.get('label', domain.get('domain_id',''))}",
            f"- domain_id: `{domain.get('domain_id','')}`",
            f"- description: {domain.get('description','')}",
            f"- recommended store contents: concept hubs + source files mapped to this domain",
            "",
        ])
    upload.extend([
        "## General recommendation",
        "",
        "- Upload concept hub files and source documents together.",
        "- Use concept hub files as high-level retrieval anchors.",
        "- Use source files as factual evidence backing the concept hubs.",
        "- Preserve the upload manifest so each file can be attached with metadata.",
        "",
    ])
    (dirs["reports"] / "openai_upload_strategy.md").write_text("\n".join(upload), encoding="utf-8")


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build concept hub corpus from Phase 1, 2, and 2.5 outputs")
    parser.add_argument("--phase1", required=True, help="Path to prepared_corpus_phase1")
    parser.add_argument("--phase2", required=True, help="Path to prepared_corpus_phase2")
    parser.add_argument("--phase2_5", required=True, help="Path to prepared_corpus_phase2_5")
    parser.add_argument("--output", required=True, help="Path to final output folder")
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)

    phase1_dir = Path(args.phase1).expanduser().resolve()
    phase2_dir = Path(args.phase2).expanduser().resolve()
    phase2_5_dir = Path(args.phase2_5).expanduser().resolve()
    output_dir = Path(args.output).expanduser().resolve()

    dirs = make_dirs(output_dir, {
        "concepts": "concepts",
        "sources": "sources",
        "metadata": "metadata",
        "reports": "reports",
    })

    source_docs = load_source_docs(phase1_dir)

    phase2_meta = phase2_dir / "metadata"
    phase25_meta = phase2_5_dir / "metadata"

    concept_document_map = load_json(phase2_meta / "concept_document_map.json")
    refined_core_concepts = load_json(phase25_meta / "refined_core_concepts.json")
    refined_domains = load_json(phase25_meta / "refined_domains.json")
    alias_groups = load_json(phase25_meta / "refined_alias_groups.json")

    alias_lookup = defaultdict(list)
    for group in alias_groups:
        canon = group.get("canonical_label", "")
        aliases = group.get("aliases", [])
        if canon:
            alias_lookup[canon].extend(aliases)

    refined_domains_by_id = {d.get("domain_id"): d for d in refined_domains}

    concept_rows = []
    for concept in refined_core_concepts:
        # attach source documents from raw source concept ids
        raw_source_ids = concept.get("source_concept_ids", [])
        source_doc_ids = []
        for raw_id in raw_source_ids:
            source_doc_ids.extend(concept_document_map.get(raw_id, {}).get("document_ids", []))
        source_doc_ids = list(dict.fromkeys(source_doc_ids))
        concept["source_documents"] = source_doc_ids

        md, row = build_concept_hub(concept, source_docs, alias_lookup, refined_domains_by_id)
        filename = row["output_filename"]
        (dirs["concepts"] / filename).write_text(md, encoding="utf-8")
        concept_rows.append(row)

    source_rows = copy_source_docs(source_docs, dirs["sources"])

    upload_manifest = concept_rows + source_rows
    safe_json_dump(upload_manifest, dirs["metadata"] / "upload_manifest.json")
    write_csv(upload_manifest, dirs["metadata"] / "upload_manifest.csv")

    concept_to_hub_map = [
        {
            "concept_id": row["concept_id"],
            "preferred_label": row["preferred_label"],
            "concept_hub_file": row["output_filename"],
            "domain_id": row["domain_id"],
            "source_document_count": row["source_document_count"],
            "source_documents": row["source_documents"],
        }
        for row in concept_rows
    ]
    safe_json_dump(concept_to_hub_map, dirs["metadata"] / "concept_hub_map.json")
    write_csv(concept_to_hub_map, dirs["metadata"] / "concept_hub_map.csv")

    chunk_plan = []
    for row in upload_manifest:
        chunk_plan.append({
            "output_filename": row["output_filename"],
            "output_type": row["output_type"],
            "recommended_chunk_policy": row["recommended_chunk_policy"],
            "recommended_chunk_size_tokens": row["recommended_chunk_size_tokens"],
            "recommended_chunk_overlap_tokens": row["recommended_chunk_overlap_tokens"],
        })
    safe_json_dump(chunk_plan, dirs["metadata"] / "chunk_plan.json")
    write_csv(chunk_plan, dirs["metadata"] / "chunk_plan.csv")

    build_reports(dirs, refined_domains, refined_core_concepts, concept_rows, source_rows)

    print(f"Phase 3 complete.")
    print(f"Concept hubs created: {len(concept_rows)}")
    print(f"Source files copied: {len(source_rows)}")
    print(f"Output written to: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
