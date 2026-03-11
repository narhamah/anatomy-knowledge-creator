#!/usr/bin/env python3
"""
Phase 4 OpenAI Vector Store Uploader

Creates project-scoped vector stores and uploads the final corpus built in Phase 3.

Default behavior:
- creates one vector store for concept hub files
- creates one vector store for source files
- uploads files
- attaches per-file attributes from upload_manifest.json
- polls until processing completes
- writes vector store IDs and upload results to disk

Usage:
    python phase4_openai_vectorstore_uploader.py \
        --phase3 "./prepared_corpus_final" \
        --output "./openai_vectorstores" \
        --project "anatomy"
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional

from utils import utc_now, safe_json_dump, write_csv, load_json, make_dirs
from openai_client import get_openai_client, load_api_key


def sanitize_attributes(row: dict) -> dict:
    allowed = {}
    for k, v in row.items():
        if v is None:
            continue
        if isinstance(v, (str, int, float, bool)):
            val = v
        else:
            val = str(v)
        k2 = str(k)[:64]
        if isinstance(val, str):
            val = val[:512]
        allowed[k2] = val
    return allowed


def parse_manifest(phase3_dir: Path) -> tuple[list[dict], Path, Path]:
    metadata_dir = phase3_dir / "metadata"
    manifest_path = metadata_dir / "upload_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing upload_manifest.json: {manifest_path}")
    manifest = load_json(manifest_path)
    return manifest, phase3_dir / "concepts", phase3_dir / "sources"


def build_file_payloads(
    manifest: list[dict],
    file_uploads: dict[str, str],
) -> dict[str, list[dict]]:
    by_store = defaultdict(list)

    for row in manifest:
        file_id = file_uploads[row["output_filename"]]
        output_type = row["output_type"]

        attrs = sanitize_attributes({
            "output_type": row.get("output_type"),
            "concept_id": row.get("concept_id", ""),
            "preferred_label": row.get("preferred_label", ""),
            "domain_id": row.get("domain_id", ""),
            "domain_label": row.get("domain_label", ""),
            "document_id": row.get("document_id", ""),
            "title": row.get("title", ""),
            "source_kind": row.get("source_kind", ""),
        })

        chunking_strategy = None
        size = row.get("recommended_chunk_size_tokens")
        overlap = row.get("recommended_chunk_overlap_tokens")
        if isinstance(size, int) and isinstance(overlap, int):
            chunking_strategy = {
                "type": "static",
                "static": {
                    "max_chunk_size_tokens": size,
                    "chunk_overlap_tokens": overlap,
                },
            }

        item = {
            "file_id": file_id,
            "attributes": attrs,
        }
        if chunking_strategy:
            item["chunking_strategy"] = chunking_strategy

        if output_type == "concept_hub":
            by_store["concepts"].append(item)
        elif output_type == "source_document":
            by_store["sources"].append(item)
        else:
            by_store["misc"].append(item)

    return by_store


def poll_batch(client, vector_store_id: str, batch_id: str, interval: int = 5, max_wait: int = 1800) -> dict:
    start = time.time()
    while True:
        batch = client.vector_stores.file_batches.retrieve(
            vector_store_id=vector_store_id,
            batch_id=batch_id,
        )
        batch_dict = batch.model_dump() if hasattr(batch, "model_dump") else batch
        status = batch_dict.get("status")
        if status in {"completed", "failed", "cancelled"}:
            return batch_dict
        if time.time() - start > max_wait:
            raise TimeoutError(f"Timed out waiting for batch {batch_id} in vector store {vector_store_id}")
        time.sleep(interval)


def build_report(vector_stores: list[dict], batches: list[dict], uploaded_files: list[dict], project: str) -> str:
    lines = [
        "# Phase 4 OpenAI Upload Report",
        "",
        f"Generated: {utc_now()}",
        f"Project: {project}",
        "",
        "## Vector stores",
        "",
    ]
    for vs in vector_stores:
        lines.extend([
            f"### {vs.get('name','')}",
            f"- id: `{vs.get('id','')}`",
            f"- status: {vs.get('status','')}",
            f"- file_counts: {json.dumps(vs.get('file_counts', {}), ensure_ascii=False)}",
            "",
        ])

    lines.extend(["## File batches", ""])
    for batch in batches:
        lines.extend([
            f"### Vector store `{batch.get('vector_store_id','')}`",
            f"- batch_id: `{batch.get('id','')}`",
            f"- status: {batch.get('status','')}",
            f"- file_counts: {json.dumps(batch.get('file_counts', {}), ensure_ascii=False)}",
            "",
        ])

    lines.extend([
        "## Upload counts",
        "",
        f"- files_uploaded: {len(uploaded_files)}",
        "",
    ])
    return "\n".join(lines)


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Upload Phase 3 corpus into project-scoped OpenAI vector stores")
    parser.add_argument("--phase3", required=True, help="Path to prepared_corpus_final")
    parser.add_argument("--output", required=True, help="Path to output folder for vector store metadata")
    parser.add_argument("--project", required=True, help="Project slug/prefix, e.g. anatomy")
    parser.add_argument("--key-file", default=None, help="Path to API key file (or set OPENAI_API_KEY in .env)")
    parser.add_argument("--timeout", type=int, default=300, help="HTTP timeout in seconds")
    parser.add_argument("--poll-interval", type=int, default=5, help="Polling interval in seconds")
    parser.add_argument("--expires-days", type=int, default=30, help="Vector store expiration days from last activity")
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    phase3_dir = Path(args.phase3).expanduser().resolve()
    output_dir = Path(args.output).expanduser().resolve()
    dirs = make_dirs(output_dir, {
        "metadata": "metadata",
        "reports": "reports",
    })

    client = get_openai_client(
        load_api_key(args.key_file) if args.key_file else None
    )

    manifest, concepts_dir, sources_dir = parse_manifest(phase3_dir)

    project = args.project.strip().lower().replace(" ", "_")
    vector_store_defs = [
        {
            "key": "concepts",
            "name": f"{project}_concepts",
            "metadata": {"project": project, "workspace_type": "concepts"},
        },
        {
            "key": "sources",
            "name": f"{project}_sources",
            "metadata": {"project": project, "workspace_type": "sources"},
        },
    ]

    # Create vector stores
    vector_stores = []
    vector_store_id_by_key = {}
    for item in vector_store_defs:
        expires_after = {"anchor": "last_active_at", "days": args.expires_days}
        vs = client.vector_stores.create(
            name=item["name"],
            metadata=item["metadata"],
            expires_after=expires_after,
        )
        vs_dict = vs.model_dump() if hasattr(vs, "model_dump") else vs
        vector_stores.append(vs_dict)
        vector_store_id_by_key[item["key"]] = vs_dict["id"]

    # Upload files to Files API
    uploaded_files = []
    file_id_by_output_filename = {}
    for row in manifest:
        if row["output_type"] == "concept_hub":
            file_path = concepts_dir / row["output_filename"]
        elif row["output_type"] == "source_document":
            file_path = sources_dir / row["output_filename"]
        else:
            continue

        if not file_path.exists():
            raise FileNotFoundError(f"Missing file to upload: {file_path}")

        with file_path.open("rb") as f:
            file_obj = client.files.create(file=f, purpose="assistants")
        file_dict = file_obj.model_dump() if hasattr(file_obj, "model_dump") else file_obj
        uploaded_files.append({
            "output_filename": row["output_filename"],
            "file_id": file_dict["id"],
            "bytes": file_dict.get("bytes"),
            "created_at": file_dict.get("created_at"),
            "purpose": file_dict.get("purpose"),
        })
        file_id_by_output_filename[row["output_filename"]] = file_dict["id"]

    # Build file batch payloads
    by_store = build_file_payloads(manifest, file_id_by_output_filename)

    # Create file batches
    batch_results = []
    final_batch_states = []
    for store_key in ["concepts", "sources"]:
        files_payload = by_store.get(store_key, [])
        if not files_payload:
            continue
        vector_store_id = vector_store_id_by_key[store_key]
        batch = client.vector_stores.file_batches.create(
            vector_store_id=vector_store_id,
            files=files_payload,
        )
        batch_dict = batch.model_dump() if hasattr(batch, "model_dump") else batch
        batch_dict["vector_store_id"] = vector_store_id
        batch_results.append(batch_dict)

        final_batch = poll_batch(
            client,
            vector_store_id=vector_store_id,
            batch_id=batch_dict["id"],
            interval=args.poll_interval,
        )
        final_batch["vector_store_id"] = vector_store_id
        final_batch_states.append(final_batch)

    # Refresh vector stores after processing
    final_vector_stores = []
    for vs_id in vector_store_id_by_key.values():
        vs = client.vector_stores.retrieve(vector_store_id=vs_id)
        vs_dict = vs.model_dump() if hasattr(vs, "model_dump") else vs
        final_vector_stores.append(vs_dict)

    safe_json_dump(final_vector_stores, dirs["metadata"] / "vector_stores.json")
    safe_json_dump(uploaded_files, dirs["metadata"] / "uploaded_files.json")
    safe_json_dump(batch_results, dirs["metadata"] / "created_file_batches.json")
    safe_json_dump(final_batch_states, dirs["metadata"] / "upload_results.json")

    write_csv(uploaded_files, dirs["metadata"] / "uploaded_files.csv")
    write_csv(final_batch_states, dirs["metadata"] / "upload_results.csv")
    write_csv(
        [
            {
                "vector_store_id": vs["id"],
                "name": vs.get("name", ""),
                "status": vs.get("status", ""),
                "usage_bytes": vs.get("usage_bytes", 0),
                "file_counts_total": vs.get("file_counts", {}).get("total", 0),
                "file_counts_completed": vs.get("file_counts", {}).get("completed", 0),
                "file_counts_failed": vs.get("file_counts", {}).get("failed", 0),
                "file_counts_in_progress": vs.get("file_counts", {}).get("in_progress", 0),
            }
            for vs in final_vector_stores
        ],
        dirs["metadata"] / "vector_stores.csv"
    )

    report_md = build_report(final_vector_stores, final_batch_states, uploaded_files, project)
    (dirs["reports"] / "phase4_upload_report.md").write_text(report_md, encoding="utf-8")

    print("Phase 4 complete.")
    for vs in final_vector_stores:
        print(f"Vector store: {vs.get('name','')} | id: {vs.get('id','')} | status: {vs.get('status','')} | total files: {vs.get('file_counts',{}).get('total',0)} | completed: {vs.get('file_counts',{}).get('completed',0)} | failed: {vs.get('file_counts',{}).get('failed',0)}")
    print(f"Output written to: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
