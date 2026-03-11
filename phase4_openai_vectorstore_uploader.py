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

Windows PowerShell example:
python .\phase4_openai_vectorstore_uploader.py `
  --phase3 ".\prepared_corpus_final" `
  --output ".\openai_vectorstores" `
  --project "anatomy"

Requires:
- requests
- OPENAIKEY.txt in current folder, or --key-file
"""

from __future__ import annotations

import argparse
import csv
import json
import mimetypes
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

try:
    import requests
except ImportError:
    raise SystemExit("Missing dependency: requests. Install with: pip install requests")

OPENAI_BASE = "https://api.openai.com/v1"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def safe_json_dump(obj: Any, path: Path) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def write_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = []
    seen = set()
    for row in rows:
        for k in row.keys():
            if k not in seen:
                seen.add(k)
                fields.append(k)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def ensure_dirs(output_dir: Path) -> dict[str, Path]:
    dirs = {
        "root": output_dir,
        "metadata": output_dir / "metadata",
        "reports": output_dir / "reports",
    }
    for p in dirs.values():
        p.mkdir(parents=True, exist_ok=True)
    return dirs


def find_key_file(explicit: Optional[str] = None) -> Path:
    candidates = []
    if explicit:
        candidates.append(Path(explicit).expanduser().resolve())
    cwd = Path.cwd()
    candidates.extend([
        cwd / "OPENAIKEY.txt",
        cwd / "openaikey.txt",
        Path(__file__).resolve().parent / "OPENAIKEY.txt",
    ])
    for path in candidates:
        if path.exists() and path.is_file():
            return path
    raise FileNotFoundError("Could not find OPENAIKEY.txt. Pass --key-file or place it in the working directory.")


def read_api_key(key_file: Path) -> str:
    key = key_file.read_text(encoding="utf-8").strip()
    if not key:
        raise ValueError(f"API key file is empty: {key_file}")
    return key


class OpenAIClient:
    def __init__(self, api_key: str, timeout: int = 300):
        self.session = requests.Session()
        self.session.headers.update({"Authorization": f"Bearer {api_key}"})
        self.timeout = timeout

    def _handle(self, response: requests.Response) -> dict:
        if response.status_code >= 400:
            raise RuntimeError(f"OpenAI API error {response.status_code}: {response.text}")
        return response.json()

    def create_vector_store(self, name: str, metadata: Optional[dict] = None, expires_days: Optional[int] = None) -> dict:
        body: dict[str, Any] = {"name": name}
        if metadata:
            body["metadata"] = metadata
        if expires_days:
            body["expires_after"] = {"anchor": "last_active_at", "days": expires_days}
        r = self.session.post(f"{OPENAI_BASE}/vector_stores", json=body, timeout=self.timeout)
        return self._handle(r)

    def get_vector_store(self, vector_store_id: str) -> dict:
        r = self.session.get(f"{OPENAI_BASE}/vector_stores/{vector_store_id}", timeout=self.timeout)
        return self._handle(r)

    def upload_file(self, file_path: Path, purpose: str = "assistants") -> dict:
        mime = mimetypes.guess_type(file_path.name)[0] or "text/markdown"
        with file_path.open("rb") as f:
            files = {
                "file": (file_path.name, f, mime),
            }
            data = {
                "purpose": purpose,
            }
            r = self.session.post(f"{OPENAI_BASE}/files", data=data, files=files, timeout=self.timeout)
        return self._handle(r)

    def create_file_batch(self, vector_store_id: str, files_payload: list[dict]) -> dict:
        body = {"files": files_payload}
        r = self.session.post(f"{OPENAI_BASE}/vector_stores/{vector_store_id}/file_batches", json=body, timeout=self.timeout)
        return self._handle(r)

    def get_file_batch(self, vector_store_id: str, batch_id: str) -> dict:
        r = self.session.get(f"{OPENAI_BASE}/vector_stores/{vector_store_id}/file_batches/{batch_id}", timeout=self.timeout)
        return self._handle(r)


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
        policy = row.get("recommended_chunk_policy", "")
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


def poll_batch(client: OpenAIClient, vector_store_id: str, batch_id: str, interval: int = 5, max_wait: int = 1800) -> dict:
    start = time.time()
    while True:
        batch = client.get_file_batch(vector_store_id, batch_id)
        status = batch.get("status")
        if status in {"completed", "failed", "cancelled"}:
            return batch
        if time.time() - start > max_wait:
            raise TimeoutError(f"Timed out waiting for batch {batch_id} in vector store {vector_store_id}")
        time.sleep(interval)


def build_report(vector_stores: list[dict], batches: list[dict], uploaded_files: list[dict], project: str, key_file: Path) -> str:
    lines = [
        "# Phase 4 OpenAI Upload Report",
        "",
        f"Generated: {utc_now()}",
        f"Project: {project}",
        f"API key file used: `{key_file}`",
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Upload Phase 3 corpus into project-scoped OpenAI vector stores")
    parser.add_argument("--phase3", required=True, help="Path to prepared_corpus_final")
    parser.add_argument("--output", required=True, help="Path to output folder for vector store metadata")
    parser.add_argument("--project", required=True, help="Project slug/prefix, e.g. anatomy")
    parser.add_argument("--key-file", default=None, help="Path to OPENAIKEY.txt")
    parser.add_argument("--timeout", type=int, default=300, help="HTTP timeout in seconds")
    parser.add_argument("--poll-interval", type=int, default=5, help="Polling interval in seconds")
    parser.add_argument("--expires-days", type=int, default=30, help="Vector store expiration days from last activity")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    phase3_dir = Path(args.phase3).expanduser().resolve()
    output_dir = Path(args.output).expanduser().resolve()
    dirs = ensure_dirs(output_dir)

    key_file = find_key_file(args.key_file)
    api_key = read_api_key(key_file)
    client = OpenAIClient(api_key, timeout=args.timeout)

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
        vs = client.create_vector_store(
            name=item["name"],
            metadata=item["metadata"],
            expires_days=args.expires_days,
        )
        vector_stores.append(vs)
        vector_store_id_by_key[item["key"]] = vs["id"]

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

        file_obj = client.upload_file(file_path)
        uploaded_files.append({
            "output_filename": row["output_filename"],
            "file_id": file_obj["id"],
            "bytes": file_obj.get("bytes"),
            "created_at": file_obj.get("created_at"),
            "purpose": file_obj.get("purpose"),
        })
        file_id_by_output_filename[row["output_filename"]] = file_obj["id"]

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
        batch = client.create_file_batch(vector_store_id, files_payload)
        batch["vector_store_id"] = vector_store_id
        batch_results.append(batch)

        final_batch = poll_batch(
            client,
            vector_store_id=vector_store_id,
            batch_id=batch["id"],
            interval=args.poll_interval,
        )
        final_batch["vector_store_id"] = vector_store_id
        final_batch_states.append(final_batch)

    # Refresh vector stores after processing
    final_vector_stores = [client.get_vector_store(vs_id) for vs_id in vector_store_id_by_key.values()]

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

    report_md = build_report(final_vector_stores, final_batch_states, uploaded_files, project, key_file)
    (dirs["reports"] / "phase4_upload_report.md").write_text(report_md, encoding="utf-8")

    print("Phase 4 complete.")
    for vs in final_vector_stores:
        print(f"Vector store: {vs.get('name','')} | id: {vs.get('id','')} | status: {vs.get('status','')} | total files: {vs.get('file_counts',{}).get('total',0)} | completed: {vs.get('file_counts',{}).get('completed',0)} | failed: {vs.get('file_counts',{}).get('failed',0)}")
    print(f"Output written to: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
