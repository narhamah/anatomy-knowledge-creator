#!/usr/bin/env python3
"""
Phase 2.5 OpenAI Ontology Refiner

Reads Phase 2 heuristic ontology outputs and uses the OpenAI Responses API
to produce a cleaner, corpus-grounded refinement layer:
- refined core concepts
- refined domains
- alias groups
- bridge/generic terms to demote
- refined ontology relationships
- refinement report

Default API key source:
    OPENAIKEY.txt

Usage:
    python phase2_5_openai_refiner.py --phase2 "./prepared_corpus_phase2" --output "./prepared_corpus_phase2_5"

Optional:
    python phase2_5_openai_refiner.py --phase2 "./prepared_corpus_phase2" --output "./prepared_corpus_phase2_5" --model "gpt-5-mini"
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

try:
    import requests
except ImportError as e:
    raise SystemExit("Missing dependency: requests. Install with: pip install requests")


API_URL = "https://api.openai.com/v1/responses"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def safe_json_dump(obj: Any, path: Path) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def ensure_dirs(output_dir: Path) -> dict[str, Path]:
    dirs = {
        "root": output_dir,
        "metadata": output_dir / "metadata",
        "reports": output_dir / "reports",
    }
    for p in dirs.values():
        p.mkdir(parents=True, exist_ok=True)
    return dirs


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def find_key_file(explicit: Optional[str] = None) -> Path:
    candidates = []
    if explicit:
        candidates.append(Path(explicit).expanduser().resolve())

    cwd = Path.cwd()
    candidates.extend([
        cwd / "OPENAIKEY.txt",
        cwd / "openaikey.txt",
        cwd / ".env.openai",
        Path(__file__).resolve().parent / "OPENAIKEY.txt",
    ])

    for path in candidates:
        if path.exists() and path.is_file():
            return path

    raise FileNotFoundError(
        "Could not find OPENAIKEY.txt. Pass --key-file or place OPENAIKEY.txt in the working directory."
    )


def read_api_key(key_file: Path) -> str:
    text = key_file.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"API key file is empty: {key_file}")
    return text


def compact_concept_record(concept: dict, relationships: dict[str, dict], doc_map: dict[str, dict]) -> dict:
    cid = concept["concept_id"]
    rel = relationships.get(cid, {})
    dmap = doc_map.get(cid, {})
    return {
        "concept_id": cid,
        "label": concept.get("preferred_label", ""),
        "type": concept.get("concept_type", "other"),
        "support_count": concept.get("support_count", dmap.get("support_count", 0)),
        "alternate_labels": concept.get("alternate_labels", [])[:8],
        "cluster_id": concept.get("cluster_id"),
        "parent_concepts": rel.get("parent_concepts", [])[:5],
        "child_concepts": rel.get("child_concepts", [])[:5],
        "related_concepts": rel.get("related_concepts", [])[:8],
        "definition_hint": concept.get("definition_inferred_from_corpus", ""),
        "representative_phrases": concept.get("representative_phrases", [])[:2],
        "source_documents": concept.get("source_documents", dmap.get("document_ids", []))[:8],
    }


def build_payload(
    concept_index: list[dict],
    relationships: dict[str, dict],
    doc_map: dict[str, dict],
    clusters: list[dict],
    *,
    max_core_concepts: int,
) -> dict:
    compact_concepts = [
        compact_concept_record(c, relationships, doc_map)
        for c in concept_index
    ]

    compact_clusters = []
    for cl in clusters:
        compact_clusters.append({
            "cluster_id": cl.get("cluster_id"),
            "cluster_label": cl.get("cluster_label"),
            "dominant_type": cl.get("dominant_type"),
            "size": cl.get("size"),
            "concept_ids": cl.get("concept_ids", [])[:30],
            "rationale": cl.get("rationale", ""),
        })

    return {
        "task": "refine corpus-derived scientific ontology",
        "max_core_concepts": max_core_concepts,
        "input_summary": {
            "concept_count": len(compact_concepts),
            "cluster_count": len(compact_clusters),
        },
        "concepts": compact_concepts,
        "clusters": compact_clusters,
    }


def build_prompt(payload: dict) -> str:
    return f"""
You are refining a corpus-derived scientific ontology.

Your job is to improve a raw ontology that was heuristically extracted from a normalized technical corpus.
The corpus is domain-specific and scientific. The current ontology is noisy and over-fragmented.

Refinement goals:
1. Merge shallow variants into stronger core concepts.
2. Demote generic bridge terms that connect everything but add little retrieval value.
3. Build a clean domain structure from the concepts actually present.
4. Preserve only concepts that are materially useful for retrieval and question answering.
5. Produce a corpus-grounded ontology, not a generic one.

Key constraints:
- Derive everything from the supplied concepts and relationships.
- Prefer fewer deeper concepts over many shallow surface terms.
- Keep the ontology usable for retrieval systems.
- Preserve scientific precision.
- Preserve distinctions between structures, properties, damage mechanisms, repair mechanisms, methods, materials, and patent-claim abstractions when supported by the data.
- Identify ambiguous terms explicitly.
- Use stable, descriptive labels.

Return ONLY valid JSON with this exact top-level structure:

{{
  "refined_domains": [
    {{
      "domain_id": "string",
      "label": "string",
      "description": "string",
      "core_concept_ids": ["string"],
      "rationale": "string"
    }}
  ],
  "refined_core_concepts": [
    {{
      "concept_id": "string",
      "preferred_label": "string",
      "concept_type": "string",
      "definition": "string",
      "aliases": ["string"],
      "parent_concepts": ["string"],
      "child_concepts": ["string"],
      "related_concepts": ["string"],
      "domain_id": "string",
      "source_concept_ids": ["string"],
      "confidence": "high|medium|low",
      "notes": "string"
    }}
  ],
  "alias_groups": [
    {{
      "canonical_label": "string",
      "aliases": ["string"],
      "rationale": "string"
    }}
  ],
  "bridge_terms_to_demote": [
    {{
      "label": "string",
      "reason": "string",
      "suggested_handling": "string"
    }}
  ],
  "ambiguous_terms": [
    {{
      "label": "string",
      "ambiguity": "string",
      "recommended_disambiguation": "string"
    }}
  ],
  "refined_ontology": {{
    "root_domains": ["string"],
    "relationships": [
      {{
        "source": "string",
        "relationship": "parent_of|child_of|related_to|part_of|measured_by|damaged_by|repaired_by|claimed_in",
        "target": "string",
        "rationale": "string"
      }}
    ]
  }},
  "summary": {{
    "overall_assessment": "string",
    "main_cleanup_actions": ["string"],
    "ontology_quality_notes": "string"
  }}
}}

Additional instructions:
- Create no more than {payload["max_core_concepts"]} refined_core_concepts.
- The "source_concept_ids" field should contain the IDs from the raw concept list that were merged into that refined concept.
- Do not preserve raw concept clutter just because it appeared in the input.
- Only create domains that are clearly supported by the supplied concepts.
- Use the supplied concept IDs where appropriate so the result can be mapped back to the raw ontology.

INPUT DATA:
{json.dumps(payload, ensure_ascii=False, indent=2)}
""".strip()


def call_openai_responses(api_key: str, model: str, prompt: str, timeout: int = 300) -> dict:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    body = {
        "model": model,
        "input": [
            {
                "role": "developer",
                "content": [
                    {
                        "type": "input_text",
                        "text": "You are an expert ontology refiner for scientific corpora. Return only valid JSON."
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": prompt
                    }
                ],
            },
        ],
        "max_output_tokens": 16000,
    }

    response = requests.post(API_URL, headers=headers, json=body, timeout=timeout)
    if response.status_code >= 400:
        raise RuntimeError(f"OpenAI API error {response.status_code}: {response.text}")
    return response.json()


def extract_text_output(resp_json: dict) -> str:
    if isinstance(resp_json.get("output_text"), str) and resp_json["output_text"].strip():
        return resp_json["output_text"].strip()

    chunks = []

    for item in resp_json.get("output", []):
        # common assistant-message style
        if item.get("type") == "message":
            for content in item.get("content", []):
                if content.get("type") in {"output_text", "text"}:
                    txt = content.get("text")
                    if isinstance(txt, str):
                        chunks.append(txt)
                    elif isinstance(txt, dict) and "value" in txt:
                        chunks.append(str(txt["value"]))
        # fallback
        if "content" in item and isinstance(item["content"], str):
            chunks.append(item["content"])

    text = "\n".join(x for x in chunks if x).strip()
    if text:
        return text

    raise ValueError("Could not extract text output from the OpenAI response.")


def parse_json_from_text(text: str) -> dict:
    text = text.strip()
    # raw JSON
    try:
        return json.loads(text)
    except Exception:
        pass

    # fenced JSON
    fence_match = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, flags=re.DOTALL)
    if fence_match:
        return json.loads(fence_match.group(1))

    # first JSON object heuristic
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return json.loads(text[start:end+1])

    raise ValueError("Model output did not contain valid JSON.")


def build_markdown_report(refined: dict, model: str, source_phase2: Path, key_file: Path) -> str:
    domains = refined.get("refined_domains", [])
    concepts = refined.get("refined_core_concepts", [])
    aliases = refined.get("alias_groups", [])
    bridges = refined.get("bridge_terms_to_demote", [])
    ambiguous = refined.get("ambiguous_terms", [])
    relationships = refined.get("refined_ontology", {}).get("relationships", [])
    summary = refined.get("summary", {})

    lines = [
        "# Phase 2.5 Refinement Report",
        "",
        f"Generated: {utc_now()}",
        f"Model: {model}",
        f"Phase 2 source: `{source_phase2}`",
        f"API key file used: `{key_file}`",
        "",
        "## Summary",
        "",
        f"- Refined domains: {len(domains)}",
        f"- Refined core concepts: {len(concepts)}",
        f"- Alias groups: {len(aliases)}",
        f"- Bridge terms to demote: {len(bridges)}",
        f"- Ambiguous terms: {len(ambiguous)}",
        f"- Ontology relationships: {len(relationships)}",
        "",
        f"Overall assessment: {summary.get('overall_assessment', '')}",
        "",
        "## Main cleanup actions",
        "",
    ]

    for action in summary.get("main_cleanup_actions", []):
        lines.append(f"- {action}")

    lines.extend(["", f"Ontology quality notes: {summary.get('ontology_quality_notes', '')}", "", "## Domains", ""])

    for domain in domains:
        lines.extend([
            f"### {domain.get('label', domain.get('domain_id',''))}",
            f"- domain_id: `{domain.get('domain_id','')}`",
            f"- description: {domain.get('description','')}",
            f"- rationale: {domain.get('rationale','')}",
            f"- core_concept_ids: {', '.join(domain.get('core_concept_ids', [])) if domain.get('core_concept_ids') else 'none'}",
            "",
        ])

    lines.extend(["## Bridge Terms to Demote", ""])
    for item in bridges:
        lines.extend([
            f"### {item.get('label','')}",
            f"- reason: {item.get('reason','')}",
            f"- suggested_handling: {item.get('suggested_handling','')}",
            "",
        ])

    lines.extend(["## Ambiguous Terms", ""])
    for item in ambiguous:
        lines.extend([
            f"### {item.get('label','')}",
            f"- ambiguity: {item.get('ambiguity','')}",
            f"- recommended_disambiguation: {item.get('recommended_disambiguation','')}",
            "",
        ])

    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 2.5 OpenAI ontology refinement")
    parser.add_argument("--phase2", required=True, help="Path to prepared_corpus_phase2")
    parser.add_argument("--output", required=True, help="Path to output folder for Phase 2.5")
    parser.add_argument("--key-file", default=None, help="Path to OPENAIKEY.txt")
    parser.add_argument("--model", default="gpt-5-mini", help="OpenAI model to use")
    parser.add_argument("--max-core-concepts", type=int, default=60, help="Maximum number of refined core concepts")
    parser.add_argument("--timeout", type=int, default=300, help="HTTP timeout in seconds")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    phase2_dir = Path(args.phase2).expanduser().resolve()
    output_dir = Path(args.output).expanduser().resolve()
    out_dirs = ensure_dirs(output_dir)

    key_file = find_key_file(args.key_file)
    api_key = read_api_key(key_file)

    metadata_dir = phase2_dir / "metadata"
    if not metadata_dir.exists():
        raise SystemExit(f"Missing metadata folder: {metadata_dir}")

    concept_index = load_json(metadata_dir / "concept_index.json")
    concept_relationships = load_json(metadata_dir / "concept_relationships.json")
    concept_document_map = load_json(metadata_dir / "concept_document_map.json")
    concept_clusters = load_json(metadata_dir / "concept_clusters.json")

    payload = build_payload(
        concept_index=concept_index,
        relationships=concept_relationships,
        doc_map=concept_document_map,
        clusters=concept_clusters,
        max_core_concepts=args.max_core_concepts,
    )
    safe_json_dump(payload, out_dirs["metadata"] / "refinement_input_payload.json")

    prompt = build_prompt(payload)
    (out_dirs["reports"] / "refinement_prompt.txt").write_text(prompt, encoding="utf-8")

    raw_response = call_openai_responses(
        api_key=api_key,
        model=args.model,
        prompt=prompt,
        timeout=args.timeout,
    )
    safe_json_dump(raw_response, out_dirs["metadata"] / "raw_openai_response.json")

    text_output = extract_text_output(raw_response)
    (out_dirs["reports"] / "raw_model_output.txt").write_text(text_output, encoding="utf-8")

    refined = parse_json_from_text(text_output)
    safe_json_dump(refined, out_dirs["metadata"] / "refined_ontology_bundle.json")
    safe_json_dump(refined.get("refined_domains", []), out_dirs["metadata"] / "refined_domains.json")
    safe_json_dump(refined.get("refined_core_concepts", []), out_dirs["metadata"] / "refined_core_concepts.json")
    safe_json_dump(refined.get("alias_groups", []), out_dirs["metadata"] / "refined_alias_groups.json")
    safe_json_dump(refined.get("bridge_terms_to_demote", []), out_dirs["metadata"] / "bridge_terms_to_demote.json")
    safe_json_dump(refined.get("ambiguous_terms", []), out_dirs["metadata"] / "ambiguous_terms.json")
    safe_json_dump(refined.get("refined_ontology", {}), out_dirs["metadata"] / "refined_ontology.json")
    safe_json_dump(refined.get("summary", {}), out_dirs["metadata"] / "refinement_summary.json")

    report_md = build_markdown_report(refined, args.model, phase2_dir, key_file)
    (out_dirs["reports"] / "phase2_5_refinement_report.md").write_text(report_md, encoding="utf-8")

    print(f"Phase 2.5 refinement complete.")
    print(f"Phase 2 source: {phase2_dir}")
    print(f"Output written to: {output_dir}")
    print(f"Model used: {args.model}")
    print(f"Refined domains: {len(refined.get('refined_domains', []))}")
    print(f"Refined core concepts: {len(refined.get('refined_core_concepts', []))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
