#!/usr/bin/env python3
"""
Phase 2 Ontology Builder
Reads Phase 1 normalized corpus files and builds:
- candidate concepts
- reduced concepts
- concept clusters
- concept relationships / ontology
- concept index
- concept-document map
- reports

This version supports corpus-derived ontology discovery with a local heuristic engine.
It also leaves the output in a clean structure for later OpenAI-assisted refinement if desired.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional

STOPWORDS = {
    "a","an","and","are","as","at","be","been","being","by","for","from","had","has","have","he","her","his",
    "i","if","in","into","is","it","its","may","might","more","most","not","of","on","or","our","such","than",
    "that","the","their","them","there","these","they","this","those","to","was","were","will","with","within",
    "without","would","you","your","we","can","could","should","also","using","used","use","between","during",
    "via","per","over","under","after","before","about","than","then","because","which","who","whom","whose",
    "very","much","many","some","any","all","each","other","another","both","either","neither","one","two",
    "three","however","therefore","thus","etc","e.g","i.e","figure","table","tables","fig","et","al","md",
    "document_id","source_kind","page_count","language_guess","title","source_path"
}

TYPE_CUES = {
    "structure": {"structure","fiber","fibre","cuticle","cortex","medulla","cell","protein","keratin","layer","matrix","filament","microfibril","macrofibril"},
    "property": {"strength","elasticity","modulus","friction","hydration","swelling","porosity","smoothness","mechanical","thermal","stability","resistance","surface","breakage","combability"},
    "damage_type": {"damage","damaged","degradation","oxidation","bleach","bleaching","thermal","uv","weathering","fracture","crack","erosion","denaturation","hydrolysis"},
    "repair_approach": {"repair","restore","rebuild","reconnect","reformation","recombine","recombination","treatment","conditioning","crosslinking","deposition","bonding"},
    "mechanism": {"mechanism","reaction","covalent","noncovalent","thiol","disulfide","bond","interaction","cleavage","formation","pathway","adsorption"},
    "method": {"method","assay","measure","measurement","microscopy","spectroscopy","tensile","analysis","experiment","experimental","protocol","instrument","test"},
    "material": {"molecule","polymer","peptide","amino","acid","surfactant","silicone","oil","protein","keratin","cysteine","cystine","sulfur","sulphur","water"},
    "patent_claim_theme": {"claim","claims","composition","embodiment","formulation","example","method of treating","patent","comprising","wherein","applicant"},
    "application_domain": {"hair","cosmetic","care","treatment","formulation","shampoo","conditioner","serum","bleaching","dyeing"},
    "measurement": {"percent","increase","decrease","measured","tested","evaluated","assessed","result","results","comparison"},
}

PARENT_TYPE_MAP = {
    "structure": "scientific_structure",
    "property": "material_property",
    "damage_type": "damage_mechanism",
    "repair_approach": "repair_mechanism",
    "mechanism": "scientific_mechanism",
    "method": "experimental_method",
    "material": "material_or_agent",
    "patent_claim_theme": "patent_domain",
    "application_domain": "application_domain",
    "measurement": "measurement_or_outcome",
    "other": "other",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


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


def slugify(text: str, max_len: int = 100) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^\w\s-]", "", text, flags=re.UNICODE)
    text = re.sub(r"[-\s]+", "_", text)
    return text.strip("_")[:max_len] or "item"


def strip_frontmatter(md: str) -> tuple[dict[str, str], str]:
    if md.startswith("---\n"):
        parts = md.split("\n---\n", 1)
        if len(parts) == 2:
            raw_meta = parts[0][4:]
            body = parts[1]
            meta = {}
            for line in raw_meta.splitlines():
                if ":" in line:
                    k, v = line.split(":", 1)
                    meta[k.strip()] = v.strip().strip('"')
            return meta, body
    return {}, md


@dataclass
class Document:
    document_id: str
    path: Path
    title: str
    source_kind: str
    body: str
    headings: list[str]
    paragraphs: list[str]
    sentences: list[str]


def find_headings(md_body: str) -> list[str]:
    headings = []
    for line in md_body.splitlines():
        line = line.strip()
        if line.startswith("#"):
            headings.append(re.sub(r"^#+\s*", "", line).strip())
    return headings


def split_sentences(text: str) -> list[str]:
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    parts = re.split(r"(?<=[\.\!\?])\s+(?=[A-Z0-9])", text)
    return [p.strip() for p in parts if p.strip()]


def load_documents(normalized_dir: Path) -> list[Document]:
    docs = []
    for path in sorted(normalized_dir.glob("*.md")):
        md = path.read_text(encoding="utf-8")
        meta, body = strip_frontmatter(md)
        headings = find_headings(body)
        paragraphs = [p.strip() for p in body.split("\n\n") if p.strip()]
        text_only = "\n\n".join(p for p in paragraphs if not p.startswith("#"))
        sentences = split_sentences(text_only)
        docs.append(
            Document(
                document_id=meta.get("document_id", path.stem),
                path=path,
                title=meta.get("title", headings[0] if headings else path.stem),
                source_kind=meta.get("source_kind", "unknown"),
                body=text_only,
                headings=headings,
                paragraphs=paragraphs,
                sentences=sentences,
            )
        )
    return docs


def normalize_phrase(phrase: str) -> str:
    p = phrase.lower().strip()
    p = re.sub(r"[^\w\s\-]", " ", p)
    p = re.sub(r"\s+", " ", p).strip()
    words = []
    for w in p.split():
        if len(w) > 4 and w.endswith("ies"):
            w = w[:-3] + "y"
        elif len(w) > 4 and w.endswith("ses"):
            w = w[:-2]
        elif len(w) > 4 and w.endswith("s") and not w.endswith("ss"):
            w = w[:-1]
        words.append(w)
    return " ".join(words)


def tokenise(text: str) -> list[str]:
    return re.findall(r"[A-Za-z][A-Za-z\-]{1,}", text.lower())


def extract_candidate_phrases(docs: list[Document], top_k: int = 350) -> list[dict]:
    corpus_tf = Counter()
    doc_freq = Counter()
    heading_bonus = Counter()
    contexts = defaultdict(list)
    doc_hits = defaultdict(set)

    for doc in docs:
        doc_seen = set()
        full_text = (doc.title + "\n" + "\n".join(doc.headings) + "\n" + doc.body)
        tokens = tokenise(full_text)
        for n in range(1, 5):
            for i in range(len(tokens) - n + 1):
                grams = tokens[i:i+n]
                if all(g in STOPWORDS for g in grams):
                    continue
                if grams[0] in STOPWORDS or grams[-1] in STOPWORDS:
                    continue
                phrase = " ".join(grams)
                if len(phrase) < 4 or len(phrase) > 64:
                    continue
                if phrase.count("-") > 3:
                    continue
                corpus_tf[phrase] += 1
                doc_hits[phrase].add(doc.document_id)
                if phrase not in doc_seen:
                    doc_freq[phrase] += 1
                    doc_seen.add(phrase)

        for h in [doc.title] + doc.headings:
            htoks = tokenise(h)
            for n in range(1, 5):
                for i in range(len(htoks) - n + 1):
                    grams = htoks[i:i+n]
                    if all(g in STOPWORDS for g in grams):
                        continue
                    phrase = " ".join(grams)
                    if 4 <= len(phrase) <= 64:
                        heading_bonus[phrase] += 1

        for sent in doc.sentences:
            low = sent.lower()
            for phrase in list(doc_seen)[:1000]:
                if phrase in low and len(contexts[phrase]) < 12:
                    contexts[phrase].append(sent.strip())

    scored = []
    min_df = 2 if len(docs) >= 6 else 1
    for phrase, tf in corpus_tf.items():
        df = doc_freq[phrase]
        if df < min_df:
            continue
        words = phrase.split()
        if len(words) == 1 and (len(words[0]) < 5 or words[0] in STOPWORDS):
            continue
        score = (df * 3.0) + math.log(tf + 1, 2) + (heading_bonus[phrase] * 2.0) + (len(words) * 0.15)
        scored.append({
            "candidate_phrase": phrase,
            "normalized_phrase": normalize_phrase(phrase),
            "term_frequency": tf,
            "document_frequency": df,
            "heading_bonus": heading_bonus[phrase],
            "score": round(score, 3),
            "document_ids": sorted(doc_hits[phrase]),
            "sample_contexts": contexts[phrase][:5],
        })

    scored.sort(key=lambda x: (-x["score"], -x["document_frequency"], -x["term_frequency"], x["candidate_phrase"]))
    return scored[:top_k]


def reduce_concepts(candidates: list[dict]) -> list[dict]:
    groups = defaultdict(list)

    for item in candidates:
        key = item["normalized_phrase"]
        groups[key].append(item)

    reduced = []
    consumed = set()
    keys = sorted(groups.keys(), key=lambda k: (-max(i["score"] for i in groups[k]), -len(k), k))

    for key in keys:
        if key in consumed:
            continue
        base_items = groups[key][:]
        aliases = set(i["candidate_phrase"] for i in base_items)
        docs = set()
        contexts = []
        best = max(base_items, key=lambda x: (x["score"], len(x["candidate_phrase"])))
        for bi in base_items:
            docs.update(bi["document_ids"])
            contexts.extend(bi["sample_contexts"])

        for other in keys:
            if other == key or other in consumed:
                continue
            if not other or not key:
                continue
            # merge if near-duplicate or tight containment
            same_prefix = other.startswith(key + " ") or key.startswith(other + " ")
            overlap = len(set(key.split()) & set(other.split()))
            if same_prefix or (overlap >= min(len(key.split()), len(other.split())) and abs(len(key.split()) - len(other.split())) <= 1):
                if len(set(key.split()) | set(other.split())) <= max(len(key.split()), len(other.split())) + 1:
                    consumed.add(other)
                    aliases.update(i["candidate_phrase"] for i in groups[other])
                    for oi in groups[other]:
                        docs.update(oi["document_ids"])
                        contexts.extend(oi["sample_contexts"])

        concept_id = slugify(best["normalized_phrase"], 64)
        reduced.append({
            "concept_id": concept_id,
            "preferred_label": best["candidate_phrase"],
            "normalized_label": best["normalized_phrase"],
            "alternate_labels": sorted(aliases - {best["candidate_phrase"]}),
            "document_ids": sorted(docs),
            "support_count": len(docs),
            "sample_contexts": contexts[:8],
            "seed_score": best["score"],
            "rationale_for_consolidation": "merged normalized variants, singular/plural variants, and high-overlap phrase variants",
        })
        consumed.add(key)

    reduced.sort(key=lambda x: (-x["support_count"], -x["seed_score"], x["preferred_label"]))
    return reduced


def infer_type_for_concept(concept: dict, docs_by_id: dict[str, Document]) -> tuple[str, str]:
    scores = Counter()
    contexts = concept.get("sample_contexts", [])[:]
    phrase_variants = [concept["preferred_label"]] + concept.get("alternate_labels", [])[:5]
    for doc_id in concept.get("document_ids", []):
        doc = docs_by_id.get(doc_id)
        if not doc:
            continue
        for sent in doc.sentences[:400]:
            low = sent.lower()
            if any(v.lower() in low for v in phrase_variants):
                contexts.append(sent)
                if len(contexts) >= 20:
                    break
        if len(contexts) >= 20:
            break

    joined = " ".join(contexts).lower()
    for t, cues in TYPE_CUES.items():
        for cue in cues:
            if cue in joined or cue in concept["normalized_label"]:
                scores[t] += 1

    if not scores:
        return "other", "low cue match"

    concept_type, score = scores.most_common(1)[0]
    return concept_type, f"top cue count {score}"


def build_cooccurrence(reduced: list[dict], docs: list[Document]) -> dict[str, Counter]:
    labels = {c["concept_id"]: [c["preferred_label"].lower()] + [a.lower() for a in c.get("alternate_labels", [])] for c in reduced}
    co = defaultdict(Counter)

    for doc in docs:
        for sent in doc.sentences:
            low = sent.lower()
            present = []
            for cid, variants in labels.items():
                if any(v in low for v in variants):
                    present.append(cid)
            present = list(dict.fromkeys(present))
            for i in range(len(present)):
                for j in range(i + 1, len(present)):
                    a, b = present[i], present[j]
                    co[a][b] += 1
                    co[b][a] += 1
    return co


def cluster_concepts(reduced: list[dict], co: dict[str, Counter]) -> list[dict]:
    ids = [c["concept_id"] for c in reduced]
    visited = set()
    concept_by_id = {c["concept_id"]: c for c in reduced}
    clusters = []

    for cid in ids:
        if cid in visited:
            continue
        stack = [cid]
        visited.add(cid)
        component = []
        while stack:
            node = stack.pop()
            component.append(node)
            for neigh, weight in co.get(node, {}).items():
                if weight >= 2 and neigh not in visited:
                    visited.add(neigh)
                    stack.append(neigh)

        if len(component) == 1:
            # allow singletons too
            pass

        types = Counter(concept_by_id[x].get("concept_type", "other") for x in component)
        labels = [concept_by_id[x]["preferred_label"] for x in component]
        label_seed = ", ".join(labels[:3])
        dominant_type = types.most_common(1)[0][0] if types else "other"
        cluster_id = f"cluster_{len(clusters)+1:03d}"
        clusters.append({
            "cluster_id": cluster_id,
            "cluster_label": f"{PARENT_TYPE_MAP.get(dominant_type, dominant_type)}__{slugify(label_seed, 40)}",
            "dominant_type": dominant_type,
            "concept_ids": sorted(component),
            "size": len(component),
            "label_seed": label_seed,
            "rationale": f"connected by sentence-level co-occurrence with dominant inferred type {dominant_type}",
        })

    clusters.sort(key=lambda x: (-x["size"], x["cluster_id"]))
    return clusters


def build_relationships(reduced: list[dict], co: dict[str, Counter]) -> dict[str, dict]:
    by_id = {c["concept_id"]: c for c in reduced}
    rel = {}

    for concept in reduced:
        cid = concept["concept_id"]
        label_words = set(concept["normalized_label"].split())
        related = [n for n, w in co.get(cid, {}).most_common(8) if w >= 1]
        parents = []
        children = []
        for other in reduced:
            if other["concept_id"] == cid:
                continue
            other_words = set(other["normalized_label"].split())
            if label_words and label_words < other_words:
                parents.append(other["concept_id"])
            elif other_words and other_words < label_words:
                children.append(other["concept_id"])

        rel[cid] = {
            "preferred_label": concept["preferred_label"],
            "concept_type": concept.get("concept_type", "other"),
            "parent_concepts": parents[:5],
            "child_concepts": children[:5],
            "related_concepts": related,
            "frequently_cooccurring_concepts": related[:5],
        }

    return rel


def build_concept_document_map(reduced: list[dict]) -> dict[str, Any]:
    return {
        c["concept_id"]: {
            "preferred_label": c["preferred_label"],
            "document_ids": c["document_ids"],
            "support_count": c["support_count"],
        }
        for c in reduced
    }


def make_reports(
    out_dirs: dict[str, Path],
    docs: list[Document],
    candidates: list[dict],
    reduced: list[dict],
    clusters: list[dict],
    relationships: dict[str, dict],
) -> None:
    report = [
        "# Candidate Concepts Report",
        "",
        f"Generated: {utc_now()}",
        "",
        f"Documents read: {len(docs)}",
        f"Candidate concepts: {len(candidates)}",
        "",
        "## Top candidates",
        "",
    ]
    for c in candidates[:50]:
        report.extend([
            f"### {c['candidate_phrase']}",
            f"- normalized: {c['normalized_phrase']}",
            f"- score: {c['score']}",
            f"- document_frequency: {c['document_frequency']}",
            f"- term_frequency: {c['term_frequency']}",
            "",
        ])
    (out_dirs["reports"] / "candidate_concepts_report.md").write_text("\n".join(report), encoding="utf-8")

    reduction = [
        "# Concept Reduction Report",
        "",
        f"Generated: {utc_now()}",
        "",
        f"Reduced concepts: {len(reduced)}",
        "",
    ]
    for c in reduced[:80]:
        reduction.extend([
            f"## {c['preferred_label']}",
            f"- concept_id: `{c['concept_id']}`",
            f"- normalized_label: {c['normalized_label']}",
            f"- support_count: {c['support_count']}",
            f"- alternate_labels: {', '.join(c['alternate_labels']) if c['alternate_labels'] else 'none'}",
            f"- concept_type: {c.get('concept_type','other')}",
            f"- type_rationale: {c.get('type_rationale','')}",
            "",
        ])
    (out_dirs["reports"] / "concept_reduction_report.md").write_text("\n".join(reduction), encoding="utf-8")

    clustering = [
        "# Concept Clustering Report",
        "",
        f"Generated: {utc_now()}",
        "",
        f"Clusters: {len(clusters)}",
        "",
    ]
    by_id = {c["concept_id"]: c for c in reduced}
    for cl in clusters:
        clustering.extend([
            f"## {cl['cluster_label']}",
            f"- cluster_id: `{cl['cluster_id']}`",
            f"- dominant_type: {cl['dominant_type']}",
            f"- size: {cl['size']}",
            f"- rationale: {cl['rationale']}",
            "- concepts:",
        ])
        for cid in cl["concept_ids"][:20]:
            clustering.append(f"  - {by_id[cid]['preferred_label']}")
        clustering.append("")
    (out_dirs["reports"] / "concept_clustering_report.md").write_text("\n".join(clustering), encoding="utf-8")

    ontology = [
        "# Ontology Report",
        "",
        f"Generated: {utc_now()}",
        "",
        "## Relationship overview",
        "",
    ]
    for cid, rel in list(relationships.items())[:100]:
        ontology.extend([
            f"## {rel['preferred_label']}",
            f"- concept_type: {rel['concept_type']}",
            f"- parent_concepts: {', '.join(rel['parent_concepts']) if rel['parent_concepts'] else 'none'}",
            f"- child_concepts: {', '.join(rel['child_concepts']) if rel['child_concepts'] else 'none'}",
            f"- related_concepts: {', '.join(rel['related_concepts']) if rel['related_concepts'] else 'none'}",
            "",
        ])
    (out_dirs["reports"] / "ontology_report.md").write_text("\n".join(ontology), encoding="utf-8")

    index_report = [
        "# Concept Index Report",
        "",
        f"Generated: {utc_now()}",
        "",
        f"Total reduced concepts: {len(reduced)}",
        f"Total clusters: {len(clusters)}",
        "",
        "## Dominant concept types",
        "",
    ]
    type_counts = Counter(c.get("concept_type","other") for c in reduced)
    for t, n in type_counts.most_common():
        index_report.append(f"- {t}: {n}")
    index_report.extend(["", "## Largest clusters", ""])
    for cl in clusters[:15]:
        index_report.append(f"- {cl['cluster_label']} ({cl['size']})")
    (out_dirs["reports"] / "concept_index_report.md").write_text("\n".join(index_report), encoding="utf-8")


def ensure_dirs(output_dir: Path) -> dict[str, Path]:
    dirs = {
        "root": output_dir,
        "metadata": output_dir / "metadata",
        "reports": output_dir / "reports",
    }
    for p in dirs.values():
        p.mkdir(parents=True, exist_ok=True)
    return dirs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build corpus-derived ontology from Phase 1 normalized text")
    parser.add_argument("--phase1", required=True, help="Path to prepared_corpus_phase1")
    parser.add_argument("--output", required=True, help="Path to Phase 2 output directory")
    parser.add_argument("--top-k", type=int, default=350, help="Maximum candidate concepts to keep before reduction")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    phase1_dir = Path(args.phase1).expanduser().resolve()
    output_dir = Path(args.output).expanduser().resolve()

    normalized_dir = phase1_dir / "normalized_text"
    if not normalized_dir.exists():
        raise SystemExit(f"Missing normalized_text folder: {normalized_dir}")

    out_dirs = ensure_dirs(output_dir)
    docs = load_documents(normalized_dir)
    if not docs:
        raise SystemExit("No normalized .md files found in Phase 1 output.")

    docs_by_id = {d.document_id: d for d in docs}

    candidates = extract_candidate_phrases(docs, top_k=args.top_k)
    reduced = reduce_concepts(candidates)

    for concept in reduced:
        ctype, rationale = infer_type_for_concept(concept, docs_by_id)
        concept["concept_type"] = ctype
        concept["type_rationale"] = rationale

    co = build_cooccurrence(reduced, docs)
    clusters = cluster_concepts(reduced, co)

    cluster_lookup = {}
    for cl in clusters:
        for cid in cl["concept_ids"]:
            cluster_lookup[cid] = cl["cluster_id"]

    relationships = build_relationships(reduced, co)
    concept_document_map = build_concept_document_map(reduced)

    concept_index = []
    for concept in reduced:
        cid = concept["concept_id"]
        rel = relationships.get(cid, {})
        concept_index.append({
            "concept_id": cid,
            "preferred_label": concept["preferred_label"],
            "alternate_labels": concept["alternate_labels"],
            "definition_inferred_from_corpus": concept["sample_contexts"][0] if concept["sample_contexts"] else "",
            "concept_type": concept["concept_type"],
            "parent_concepts": rel.get("parent_concepts", []),
            "child_concepts": rel.get("child_concepts", []),
            "related_concepts": rel.get("related_concepts", []),
            "representative_phrases": concept["sample_contexts"][:3],
            "frequently_cooccurring_concepts": rel.get("frequently_cooccurring_concepts", []),
            "source_documents": concept["document_ids"],
            "cluster_id": cluster_lookup.get(cid),
            "support_count": concept["support_count"],
            "confidence": "medium" if concept["support_count"] >= 2 else "low",
            "ambiguity_notes": "",
        })

    concept_index_rows = []
    for row in concept_index:
        concept_index_rows.append({
            "concept_id": row["concept_id"],
            "preferred_label": row["preferred_label"],
            "concept_type": row["concept_type"],
            "cluster_id": row["cluster_id"],
            "support_count": row["support_count"],
            "alternate_labels": " | ".join(row["alternate_labels"]),
            "source_documents": " | ".join(row["source_documents"]),
            "parent_concepts": " | ".join(row["parent_concepts"]),
            "child_concepts": " | ".join(row["child_concepts"]),
            "related_concepts": " | ".join(row["related_concepts"]),
            "definition_inferred_from_corpus": row["definition_inferred_from_corpus"],
        })

    # Save raw outputs
    safe_json_dump(candidates, out_dirs["metadata"] / "candidate_concepts.json")
    safe_json_dump(reduced, out_dirs["metadata"] / "reduced_concepts.json")
    safe_json_dump(clusters, out_dirs["metadata"] / "concept_clusters.json")
    safe_json_dump(relationships, out_dirs["metadata"] / "concept_relationships.json")
    safe_json_dump(concept_index, out_dirs["metadata"] / "concept_index.json")
    safe_json_dump(concept_document_map, out_dirs["metadata"] / "concept_document_map.json")
    write_csv(concept_index_rows, out_dirs["metadata"] / "concept_index.csv")

    make_reports(out_dirs, docs, candidates, reduced, clusters, relationships)

    summary = [
        "# Phase 2 Summary",
        "",
        f"Generated: {utc_now()}",
        "",
        f"- documents_read: {len(docs)}",
        f"- candidate_concepts: {len(candidates)}",
        f"- reduced_concepts: {len(reduced)}",
        f"- concept_clusters: {len(clusters)}",
        "",
        "## Recommended next step",
        "",
        "Use these outputs to refine concept naming, validate coverage, and then build concept-aware segmentation and upload strategy.",
        "",
    ]
    (out_dirs["reports"] / "phase2_summary.md").write_text("\n".join(summary), encoding="utf-8")

    print(f"Read {len(docs)} normalized files from: {normalized_dir}")
    print(f"Built {len(candidates)} candidate concepts, {len(reduced)} reduced concepts, and {len(clusters)} clusters")
    print(f"Output written to: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
