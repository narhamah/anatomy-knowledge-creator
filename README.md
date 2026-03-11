# Phase 2 Ontology Builder

This phase reads the normalized Markdown corpus created by Phase 1 and builds a **corpus-derived scientific ontology layer**.

## Inputs

Point it to your existing Phase 1 output folder:

- `prepared_corpus_phase1/normalized_text/*.md`

## Outputs

It creates:

- `metadata/candidate_concepts.json`
- `metadata/reduced_concepts.json`
- `metadata/concept_clusters.json`
- `metadata/concept_relationships.json`
- `metadata/concept_index.json`
- `metadata/concept_index.csv`
- `metadata/concept_document_map.json`

And reports:

- `reports/candidate_concepts_report.md`
- `reports/concept_reduction_report.md`
- `reports/concept_clustering_report.md`
- `reports/ontology_report.md`
- `reports/concept_index_report.md`
- `reports/phase2_summary.md`

## What this phase does

1. Loads normalized text files from Phase 1
2. Extracts candidate concepts from the corpus
3. Reduces near-duplicates and aliases
4. Infers concept types
5. Builds co-occurrence-based clusters
6. Builds a concept relationship map
7. Creates a concept index and concept→document map

## Install

No extra packages are required beyond the Python standard library.

## Run

```bash
python phase2_ontology_builder.py --phase1 "./prepared_corpus_phase1" --output "./prepared_corpus_phase2"
```

Optional:

```bash
python phase2_ontology_builder.py --phase1 "./prepared_corpus_phase1" --output "./prepared_corpus_phase2" --top-k 500
```

## Notes

This version is intentionally local and deterministic.

It is a strong **Phase 2 foundation**, but it is still heuristic.  
The next upgrade can add an **OpenAI-assisted refinement pass** that:

- improves concept naming
- resolves ambiguous aliases
- strengthens ontology parent/child relationships
- improves concept cluster labels
- generates a concept coverage validation set
