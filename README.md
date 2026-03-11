# Anatomy Knowledge Creator

A 5-phase pipeline that extracts text from PDFs/DOCX, builds a scientific ontology, refines it with OpenAI, and uploads to OpenAI vector stores for retrieval.

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Configure your OpenAI API key
cp .env.example .env
# Edit .env and fill in your OPENAI_API_KEY

# 3. Run the full pipeline
python run_pipeline.py --source ./pdfs --workspace ./output --project anatomy
```

## Configuration

Set your OpenAI API key in `.env`:

```
OPENAI_API_KEY=sk-...
```

Alternatively, place an `OPENAIKEY.txt` file in the working directory (backward compatible).

## Pipeline Phases

### Phase 1: Corpus Builder

Scans a folder for PDFs and DOCX files, extracts text, normalizes it, and produces a Markdown corpus with quality reports.

```bash
python phase1_corpus_builder.py --source ./pdfs --output ./output/phase1
```

### Phase 2: Ontology Builder

Reads the Phase 1 corpus and builds a heuristic ontology: candidate concepts, reduced concepts, clusters, relationships, and a concept index.

```bash
python phase2_ontology_builder.py --phase1 ./output/phase1 --output ./output/phase2
```

### Phase 2.5: OpenAI Ontology Refiner

Uses the OpenAI API to refine the heuristic ontology: merges variants, demotes bridge terms, builds clean domains.

```bash
python phase2_5_openai_refiner.py --phase2 ./output/phase2 --output ./output/phase2_5
```

### Phase 3: Concept Hub Builder

Combines all previous outputs into vector-store-ready concept hub documents with upload manifests.

```bash
python phase3_concept_hub_builder.py \
    --phase1 ./output/phase1 \
    --phase2 ./output/phase2 \
    --phase2_5 ./output/phase2_5 \
    --output ./output/phase3
```

### Phase 4: OpenAI Vector Store Uploader

Creates vector stores in OpenAI and uploads concept hub + source files with metadata attributes.

```bash
python phase4_openai_vectorstore_uploader.py \
    --phase3 ./output/phase3 \
    --output ./output/phase4 \
    --project anatomy
```

## Unified Runner

Run all phases with a single command:

```bash
python run_pipeline.py --source ./pdfs --workspace ./output --project anatomy
```

Options:
- `--skip-upload` — stop after Phase 3
- `--start-from 3` — resume from a specific phase (1, 2, 2.5, 3, or 4)
- `--model gpt-5-mini` — choose the OpenAI model for Phase 2.5

## Project Structure

```
.env.example            # Template for API key configuration
utils.py                # Shared utility functions (DRY)
openai_client.py        # Shared OpenAI client setup
run_pipeline.py         # Unified pipeline runner
phase1_corpus_builder.py
phase2_ontology_builder.py
phase2_5_openai_refiner.py
phase3_concept_hub_builder.py
phase4_openai_vectorstore_uploader.py
```

## Requirements

- Python 3.10+
- `openai` — OpenAI Python SDK
- `python-dotenv` — .env file loading
- `pymupdf` or `pypdf` — PDF text extraction
- `python-docx` — Word document extraction
