#!/usr/bin/env python3
"""
Unified pipeline runner for the Anatomy Knowledge Creator.

Runs all phases sequentially:
    Phase 1  -> PDF/DOCX text extraction
    Phase 2  -> Heuristic ontology building
    Phase 2.5 -> OpenAI ontology refinement
    Phase 3  -> Concept hub document builder
    Phase 4  -> OpenAI vector store upload

Usage:
    python run_pipeline.py --source ./pdfs --workspace ./output --project anatomy

Skip the upload step:
    python run_pipeline.py --source ./pdfs --workspace ./output --project anatomy --skip-upload

Resume from a specific phase:
    python run_pipeline.py --source ./pdfs --workspace ./output --project anatomy --start-from 3
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PHASES = {
    1: "phase1_corpus_builder",
    2: "phase2_ontology_builder",
    2.5: "phase2_5_openai_refiner",
    3: "phase3_concept_hub_builder",
    4: "phase4_openai_vectorstore_uploader",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the full corpus-creation pipeline end-to-end",
    )
    parser.add_argument("--source", required=True, help="Path to source PDFs/DOCX files")
    parser.add_argument("--workspace", required=True, help="Base output directory for all phases")
    parser.add_argument("--project", required=True, help="Project name for vector store naming")
    parser.add_argument("--model", default="gpt-5-mini", help="OpenAI model for Phase 2.5 (default: gpt-5-mini)")
    parser.add_argument("--skip-upload", action="store_true", help="Stop after Phase 3 (skip vector store upload)")
    parser.add_argument(
        "--start-from", type=float, default=1,
        choices=[1, 2, 2.5, 3, 4],
        help="Resume from a specific phase (default: 1)",
    )
    return parser.parse_args()


def run_phase(phase_num: float, label: str, main_fn, argv: list[str]) -> None:
    phase_label = str(phase_num) if phase_num != 2.5 else "2.5"
    print(f"\n{'='*60}")
    print(f"  Phase {phase_label}: {label}")
    print(f"{'='*60}\n")

    rc = main_fn(argv)
    if rc and rc != 0:
        print(f"\nPhase {phase_label} failed with exit code {rc}.", file=sys.stderr)
        raise SystemExit(rc)


def main() -> int:
    args = parse_args()
    workspace = Path(args.workspace).expanduser().resolve()

    phase1_out = str(workspace / "phase1")
    phase2_out = str(workspace / "phase2")
    phase25_out = str(workspace / "phase2_5")
    phase3_out = str(workspace / "phase3")
    phase4_out = str(workspace / "phase4")

    # Phase 1
    if args.start_from <= 1:
        from phase1_corpus_builder import main as phase1_main
        run_phase(1, "Corpus Builder", phase1_main, [
            "--source", args.source,
            "--output", phase1_out,
        ])

    # Phase 2
    if args.start_from <= 2:
        from phase2_ontology_builder import main as phase2_main
        run_phase(2, "Ontology Builder", phase2_main, [
            "--phase1", phase1_out,
            "--output", phase2_out,
        ])

    # Phase 2.5
    if args.start_from <= 2.5:
        from phase2_5_openai_refiner import main as phase25_main
        run_phase(2.5, "OpenAI Ontology Refiner", phase25_main, [
            "--phase2", phase2_out,
            "--output", phase25_out,
            "--model", args.model,
        ])

    # Phase 3
    if args.start_from <= 3:
        from phase3_concept_hub_builder import main as phase3_main
        run_phase(3, "Concept Hub Builder", phase3_main, [
            "--phase1", phase1_out,
            "--phase2", phase2_out,
            "--phase2_5", phase25_out,
            "--output", phase3_out,
        ])

    # Phase 4
    if args.start_from <= 4 and not args.skip_upload:
        from phase4_openai_vectorstore_uploader import main as phase4_main
        run_phase(4, "OpenAI Vector Store Uploader", phase4_main, [
            "--phase3", phase3_out,
            "--output", phase4_out,
            "--project", args.project,
        ])

    print(f"\nPipeline complete. All outputs in: {workspace}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
