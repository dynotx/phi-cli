import argparse

from phi._version import __version__
from phi.config import _FILTER_PRESETS, POLL_INTERVAL

_CLI_EPILOG = """\
Fetch and prepare target structures:
  phi fetch --pdb 4ZQK --chain A --residues 56-290 --out target.pdb
  phi fetch --uniprot Q9NZQ7 --trim-low-confidence 70 --upload

Design (backbone generation):
  phi design      --target-pdb target.pdb --hotspots A45,A67 --num-designs 50
  phi design      --length 80 --num-designs 20
  phi boltzgen    --yaml design.yaml --protocol protein-anything --num-designs 10

Validation (fold + score):
  phi esmfold     --fasta sequences.fasta
  phi alphafold   --fasta complex.fasta
  phi proteinmpnn --pdb design.pdb  --num-sequences 20
  phi esm2        --fasta sequences.fasta
  phi boltz       --fasta complex.fasta

Batch filter pipeline (100-50,000 designs):
  phi upload --dir ./designs/ --file-type pdb
  phi filter --dataset-id <id> --preset default --wait
  phi download   --out ./results

Dataset management:
  phi datasets               # list your datasets
  phi dataset DATASET_ID     # show dataset details

Authentication:
  phi login                  # verify API key + print connection and identity

Research:
  phi research --question "What are known PD-L1 binding hotspots?"
"""


def _add_fasta_args(p: argparse.ArgumentParser) -> None:
    g = p.add_mutually_exclusive_group()
    g.add_argument("--fasta", metavar="FILE", help="FASTA file to submit")
    g.add_argument("--fasta-str", metavar="FASTA", help="FASTA content as a string (for scripting)")
    g.add_argument(
        "--dataset-id",
        metavar="DATASET_ID",
        help="Pre-ingested dataset ID (for batch runs of 100–50,000 files)",
    )


def _add_pdb_args(p: argparse.ArgumentParser) -> None:
    g = p.add_mutually_exclusive_group()
    g.add_argument("--pdb", metavar="FILE", help="PDB structure file")
    g.add_argument("--pdb-gcs", metavar="URI", help="Cloud storage URI to PDB (gs://…)")
    g.add_argument(
        "--dataset-id",
        metavar="DATASET_ID",
        help="Pre-ingested dataset ID (for batch runs of 100–50,000 files)",
    )


def _add_job_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--run-id", metavar="ID", help="Optional run label")
    p.add_argument(
        "--wait", action="store_true", default=True, help="Poll until job completes (default: on)"
    )
    p.add_argument(
        "--no-wait", action="store_false", dest="wait", help="Return immediately after submission"
    )
    p.add_argument("--out", metavar="DIR", help="Download results to DIR when done")
    p.add_argument("--json", action="store_true", help="Output raw JSON")


def _add_fetch_args(p: argparse.ArgumentParser) -> None:
    src = p.add_argument_group("source (pick one)")
    src.add_argument("--pdb", metavar="ID", help="RCSB PDB ID (e.g., 4ZQK)")
    src.add_argument(
        "--uniprot",
        metavar="ID",
        help="UniProt accession — downloads from AlphaFold DB (e.g., Q9NZQ7)",
    )

    crop = p.add_argument_group("cropping (optional)")
    crop.add_argument("--chain", metavar="CHAIN", help="Extract a single chain (e.g., A)")
    crop.add_argument(
        "--residues",
        metavar="START-END",
        help="Keep only residues in this range (e.g., 56-290)",
    )
    crop.add_argument(
        "--trim-low-confidence",
        type=float,
        metavar="PLDDT",
        dest="trim_low_confidence",
        help=(
            "Remove residues with pLDDT below this threshold. "
            "AlphaFold DB structures store pLDDT in the B-factor column. "
            "Typical value: 70."
        ),
    )

    out = p.add_argument_group("output")
    out.add_argument(
        "--out",
        metavar="FILE",
        help="Output PDB file path (default: {ID}[_{chain}].pdb in current directory)",
    )
    out.add_argument(
        "--upload",
        action="store_true",
        help=(
            "Upload to Dyno cloud storage after saving — creates a dataset and "
            "prints the GCS URI for use with 'phi design --target-pdb-gcs'"
        ),
    )
    out.add_argument(
        "--name",
        metavar="NAME",
        help="Dataset name label when using --upload (default: PDB/UniProt ID)",
    )


def _add_rfdiffusion3_args(p: argparse.ArgumentParser) -> None:
    mode = p.add_argument_group("design mode (pick one)")
    mode.add_argument("--length", type=int, metavar="N", help="Backbone length for de novo generation")
    mode.add_argument("--target-pdb", metavar="FILE", help="Target PDB for binder design")
    mode.add_argument(
        "--target-pdb-gcs",
        metavar="URI",
        help="Cloud storage URI to target PDB (gs://…)",
    )
    mode.add_argument("--motif-pdb", metavar="FILE", help="Motif PDB for scaffolding")
    mode.add_argument(
        "--motif-pdb-gcs",
        metavar="URI",
        help="Cloud storage URI to motif PDB (gs://…)",
    )

    binder = p.add_argument_group("binder design options")
    binder.add_argument("--target-chain", metavar="CHAIN", help="Target chain ID (e.g., A)")
    binder.add_argument(
        "--hotspots",
        metavar="A45,A67",
        help="Comma-separated hotspot residues for interface design (e.g., A45,A67,A89)",
    )
    binder.add_argument(
        "--motif-residues",
        metavar="10-20,45-55",
        help="Comma-separated motif residue ranges (e.g., 10-20,45-55)",
    )

    gen = p.add_argument_group("generation parameters")
    gen.add_argument("--num-designs", type=int, default=10, metavar="N", help="Designs to generate (default: 10)")
    gen.add_argument(
        "--steps",
        type=int,
        default=50,
        metavar="N",
        help="Diffusion inference steps — higher improves quality (default: 50)",
    )
    gen.add_argument("--contigs", metavar="STR", help="Contig specification string for advanced control")
    gen.add_argument("--symmetry", metavar="C3", help="Symmetry specification (e.g., C3, D2, C5)")


def _add_boltzgen_args(p: argparse.ArgumentParser) -> None:
    inp = p.add_argument_group("input (pick one)")
    inp.add_argument("--yaml", metavar="FILE", help="Local YAML design specification file")
    inp.add_argument("--yaml-gcs", metavar="URI", help="Cloud storage URI to YAML file (gs://…)")
    inp.add_argument(
        "--structure-gcs",
        metavar="URI",
        help="Cloud storage URI to structure file referenced in the YAML (gs://…)",
    )

    gen = p.add_argument_group("generation parameters")
    gen.add_argument(
        "--protocol",
        default="protein-anything",
        choices=[
            "protein-anything",
            "peptide-anything",
            "protein-small_molecule",
            "antibody-anything",
            "nanobody-anything",
            "protein-redesign",
        ],
        metavar="PROTOCOL",
        help=(
            "Design protocol: protein-anything (default), peptide-anything, "
            "protein-small_molecule, antibody-anything, nanobody-anything, protein-redesign"
        ),
    )
    gen.add_argument(
        "--num-designs",
        type=int,
        default=10,
        metavar="N",
        help="Intermediate designs to generate (default: 10; use 10,000–60,000 for production)",
    )
    gen.add_argument(
        "--budget",
        type=int,
        metavar="N",
        help="Final diversity-optimized design count (default: num_designs // 10)",
    )
    gen.add_argument(
        "--boltzgen-steps",
        metavar="STEPS",
        dest="steps",
        help="Specific pipeline steps, space-separated (e.g., 'design inverse_folding folding'). Omit to run full pipeline.",
    )

    inv = p.add_argument_group("inverse folding only")
    inv.add_argument(
        "--only-inverse-fold",
        action="store_true",
        dest="only_inverse_fold",
        help="Run inverse folding on an existing structure YAML — skips backbone design",
    )
    inv.add_argument(
        "--inverse-fold-num-sequences",
        type=int,
        metavar="N",
        dest="inverse_fold_num_sequences",
        help="Sequences per design when using --only-inverse-fold (default: 2)",
    )


def build_parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(
        prog="phi",
        description="Dyno Phi protein design platform CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=_CLI_EPILOG,
    )
    root.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )
    root.add_argument(
        "--poll-interval",
        type=float,
        default=None,
        metavar="S",
        help=f"Seconds between status-poll requests (default: {POLL_INTERVAL})",
    )
    sub = root.add_subparsers(dest="command", required=True)

    p = sub.add_parser("login", help="Verify API key and print connection + identity details")
    p.add_argument("--json", action="store_true")

    p = sub.add_parser(
        "upload",
        help="Upload files → ingest → dataset (batch workflow entry point)",
    )
    p.add_argument(
        "files",
        nargs="*",
        metavar="FILE",
        help="Files to upload (positional). Use --dir for directories.",
    )
    p.add_argument("--dir", metavar="DIR", help="Upload all matching files in this directory")
    p.add_argument(
        "--file-type",
        metavar="TYPE",
        help="Override auto-detected file type (pdb, cif, fasta, csv). "
        "When omitted, type is inferred from file extensions.",
    )
    p.add_argument(
        "--gcs",
        metavar="URI",
        help="[Future] Import from external cloud storage (gs://bucket/prefix/)",
    )
    p.add_argument("--run-id", metavar="ID", help="Label for this ingest session")
    p.add_argument(
        "--wait",
        action="store_true",
        default=True,
        help="Poll until dataset is READY (default: on)",
    )
    p.add_argument(
        "--no-wait",
        action="store_false",
        dest="wait",
        help="Return after finalizing without polling",
    )

    p = sub.add_parser("datasets", help="List your datasets")
    p.add_argument("--limit", type=int, default=20, metavar="N")
    p.add_argument("--json", action="store_true")

    p = sub.add_parser("dataset", help="Show details for a dataset")
    p.add_argument("dataset_id")
    p.add_argument("--json", action="store_true")

    p = sub.add_parser("ingest-session", help="Show status of an ingest session")
    p.add_argument("session_id", metavar="SESSION_ID")
    p.add_argument("--json", action="store_true")

    p = sub.add_parser("esmfold", aliases=["folding"], help="Fast structure prediction (~1 min)")
    _add_fasta_args(p)
    p.add_argument(
        "--recycles", type=int, default=3, metavar="N", help="Recycling iterations (default: 3)"
    )
    p.add_argument("--no-confidence", action="store_true", help="Skip per-residue pLDDT extraction")
    p.add_argument(
        "--fasta-name",
        metavar="NAME",
        help="Name label for output files (single-sequence mode only)",
    )
    _add_job_args(p)

    p = sub.add_parser(
        "alphafold",
        aliases=["complex_folding"],
        help="Structure prediction — monomer or multimer (8–15 min)",
    )
    _add_fasta_args(p)
    p.add_argument(
        "--models", default="1,2,3", metavar="1,2,3", help="Model numbers to run (default: 1,2,3)"
    )
    p.add_argument(
        "--model-type",
        default="auto",
        choices=["auto", "ptm", "multimer_v1", "multimer_v2", "multimer_v3"],
        help="Model type — auto picks ptm for monomers, multimer_v3 for complexes (default: auto)",
    )
    p.add_argument(
        "--msa-tool",
        default="mmseqs2",
        choices=["mmseqs2", "jackhmmer"],
        help="MSA algorithm (default: mmseqs2)",
    )
    p.add_argument(
        "--msa-databases",
        default="uniref_env",
        choices=["uniref_env", "uniref_only"],
        help="Database set searched for MSA (default: uniref_env)",
    )
    p.add_argument(
        "--template-mode",
        default="none",
        choices=["none", "pdb70"],
        help="Template structure lookup mode (default: none)",
    )
    p.add_argument(
        "--pair-mode",
        default="unpaired_paired",
        choices=["unpaired_paired", "paired", "unpaired"],
        help="MSA pairing strategy for complexes (default: unpaired+paired)",
    )
    p.add_argument(
        "--recycles", type=int, default=6, metavar="N", help="Recycling iterations (default: 6)"
    )
    p.add_argument(
        "--num-seeds", type=int, default=3, metavar="N", help="Number of model seeds (default: 3)"
    )
    p.add_argument(
        "--amber",
        action="store_true",
        help="Run AMBER force-field relaxation (removes stereochemical violations)",
    )
    p.add_argument(
        "--relax",
        type=int,
        default=0,
        metavar="N",
        help="Amber relaxation passes as int (legacy; prefer --amber)",
    )
    _add_job_args(p)

    p = sub.add_parser(
        "proteinmpnn",
        aliases=["inverse_folding"],
        help="Sequence design via inverse folding (1–2 min)",
    )
    _add_pdb_args(p)
    p.add_argument(
        "--num-sequences",
        type=int,
        default=10,
        metavar="N",
        help="Sequences to design (default: 10)",
    )
    p.add_argument(
        "--temperature",
        type=float,
        default=0.1,
        metavar="T",
        help="Sampling temperature 0–1 (default: 0.1)",
    )
    p.add_argument("--fixed", metavar="A52,A56", help="Fixed residue positions e.g. A52,A56,A63")
    _add_job_args(p)

    p = sub.add_parser("esm2", help="Language model scoring: log-likelihood and perplexity")
    _add_fasta_args(p)
    p.add_argument(
        "--mask", metavar="5,10,15", help="Comma-separated positions to mask for scoring"
    )
    _add_job_args(p)

    p = sub.add_parser("boltz", help="Biomolecular complex prediction — proteins, DNA, RNA")
    _add_fasta_args(p)
    p.add_argument("--recycles", type=int, default=3, metavar="N")
    p.add_argument("--no-msa", action="store_true", help="Disable MSA (faster, lower accuracy)")
    _add_job_args(p)

    p = sub.add_parser(
        "rfdiffusion3",
        aliases=["design"],
        help="All-atom backbone generation: binder design, de novo, motif scaffolding (2–5 min/design)",
    )
    _add_rfdiffusion3_args(p)
    _add_job_args(p)

    p = sub.add_parser(
        "boltzgen",
        help="All-atom generative design from a YAML spec — proteins, peptides, small molecules (10–20 min)",
    )
    _add_boltzgen_args(p)
    _add_job_args(p)

    p = sub.add_parser(
        "fetch",
        help="Download a structure from RCSB PDB or AlphaFold DB, optionally crop, save locally",
    )
    _add_fetch_args(p)

    p = sub.add_parser("research", help="Biological research query with citations (2–5 min)")
    p.add_argument(
        "--question",
        required=True,
        metavar="QUESTION",
        help="Research question, e.g. 'What are known binding hotspots for PD-L1?'",
    )
    p.add_argument(
        "--target",
        metavar="TARGET",
        help="Protein or gene name to focus the search (e.g. PD-L1, KRAS, EGFR)",
    )
    p.add_argument(
        "--databases",
        default="pubmed,uniprot,pdb",
        metavar="pubmed,uniprot,pdb",
        help="Comma-separated databases to query (default: pubmed,uniprot,pdb)",
    )
    p.add_argument(
        "--max-papers",
        type=int,
        default=20,
        metavar="N",
        help="Maximum papers to retrieve from PubMed (default: 20)",
    )
    p.add_argument(
        "--structures", action="store_true", help="Include related PDB structures in report"
    )
    p.add_argument("--context", metavar="TEXT", help="Additional context for the research query")
    p.add_argument(
        "--context-file",
        metavar="FILE",
        help="Path to a prior research.md file — prepended as context for this query",
    )
    p.add_argument(
        "--dataset-id",
        metavar="ID",
        help="Associate notes with a dataset and sync to cloud storage",
    )
    p.add_argument(
        "--notes-file",
        metavar="FILE",
        default="./research.md",
        help="Local append-only notes file (default: ./research.md)",
    )
    p.add_argument(
        "--no-save", action="store_true", help="Skip saving the report to the local notes file"
    )
    p.add_argument(
        "--stream",
        action="store_true",
        help="Stream results live from Modal SSE endpoint (skips job tracking)",
    )
    p.add_argument(
        "--stream-url",
        metavar="URL",
        default="https://dynotx--research-agent-streaming-fastapi-app.modal.run",
        help="Override the SSE base URL",
    )
    _add_job_args(p)

    p = sub.add_parser("notes", help="View accumulated research notes for a dataset")
    p.add_argument(
        "dataset_id", metavar="DATASET_ID", help="Dataset ID to retrieve research notes for"
    )
    p.add_argument(
        "--out",
        metavar="PATH",
        help="Save notes to PATH (.md file) or PATH/research.md (directory) instead of printing",
    )
    p.add_argument("--json", action="store_true", help="Output raw JSON")

    p = sub.add_parser("status", help="Get job status")
    p.add_argument("job_id")
    p.add_argument("--json", action="store_true")

    p = sub.add_parser("jobs", help="List recent jobs")
    p.add_argument("--limit", type=int, default=20, metavar="N")
    p.add_argument(
        "--status",
        metavar="STATUS",
        choices=["pending", "running", "completed", "failed", "cancelled"],
    )
    p.add_argument("--job-type", metavar="TYPE")
    p.add_argument("--json", action="store_true")

    p = sub.add_parser("logs", help="Print log stream URL for a job")
    p.add_argument("job_id")
    p.add_argument("--follow", action="store_true")

    p = sub.add_parser("cancel", help="Cancel a running job")
    p.add_argument("job_id")

    p = sub.add_parser(
        "use",
        help="Set the active dataset ID (cached in .phi-state.json, used as default for filter etc.)",
    )
    p.add_argument("dataset_id", metavar="DATASET_ID")

    p = sub.add_parser("download", help="Download output files for a completed job")
    p.add_argument("job_id", nargs="?", default=None, help="Job ID (default: last cached job)")
    p.add_argument("--out", default="./results", metavar="DIR")
    p.add_argument(
        "--all",
        action="store_true",
        help="Download all artifact types including MSA files, zip archives, and scripts",
    )

    p = sub.add_parser(
        "filter",
        help="Filter and score binder designs (inverse folding → folding → complex folding → score)",
    )
    p.add_argument(
        "--dataset-id",
        default=None,
        metavar="ID",
        help="Dataset of PDB/CIF designs (default: cached from last upload or `phi use`)",
    )
    p.add_argument(
        "--preset",
        choices=list(_FILTER_PRESETS),
        metavar="NAME",
        help=f"Named filter preset: {', '.join(_FILTER_PRESETS)} "
        f"(individual flags override preset values)",
    )
    p.add_argument(
        "--num-sequences",
        type=int,
        default=4,
        metavar="N",
        help="ProteinMPNN sequences per design (default: 4)",
    )
    p.add_argument(
        "--plddt-threshold",
        type=float,
        default=None,
        metavar="F",
        help="ESMFold binder pLDDT lower-bound cutoff (default preset: 0.80)",
    )
    p.add_argument(
        "--iptm-threshold",
        type=float,
        default=None,
        metavar="F",
        help="AlphaFold2 interface pTM lower-bound cutoff (default preset: 0.50)",
    )
    p.add_argument(
        "--ipae-threshold",
        type=float,
        default=None,
        metavar="F",
        help="AlphaFold2 interface PAE upper-bound in Å (default preset: 10.85 Å = BindCraft 0.35 × 31)",
    )
    p.add_argument(
        "--ptm-threshold",
        type=float,
        default=None,
        metavar="F",
        help="AlphaFold2 complex pTM lower-bound cutoff (default preset: 0.55)",
    )
    p.add_argument(
        "--rmsd-threshold",
        type=float,
        default=None,
        metavar="F",
        help="Binder backbone RMSD upper-bound cutoff in Å (default preset: 3.5)",
    )
    p.add_argument(
        "--num-recycles",
        type=int,
        default=3,
        metavar="N",
        help="AlphaFold2 recycle iterations (default: 3)",
    )
    p.add_argument("--run-id", metavar="ID", help="Optional custom run ID")
    p.add_argument("--wait", action="store_true", help="Poll until pipeline completes")
    p.add_argument("--out", metavar="DIR", help="Download results on completion")
    p.add_argument(
        "--all",
        action="store_true",
        help="When --out is set, download all artifact types including MSA files and archives",
    )
    p.add_argument(
        "--msa-tool",
        default="single_sequence",
        choices=["single_sequence", "mmseqs2", "jackhmmer"],
        metavar="TOOL",
        help=(
            "MSA algorithm for AF2 complex prediction: single_sequence (default — skips MSA, "
            "best-calibrated for novel designed binders with no natural homologs), "
            "mmseqs2, or jackhmmer. "
            "Matches the --msa-tool option on 'phi alphafold'."
        ),
    )

    p = sub.add_parser("scores", help="Display scoring metrics table for a completed job")
    p.add_argument("job_id", nargs="?", default=None, help="Job ID (default: last cached job)")
    p.add_argument(
        "--top",
        type=int,
        default=20,
        metavar="N",
        help="Show top-N candidates (default: 20)",
    )
    p.add_argument("--out", metavar="FILE", help="Save scores CSV to file")
    p.add_argument("--json", action="store_true", help="Output raw JSON")

    return root
