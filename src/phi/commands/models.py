import argparse
from pathlib import Path

from phi.api import _require_key, _submit
from phi.display import _die, _print_status, _print_submission, console
from phi.download import _download_job, _read_fasta
from phi.polling import _poll


def _run_model_job(job_type: str, params: dict, args: argparse.Namespace) -> None:
    from phi.config import POLL_INTERVAL as _INTERVAL

    dataset_id = getattr(args, "dataset_id", None)
    result = _submit(job_type, params, run_id=args.run_id, dataset_id=dataset_id)
    job_id = _require_key(result, "job_id", f"POST /jobs ({job_type})")
    _print_submission(result)
    if args.wait:
        console.print(f"\n[dim]Polling every {_INTERVAL}s …[/]")
        final = _poll(job_id)
        _print_status(final)
        if args.out and final.get("status") == "completed":
            _download_job(final, args.out)


def cmd_esmfold(args: argparse.Namespace) -> None:
    params: dict = {
        "num_recycles": args.recycles,
        "extract_confidence": not args.no_confidence,
    }
    if not getattr(args, "dataset_id", None):
        params["fasta_str"] = _read_fasta(args)
        if args.fasta_name:
            params["fasta_name"] = args.fasta_name
    _run_model_job("esmfold", params, args)


def cmd_alphafold(args: argparse.Namespace) -> None:
    models = [int(m) for m in args.models.split(",") if m.strip()]
    params: dict = {
        "models": models,
        "model_type": args.model_type,
        "msa_tool": args.msa_tool,
        "msa_databases": args.msa_databases,
        "template_mode": args.template_mode,
        "pair_mode": args.pair_mode,
        "num_recycles": args.recycles,
        "num_seeds": args.num_seeds,
        "num_relax": 1 if args.amber else args.relax,
    }
    if not getattr(args, "dataset_id", None):
        fasta_str = _read_fasta(args)
        params["fasta_str"] = fasta_str
        if ":" in fasta_str:
            console.print(f"  [dim]multimer mode ({fasta_str.count(':') + 1} chains)[/]")
    _run_model_job("alphafold", params, args)


def cmd_proteinmpnn(args: argparse.Namespace) -> None:
    params: dict = {
        "num_sequences": args.num_sequences,
        "temperature": args.temperature,
    }
    if not getattr(args, "dataset_id", None):
        if args.pdb:
            params["pdb_content"] = Path(args.pdb).read_text()
        elif args.pdb_gcs:
            params["pdb_gcs_uri"] = args.pdb_gcs
        else:
            _die("Provide --pdb FILE, --pdb-gcs GCS_URI, or --dataset-id DATASET_ID")
    if args.fixed:
        params["fixed_positions"] = args.fixed
    _run_model_job("proteinmpnn", params, args)


def cmd_esm2(args: argparse.Namespace) -> None:
    params: dict = {}
    if not getattr(args, "dataset_id", None):
        params["fasta_str"] = _read_fasta(args)
    if args.mask:
        params["mask_positions"] = args.mask
    _run_model_job("esm2", params, args)


def cmd_boltz(args: argparse.Namespace) -> None:
    params: dict = {
        "num_recycles": args.recycles,
        "use_msa": not args.no_msa,
    }
    if not getattr(args, "dataset_id", None):
        params["fasta_str"] = _read_fasta(args)
    _run_model_job("boltz", params, args)
