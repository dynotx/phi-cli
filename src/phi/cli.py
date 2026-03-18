#!/usr/bin/env python3

import phi.config as config
from phi.commands.auth import cmd_login
from phi.commands.datasets import cmd_dataset, cmd_datasets, cmd_ingest_session, cmd_upload, cmd_use
from phi.commands.filter import cmd_filter
from phi.commands.jobs import cmd_cancel, cmd_download, cmd_jobs, cmd_logs, cmd_scores, cmd_status
from phi.commands.models import (
    cmd_alphafold,
    cmd_boltz,
    cmd_boltzgen,
    cmd_esm2,
    cmd_esmfold,
    cmd_proteinmpnn,
    cmd_rfdiffusion3,
)
from phi.commands.research import cmd_notes, cmd_research
from phi.commands.structure import cmd_fetch
from phi.commands.tutorial import cmd_tutorial
from phi.config import _load_state
from phi.display import _C_BLUE, _die, console
from phi.parser import build_parser
from phi.types import PhiApiError

COMMANDS = {
    "login": cmd_login,
    "tutorial": cmd_tutorial,
    "upload": cmd_upload,
    "ingest-session": cmd_ingest_session,
    "datasets": cmd_datasets,
    "dataset": cmd_dataset,
    "esmfold": cmd_esmfold,
    "folding": cmd_esmfold,
    "alphafold": cmd_alphafold,
    "complex_folding": cmd_alphafold,
    "proteinmpnn": cmd_proteinmpnn,
    "inverse_folding": cmd_proteinmpnn,
    "esm2": cmd_esm2,
    "boltz": cmd_boltz,
    "rfdiffusion3": cmd_rfdiffusion3,
    "design": cmd_rfdiffusion3,
    "boltzgen": cmd_boltzgen,
    "fetch": cmd_fetch,
    "research": cmd_research,
    "notes": cmd_notes,
    "status": cmd_status,
    "jobs": cmd_jobs,
    "logs": cmd_logs,
    "cancel": cmd_cancel,
    "use": cmd_use,
    "download": cmd_download,
    "filter": cmd_filter,
    "scores": cmd_scores,
}


def _print_state_footer() -> None:
    state = _load_state()
    dataset_id = state.get("dataset_id")
    job_id = state.get("last_job_id")
    if not dataset_id and not job_id:
        return

    parts: list[str] = []
    if dataset_id:
        parts.append(f"dataset [{_C_BLUE}]{dataset_id}[/{_C_BLUE}]")
    if job_id:
        parts.append(f"job [{_C_BLUE}]{job_id}[/{_C_BLUE}]")

    console.print(f"[dim]Active: {' · '.join(parts)}[/dim]")
    if dataset_id:
        dashboard_url = f"https://design.dynotx.com/dashboard/datasets/{dataset_id}"
        console.print(f"[dim]Dashboard: [link={dashboard_url}]{dashboard_url}[/link][/dim]")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.poll_interval is not None:
        config.POLL_INTERVAL = args.poll_interval
    try:
        COMMANDS[args.command](args)
        _print_state_footer()
    except PhiApiError as exc:
        _die(str(exc))


if __name__ == "__main__":
    main()
