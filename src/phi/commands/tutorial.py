from __future__ import annotations

import argparse
import urllib.request
from pathlib import Path

from phi.api import _request
from phi.config import _save_state, _ssl_context
from phi.display import _C_BLUE, _C_SAND, _die, console


def cmd_tutorial(args: argparse.Namespace) -> None:
    out = Path(args.out)

    # ── 1. Fetch manifest (standard Clerk JWT auth, same as all endpoints) ───
    console.print("[dim]Fetching tutorial dataset …[/]")
    try:
        manifest = _request("GET", "/tutorial")
    except Exception as exc:
        _die(
            f"Could not reach the tutorial endpoint: {exc}\n"
            "  Check your connection and API key, then try again."
        )

    files: list[dict] = manifest.get("files", [])
    dataset_id: str | None = manifest.get("dataset_id")
    message: str | None = manifest.get("message")

    if not files:
        _die("No tutorial files returned by the API.")

    # ── 2. Download each file (plain HTTP — signed URLs are self-authenticating)
    out.mkdir(parents=True, exist_ok=True)
    console.print(f"  Downloading {len(files)} file(s) to [{_C_BLUE}]{out}/[/] …\n")

    for entry in files:
        filename: str = entry["filename"]
        url: str = entry["url"]
        dest = out / filename
        dest.parent.mkdir(parents=True, exist_ok=True)
        try:
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req, context=_ssl_context()) as resp:
                dest.write_bytes(resp.read())
            console.print(f"  [bold {_C_SAND}]✓[/]  {filename}")
        except Exception as exc:
            _die(f"Failed to download {filename}: {exc}")

    # ── 3. Cache dataset_id so phi filter needs zero extra flags ─────────────
    if dataset_id:
        _save_state({"dataset_id": dataset_id})
        console.print(
            f"\n[dim]dataset_id [{_C_BLUE}]{dataset_id}[/] cached — "
            f"run [bold]phi filter[/] to start scoring.[/]"
        )

    # ── 4. Print step-by-step guide ──────────────────────────────────────────
    if message:
        console.print(f"\n[dim]{message}[/]")

    if dataset_id:
        upload_step = "[dim]  (skipped — dataset already ready)[/]"
    else:
        upload_step = f"  [{_C_SAND}]phi upload {out}/[/]"

    console.print(f"""
[bold]── Tutorial: PD-L1 binder scoring pipeline ──────────────────[/]

You have {len(files)} example binder structures in [{_C_BLUE}]{out}/[/].

[bold]Step 1 — Upload[/]
{upload_step}

[bold]Step 2 — Run the filter pipeline[/]
  [{_C_SAND}]phi filter --preset default --wait[/]

  Runs:  ProteinMPNN → ESMFold → AlphaFold2 → score
  Typical runtime: 10–30 min for {len(files)} structures.

[bold]Step 3 — View scores[/]
  [{_C_SAND}]phi scores[/]

[bold]Step 4 — Download results[/]
  [{_C_SAND}]phi download --out ./results[/]

[bold]Dashboard[/]
  [{_C_BLUE}]https://design.dynotx.com/dashboard[/]
""")
