import argparse
import json

from rich.panel import Panel
from rich.text import Text

from phi.api import _api_key, _request
from phi.config import _base_url
from phi.display import _C_BLUE, _C_SAND, _die, console
from phi.types import PhiApiError


def cmd_login(args: argparse.Namespace) -> None:
    key = _api_key()
    masked = key[:8] + "…" if len(key) > 8 else key
    base = _base_url()

    try:
        me = _request("GET", "/auth/me")
        if args.json:
            print(json.dumps(me, indent=2))
            return

        content = Text()
        content.append("✓ Logged in\n\n", style=f"bold {_C_SAND}")
        content.append("endpoint  ", style="dim")
        content.append(f"{base}\n")
        content.append("API key   ", style="dim")
        content.append(f"{masked}\n\n")
        content.append("Identity\n", style="bold")
        for label, key_name in [
            ("user_id     ", "user_id"),
            ("email       ", "email"),
            ("display_name", "display_name"),
            ("org_id      ", "org_id"),
            ("org_name    ", "org_name"),
        ]:
            val = me.get(key_name) or "—"
            content.append(f"  {label}  ", style="dim")
            content.append(f"{val}\n")
        content.append("\n")
        content.append("Tip: ", style="bold dim")
        content.append("cache these to skip /auth/me on uploads:\n", style="dim")
        content.append(
            f"  export DYNO_USER_ID={me.get('user_id', 'YOUR_USER_ID')}\n",
            style=f"dim {_C_BLUE}",
        )
        content.append(
            f"  export DYNO_ORG_ID={me.get('org_id', 'YOUR_ORG_ID')}",
            style=f"dim {_C_BLUE}",
        )
        console.print(
            Panel(content, title=f"[{_C_BLUE}]dyno phi[/]", border_style=_C_BLUE, padding=(1, 2))
        )
        return

    except PhiApiError as exc:
        if "404" not in str(exc):
            _die(str(exc))

    try:
        _request("GET", "/jobs/?page_size=1")
        if args.json:
            print(json.dumps({"status": "connected", "auth_me": "not_deployed"}, indent=2))
            return

        content = Text()
        content.append("✓ Logged in\n\n", style=f"bold {_C_SAND}")
        content.append("endpoint  ", style="dim")
        content.append(f"{base}\n")
        content.append("API key   ", style="dim")
        content.append(f"{masked}\n\n")
        content.append("Note: ", style="bold dim")
        content.append(
            "User identity will appear here once GET /auth/me is deployed on this environment.",
            style="dim",
        )
        console.print(
            Panel(content, title=f"[{_C_BLUE}]dyno phi[/]", border_style=_C_BLUE, padding=(1, 2))
        )

    except PhiApiError as probe_exc:
        msg = f"Authentication failed — {probe_exc}"
        if "401" in str(probe_exc) and key.startswith("ak_"):
            msg += (
                "\n  This endpoint may not yet accept Clerk API keys (ak_…). "
                "Check backend config or use the API that is wired to Clerk."
            )
        _die(msg)
