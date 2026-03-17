# dyno-phi Claude Code Plugin

A Claude Code plugin that gives Claude the ability to run `phi` CLI commands for the **dyno protein analysis platform**.

## What it does

Installs the `/dyno-phi:phi` skill, which lets Claude:

- Run the full scoring workflow (research → fetch → upload → filter → download)
- Submit jobs to ESMFold, AlphaFold2, and ProteinMPNN
- Score and filter candidates using pLDDT, pTM, ipTM, iPAE, and RMSD
- Manage datasets and jobs

## Prerequisites

Install the CLI and set your API key:

```bash
pip install dyno-phi
export DYNO_API_KEY=ak_...
phi login   # verify connectivity
```

## Usage (local / development)

Load the plugin directly with `--plugin-dir`:

```bash
claude --plugin-dir ./phi-plugin
```

Then use the skill in Claude Code:

```
/dyno-phi:phi upload ./designs/
/dyno-phi:phi filter --preset default --wait
/dyno-phi:phi scores
```

Or just describe what you want — Claude will invoke the skill automatically:

```
Upload the PDB files in ./designs/ and run the default filter pipeline.
```

## Skill invocation

| Invocation | Description |
|---|---|
| `/dyno-phi:phi` | Direct skill invocation |
| Natural language | Claude invokes automatically based on context |

## Source

Part of the [phi-cli](https://github.com/dynotx/phi-cli) repository.
