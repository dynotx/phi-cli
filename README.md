# dyno-phi

**Phi CLI for the dyno protein structure analysis platform.**

`phi` is the command-line interface for uploading protein structures, running
structure prediction and inverse-folding pipelines, filtering and scoring
candidates, and downloading results from the dyno API.

Results and scores are viewable in the web dashboard at
`design.dynotx.com/dashboard/datasets/<dataset_id>`.

---

## Table of contents

- [Installation](#installation)
- [Authentication](#authentication)
- [Quick start](#quick-start)
- [Command reference](#command-reference)
- [Filter presets](#filter-presets)
- [State caching](#state-caching)
- [Claude Code skill](#claude-code-skill)
- [Development](#development)

---

## Installation

```bash
pip install dyno-phi
```

Requires Python ≥ 3.9.

---

## Authentication

Create an API key at **Settings → API keys** in the dyno web app
(`https://design.dynotx.com/dashboard/settings`), then export it:

```bash
export DYNO_API_KEY=ak_...
```

The key is cached to `.phi/state.json` on first use so you don't need to
re-export it in future sessions.

Verify your connection:

```bash
phi login
```

---

## Quick start

### Single-sequence / single-structure jobs

```bash
# Structure prediction (ESMFold)
phi folding --fasta sequences.fasta

# Complex structure prediction (AlphaFold2 multimer)
phi complex_folding --fasta binder_target.fasta

# Sequence design via inverse folding (ProteinMPNN)
phi inverse_folding --pdb design.pdb --num-sequences 20
```

### Batch scoring workflow

```bash
# 1. Upload a directory of PDB/CIF files
phi upload ./designs/

# Output:
#   dataset_id  d7c3a1b2-...
#   Dashboard:  https://design.dynotx.com/dashboard/datasets/d7c3a1b2-...
#   Run a job against this dataset:
#     phi folding          --dataset-id d7c3a1b2-...
#     phi complex_folding  --dataset-id d7c3a1b2-...
#     phi inverse_folding  --dataset-id d7c3a1b2-...
#     phi filter           --dataset-id d7c3a1b2-... --preset default --wait

# 2. Run the full filter pipeline (inverse folding → folding → complex folding → score)
phi filter --dataset-id d7c3a1b2-... --preset default --wait

# 3. Download results (structures, scores CSV, raw score JSONs)
phi download --out ./results/
```

After each command, `phi` prints the active dataset and job IDs and a link to
the dashboard:

```
Active: dataset [d7c3a1b2-...] · job [cb4553f5-...]
Dashboard: https://design.dynotx.com/dashboard/datasets/d7c3a1b2-...
```

---

## Command reference

| Command | Alias | Description |
|---|---|---|
| `phi login` | — | Verify API key and print identity |
| `phi upload` | — | Upload PDB/CIF files or a directory |
| `phi fetch` | — | Download a structure from RCSB PDB or AlphaFold DB, crop, and optionally upload |
| `phi datasets` | — | List datasets |
| `phi dataset` | — | Show dataset details |
| `phi use <dataset_id>` | — | Set active dataset (cached to `.phi/state.json`) |
| `phi folding` | `esmfold` | Single-sequence structure prediction (ESMFold) |
| `phi complex_folding` | `alphafold` | Multi-chain complex prediction (AlphaFold2 multimer) |
| `phi inverse_folding` | `proteinmpnn` | Sequence design via inverse folding (ProteinMPNN) |
| `phi esm2` | — | Sequence embedding and scoring (ESM2) |
| `phi boltz` | — | Structure prediction (Boltz-1) |
| `phi filter` | — | Full filter pipeline: inverse folding → folding → complex folding → score |
| `phi status <job_id>` | — | Poll job status |
| `phi jobs` | — | List recent jobs |
| `phi logs <job_id>` | — | Stream job logs |
| `phi cancel <job_id>` | — | Cancel a running job |
| `phi scores` | — | Display scores table for a completed filter job |
| `phi download` | — | Download job artifacts (structures, scores, raw JSONs) |
| `phi research` | — | Run a research query against the platform |
| `phi notes` | — | Manage dataset research notes |

### Common flags

| Flag | Commands | Description |
|---|---|---|
| `--dataset-id ID` | most | Target dataset (omit to use cached) |
| `--wait` | most | Poll until job completes, then print summary |
| `--out DIR` | `download`, `scores` | Output directory (default: `./results`) |
| `--preset default\|relaxed` | `filter` | Filter threshold preset |
| `--num-sequences N` | `inverse_folding` | Sequences per design (default: 4) |
| `--models 1,2` | `complex_folding` | AlphaFold2 model numbers (default: 1,2,3) |
| `--poll-interval S` | global | Seconds between status polls (default: 5) |

---

## Filter presets

`phi filter` applies a multi-stage quality-control pipeline and scores each
design against configurable thresholds.

| Metric | `default` | `relaxed` | Description |
|---|---|---|---|
| pLDDT | ≥ 0.80 | ≥ 0.80 | ESMFold per-residue confidence |
| pTM | ≥ 0.55 | ≥ 0.45 | Global TM-score proxy (ESMFold) |
| ipTM | ≥ 0.50 | ≥ 0.50 | Interface pTM (AF2 multimer) |
| iPAE | ≤ 10.85 Å | ≤ 12.4 Å | AF2 interface PAE in Å |
| RMSD | ≤ 3.5 Å | ≤ 4.5 Å | Backbone RMSD vs. reference |

Override any threshold with an explicit flag:

```bash
phi filter --dataset-id ... --plddt 0.75 --iptm 0.45
```

---

## State caching

`phi` caches the most recently used dataset ID, job ID, and API key in
`.phi/state.json` so you don't need to pass `--dataset-id` or re-export your
key repeatedly:

```bash
phi use d7c3a1b2-...         # set active dataset
phi filter --preset default  # uses cached dataset
phi scores                   # uses cached job
phi download --out ./results # uses cached job
```

The dashboard URL for the active dataset is printed after every command:

```
Dashboard: https://design.dynotx.com/dashboard/datasets/d7c3a1b2-...
```

---

## Claude Code skill

The `phi` skill is bundled at `skills/phi/SKILL.md` and is automatically
available when you open this repo in Claude Code (Cursor or the `claude` CLI).
No installation needed — just open the project and ask naturally:

```
Upload the PDB files in ./examples/ and run the default filter pipeline.
```

**To make the skill available in all your projects** (outside this repo):

```bash
mkdir -p ~/.claude/skills/phi
cp skills/phi/SKILL.md ~/.claude/skills/phi/SKILL.md
```

---

## Development

```bash
git clone https://github.com/dynotx/phi-cli
cd phi-cli
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# Lint
ruff check src/

# Type check
mypy src/phi/

# Run tests
pytest tests/
```

### Releasing to PyPI

Releases are published via GitHub Actions. Push a version tag to trigger the workflow:

```bash
# Bump src/phi/_version.py, update CHANGELOG.md, then:
git tag v0.1.0
git push origin main --tags
```

The workflow publishes to **TestPyPI** automatically, then waits for manual
approval before publishing to the real **PyPI**. See
`.github/workflows/publish.yml` for details.
