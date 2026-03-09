# dyno-phi

**Phi CLI and biomodals for the dyno protein design platform.**

`phi` is the command-line interface for submitting protein design jobs, managing
datasets, running structure prediction and inverse-folding pipelines, and
downloading results from the dyno API.

---

## Table of contents

- [Installation](#installation)
- [Authentication](#authentication)
- [Quick start](#quick-start)
- [Command reference](#command-reference)
- [Filter presets](#filter-presets)
- [State caching](#state-caching)
- [Biomodals](#biomodals)
- [Claude Code skill](#claude-code-skill)
- [Development](#development)

---

## Installation

```bash
pip install dyno-phi
```

For local biomodal development (deploying Modal GPU apps):

```bash
pip install "dyno-phi[biomodals]"
```

Requires Python ≥ 3.11.

---

## Authentication

Create an API key at **Settings → API keys** in the dyno web app, then export it:

```bash
export DYNO_API_KEY=ak_...
```

Optionally override the API base URL (defaults to the hosted API):

```bash
export DYNO_API_BASE_URL=https://api.dynotx.com
```

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

### Batch binder design workflow

```bash
# 1. Upload a directory of PDB/CIF files
phi upload ./designs/

# Output:
#   dataset_id  d7c3a1b2-...
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

After each command, `phi` prints the active dataset and job IDs:

```
Active: dataset [d7c3a1b2-...] · job [cb4553f5-...]
```

---

## Command reference

| Command | Alias | Description |
|---|---|---|
| `phi login` | — | Verify API key and print identity |
| `phi upload` | — | Upload PDB/CIF files or a directory |
| `phi datasets` | — | List datasets |
| `phi dataset` | — | Show dataset details |
| `phi use <dataset_id>` | — | Set active dataset (cached to `.phi-state.json`) |
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
| `phi scores` | — | Display scores CSV for a completed filter job |
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
| pLDDT | ≥ 0.80 | ≥ 0.70 | ESMFold per-residue confidence |
| pTM | ≥ 0.55 | ≥ 0.45 | Global TM-score proxy (ESMFold) |
| ipTM | ≥ 0.50 | ≥ 0.40 | Interface pTM (AF2 multimer) |
| iPAE | ≤ 0.35 Å | ≤ 0.50 Å | Interface PAE (AF2 multimer) |
| RMSD | ≤ 3.5 Å | ≤ 5.0 Å | Backbone RMSD vs. reference |

Override any threshold with an explicit flag:

```bash
phi filter --dataset-id ... --plddt 0.75 --iptm 0.45
```

---

## State caching

`phi` caches the most recently used dataset ID and job ID in `.phi-state.json`
so you don't need to pass `--dataset-id` or `job_id` repeatedly:

```bash
phi use d7c3a1b2-...         # set active dataset
phi filter --preset default  # uses cached dataset
phi scores                   # uses cached job
phi download --out ./results # uses cached job
```

---

## Biomodals

The `biomodals/` directory contains self-contained [Modal](https://modal.com)
GPU apps for every model used in the platform. They can be deployed
independently and are the same apps used in production.

### Prerequisites

```bash
pip install "dyno-phi[biomodals]"
modal token new          # authenticate with Modal
```

You also need a `cloudsql-credentials` Modal secret containing
`GOOGLE_APPLICATION_CREDENTIALS_JSON` (your GCS service account JSON).

### Deploying

```bash
modal deploy biomodals/modal_alphafold.py
modal deploy biomodals/modal_esmfold.py
modal deploy biomodals/modal_proteinmpnn.py
```

### Available biomodals

| File | Tool | Description |
|---|---|---|
| `modal_alphafold.py` | AlphaFold2 | Monomer + multimer structure prediction (ColabFold MSA) |
| `modal_esmfold.py` | ESMFold | Fast single-sequence structure prediction |
| `modal_proteinmpnn.py` | ProteinMPNN | Inverse folding — design sequences for a backbone |
| `modal_boltz.py` | Boltz-1 | Open-source biomolecular structure prediction |
| `modal_boltzgen.py` | BoltzGen | Diffusion-based binder design |
| `modal_bindcraft.py` | BindCraft | End-to-end hallucination binder design |
| `modal_chai1.py` | Chai-1 | Foundation model for molecular structure |
| `modal_rfdiffusion3.py` | RFdiffusion | Backbone generation for binder scaffolds |
| `modal_rf3.py` | RF3 | RoseTTAFold3 structure prediction |
| `modal_esm2_predict_masked.py` | ESM2 | Protein language model embeddings |
| `modal_ligandmpnn.py` | LigandMPNN | Ligand-aware inverse folding |
| `modal_align_structures.py` | Biotite | Structure alignment and RMSD calculation |
| `modal_af2rank.py` | AF2Rank | Rank binders using AF2 confidence metrics |
| `modal_rso.py` | RSO | Rosetta side-chain optimization |
| `modal_tm_score.py` | TMscore | TM-score calculation |

---

## Claude Code skill

Install the `phi` skill to get conversational prompting for all `phi` commands
directly inside Claude Code (Cursor or the `claude` CLI):

```bash
# Install for all your projects
mkdir -p ~/.claude/skills/phi
cp skills/phi/SKILL.md ~/.claude/skills/phi/SKILL.md
```

Then in Claude Code you can ask naturally:

```
Upload the PDB files in ./examples/binders/ and run the default filter pipeline.
```

Or invoke directly:

```
/phi upload ./examples/binders/
```

---

## Development

```bash
git clone https://github.com/dyno-tx/phi-cli
cd phi-cli
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# Lint
ruff check src/ biomodals/

# Type check
mypy src/phi/

# Run tests
pytest tests/
```

### Releasing to PyPI

```bash
pip install build twine
python -m build
twine upload dist/*
```
