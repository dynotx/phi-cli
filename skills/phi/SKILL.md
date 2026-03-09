---
name: phi
description: >
  Use the `phi` CLI to interact with the dyno-phi protein design platform:
  uploading structure datasets, running folding / inverse-folding / complex-folding
  jobs, filtering binder candidates, downloading results, and running research queries.
  Activate automatically when the user asks about phi commands, binder design
  workflows, uploading PDB/CIF files, running AlphaFold2 / ESMFold / ProteinMPNN,
  or scoring/filtering designs.
---

# phi CLI skill

`phi` is the command-line interface for **dyno-phi**, a protein design platform.
Install it from PyPI:

```bash
pip install dyno-phi
```

Configure with a single environment variable (or a `.env` file at the repo root):

```bash
export DYNO_API_KEY=sk-...
# Optional — defaults to the hosted API
export DYNO_API_BASE_URL=https://api.dynotx.com
```

Verify connectivity:

```bash
phi login
```

---

## Core workflow

```
upload designs → filter (inverse folding → folding → complex folding → score) → download results
```

### 1. Upload a dataset

```bash
# Upload a directory of PDB / CIF files (auto-expands directories)
phi upload ./designs/

# Upload specific files
phi upload binder_001.cif binder_002.pdb

# Cache the dataset ID for subsequent commands
phi use <DATASET_ID>
```

After upload, a summary panel shows the `dataset_id` and the next commands to run.

---

### 2. Run the filter pipeline

The `phi filter` command runs the full binder scoring pipeline in one step:
inverse folding (ProteinMPNN) → folding (ESMFold) → complex folding (AlphaFold2) → score.

```bash
# Run with the default filter preset and wait for completion
phi filter --dataset-id <DATASET_ID> --preset default --wait

# Or, if a dataset is already cached:
phi filter --preset default --wait

# Show scores table immediately after completion:
phi filter --preset default --wait --out ./results
```

**Filter presets:**

| Preset | pLDDT | pTM | ipTM | iPAE | RMSD |
|--------|-------|-----|------|------|------|
| `default` | ≥0.80 | ≥0.55 | ≥0.50 | ≤0.35 | ≤3.5 Å |
| `relaxed` | ≥0.80 | ≥0.45 | ≥0.50 | ≤0.40 | ≤4.5 Å |

Override any threshold individually:

```bash
phi filter --preset default --plddt-threshold 0.75 --rmsd-threshold 4.0 --wait
```

**MSA tool for complex folding** (important for novel designs):

```bash
# Default: mmseqs2 (fastest, uses sequence homologs)
phi filter --preset default --msa-tool mmseqs2 --wait

# single_sequence: skip MSA — best for truly novel de novo binders
phi filter --preset default --msa-tool single_sequence --wait
```

---

### 3. Run individual tools

All tool commands accept `--dataset-id ID` or use the cached dataset. Add `--wait` to
poll until completion.

#### Folding (ESMFold)
```bash
phi folding --dataset-id <ID> --wait
# or using the canonical name:
phi esmfold --dataset-id <ID> --recycles 3 --wait
```

#### Complex folding (AlphaFold2)
```bash
phi complex_folding --dataset-id <ID> --wait
# or using the canonical name with full options:
phi alphafold --dataset-id <ID> \
  --models 1,2 \
  --model-type multimer_v3 \
  --msa-tool mmseqs2 \
  --wait
```

#### Inverse folding (ProteinMPNN)
```bash
phi inverse_folding --dataset-id <ID> --wait
# or using the canonical name:
phi proteinmpnn --dataset-id <ID> --num-sequences 10 --temperature 0.1 --wait
```

#### ESM2 language model scoring
```bash
phi esm2 --dataset-id <ID> --wait
```

#### Boltz complex prediction
```bash
phi boltz --dataset-id <ID> --recycles 3 --wait
```

---

### 4. Download results

```bash
# Download to ./results (default) — key files: structures/, scores/
phi download

# Download a specific job
phi download <JOB_ID> --out ./my-results

# Download everything including MSA files and zip archives
phi download --all
```

`phi download` organizes output into:
- `structures/` — PDB files
- `scores/` — scores.csv and raw JSON sidecars
- `scores.csv` — merged scores table

---

### 5. View scores

```bash
# Display scores table for the last job
phi scores

# Display top-20 candidates for a specific job
phi scores <JOB_ID> --top 20

# Save scores CSV locally
phi scores --out ./scores.csv
```

---

## Research queries

```bash
# Biological research query with literature citations
phi research \
  --question "What are the key binding hotspots on PD-L1?" \
  --target PD-L1 \
  --structures \
  --dataset-id <ID>

# Stream results live
phi research --question "..." --stream

# Append notes to a local file and cloud storage
phi research --question "..." --notes-file ./research.md --dataset-id <ID>

# View accumulated notes for a dataset
phi notes <DATASET_ID>
```

---

## Job management

```bash
# List recent jobs
phi jobs

# Filter by status
phi jobs --status completed
phi jobs --status failed

# Check a specific job
phi status <JOB_ID>

# Cancel a running job
phi cancel <JOB_ID>

# Print the log stream URL
phi logs <JOB_ID> --follow
```

---

## Dataset management

```bash
# List all datasets
phi datasets

# Show details for a specific dataset
phi dataset <DATASET_ID>

# Cache a dataset as the current working dataset
phi use <DATASET_ID>
```

---

## Tuning and tips

### Polling interval
All polling commands respect `--poll-interval SECONDS` (global flag):

```bash
phi filter --preset default --wait --poll-interval 10
```

### Caching
`phi use <DATASET_ID>` writes to `.phi-state.json` in the current directory.
Commands that accept `--dataset-id` will fall back to this cached value.
`phi filter` also caches the `last_job_id`, so `phi download` and `phi scores`
work without arguments after a filter run.

### Key metrics generated by the pipeline

| Metric | Source | Good threshold |
|--------|--------|----------------|
| `plddt` | ESMFold | ≥ 0.80 |
| `ptm` | AlphaFold2 | ≥ 0.55 |
| `af2_iptm` | AlphaFold2 multimer | ≥ 0.50 |
| `af2_ipae` | AlphaFold2 multimer | ≤ 0.35 |
| `rmsd` | Binder vs design backbone | ≤ 3.5 Å |

### Tool name mapping

The generic step names alias to specific models:

| Generic name | Tool | Notes |
|---|---|---|
| `folding` | ESMFold | Fast single-sequence folding (~1 min) |
| `complex_folding` | AlphaFold2 multimer | Binder–target complex (8–15 min) |
| `inverse_folding` | ProteinMPNN | Sequence design from backbone (1–2 min) |

The specific tool names (`esmfold`, `alphafold`, `proteinmpnn`) are always available
and accept full model-configuration flags. The generic names use sensible defaults.

---

## Common workflows

### Upload and run the full filter pipeline
```bash
phi upload ./designs/
phi filter --preset default --wait --out ./results
phi scores
```

### Relax thresholds for novel designs (no natural homologs)
```bash
phi filter --preset relaxed --msa-tool single_sequence --wait
```

### Custom threshold override
```bash
phi filter --plddt-threshold 0.75 --iptm-threshold 0.45 --rmsd-threshold 4.0 --wait
```

### Research-guided design
```bash
phi research \
  --question "Which residues on EGFR domain III are most critical for antibody binding?" \
  --target EGFR \
  --structures \
  --dataset-id $DATASET_ID \
  --notes-file ./research.md
```
