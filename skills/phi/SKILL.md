---
name: phi
description: "Run phi CLI commands for the dyno protein analysis platform: fetch structures, run folding/inverse-folding pipelines (ESMFold, AlphaFold2, ProteinMPNN), filter and score candidates, download results, and run research queries. Use when the user asks about phi commands, uploading PDB/CIF files, or scoring designs."
argument-hint: "[command] [options]"
allowed-tools: Bash(phi *)
---

# phi CLI skill

`phi` is the command-line interface for **dyno-phi**, a protein structure
analysis and scoring platform.

Install it from PyPI:

```bash
pip install dyno-phi
```

Configure with a single environment variable (or via `.phi/state.json` if
already cached from a previous session):

```bash
export DYNO_API_KEY=ak_...
```

The API key is also stored in `.phi/state.json` after first use — check there
if you're unsure whether one is already configured.

Verify connectivity:

```bash
phi login
```

After every command the CLI prints the active dataset and a direct link to the
web dashboard:

```
Active: dataset [<dataset_id>] · job [<job_id>]
Dashboard: https://design.dynotx.com/dashboard/datasets/<dataset_id>
```

---

## End-to-end workflow

```
research → fetch → upload → filter → download
```

Each step is optional depending on your starting point.

---

## 1. Research a target

```bash
# Biological research query with literature citations
phi research \
  --question "What are the key binding hotspots on PD-L1?" \
  --target PD-L1 \
  --structures \
  --dataset-id <ID>

# Stream results live
# Append notes to a local file and cloud storage
phi research --question "..." --notes-file ./research.md --dataset-id <ID>

# View accumulated notes for a dataset
phi notes <DATASET_ID>
```

---

## 2. Fetch and prepare structures

`phi fetch` downloads a structure from RCSB PDB or the AlphaFold DB, optionally
crops it by chain or residue range, and uploads it to a new dataset.

```bash
# Fetch a PDB entry (all chains)
phi fetch --pdb 6M0J

# Fetch a specific chain and residue range
phi fetch --pdb 6M0J --chain A --start 1 --end 200

# Fetch an AlphaFold DB prediction and trim low-confidence regions (pLDDT < 70)
phi fetch --uniprot P12345 --plddt-cutoff 70

# Upload the prepared structure to a new dataset
phi fetch --pdb 6M0J --chain A --upload
```

---

## 3. Upload an existing structure set

```bash
# Upload a directory of PDB / CIF files (auto-expands directories)
phi upload ./designs/

# Upload specific files
phi upload binder_001.cif binder_002.pdb

# Cache the dataset ID for subsequent commands
phi use <DATASET_ID>
```

---

## 4. Run the filter pipeline

`phi filter` scores candidates end-to-end:
inverse folding (ProteinMPNN) → folding (ESMFold) → complex folding (AlphaFold2) → score.

```bash
# Run with the default filter preset and wait for completion
phi filter --dataset-id <DATASET_ID> --preset default --wait

# Or, if a dataset is cached:
phi filter --preset default --wait

# Download results immediately after completion:
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

**MSA tool for complex folding:**

```bash
# Default: mmseqs2 (uses sequence homologs — best for natural-like designs)
phi filter --preset default --msa-tool mmseqs2 --wait

# single_sequence: skip MSA — best for truly novel de novo binders
phi filter --preset relaxed --msa-tool single_sequence --wait
```

---

## 5. Run individual tools

All tool commands accept `--dataset-id ID` or use the cached dataset. Add `--wait` to
poll until completion.

#### Inverse folding (ProteinMPNN)
```bash
phi inverse_folding --dataset-id <ID> --wait
phi proteinmpnn --dataset-id <ID> --num-sequences 10 --temperature 0.1 --wait
```

#### Folding (ESMFold)
```bash
phi folding --dataset-id <ID> --wait
phi esmfold --dataset-id <ID> --recycles 3 --wait
```

#### Complex folding (AlphaFold2 multimer)
```bash
phi complex_folding --dataset-id <ID> --wait
phi alphafold --dataset-id <ID> \
  --models 1,2 \
  --model-type multimer_v3 \
  --msa-tool mmseqs2 \
  --wait
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

## 6. Download results

```bash
# Download to ./results (default) — key files: structures/, scores/
phi download

# Download a specific job
phi download <JOB_ID> --out ./my-results

# Download everything including MSA files and archives
phi download --all
```

`phi download` organizes output into:
- `structures/` — PDB files
- `scores/` — scores.csv and raw JSON sidecars
- `scores.csv` — merged scores table

---

## 7. View scores

```bash
# Display scores table for the last job
phi scores

# Display top-20 candidates for a specific job
phi scores <JOB_ID> --top 20

# Save scores CSV locally
phi scores --out ./scores.csv
```

---

## Job management

```bash
phi jobs                        # List recent jobs
phi jobs --status completed
phi status <JOB_ID>             # Check a specific job
phi cancel <JOB_ID>
phi logs <JOB_ID>               # Print the log stream URL
```

---

## Dataset management

```bash
phi datasets                    # List all datasets
phi dataset <DATASET_ID>        # Show details
phi use <DATASET_ID>            # Cache as the current working dataset
```

---

## Tips

### State caching
`phi use <DATASET_ID>` writes to `.phi/state.json` in the current directory.
`phi filter` also caches `last_job_id`, so `phi download` and `phi scores` work
without arguments after a filter run. The API key is also cached here.

### Polling interval
```bash
phi filter --preset default --wait --poll-interval 10
```

### Key scoring metrics

| Metric | Source | Good threshold |
|--------|--------|----------------|
| `plddt` | ESMFold | ≥ 0.80 |
| `ptm` | AlphaFold2 | ≥ 0.55 |
| `af2_iptm` | AlphaFold2 multimer | ≥ 0.50 |
| `af2_ipae` | AlphaFold2 multimer | ≤ 0.35 |
| `rmsd` | Binder vs design backbone | ≤ 3.5 Å |

### Tool name aliases

| Generic name | Tool | Notes |
|---|---|---|
| `folding` | ESMFold | Fast single-sequence folding |
| `complex_folding` | AlphaFold2 multimer | Binder–target complex |
| `inverse_folding` | ProteinMPNN | Sequence design from backbone |

---

## Common workflows

### Research-guided scoring
```bash
phi research \
  --question "Which residues on EGFR domain III are critical for antibody binding?" \
  --target EGFR --structures --notes-file ./research.md

phi fetch --pdb 1IVO --chain A --start 300 --end 500 --upload

phi filter --preset relaxed --msa-tool single_sequence --wait --out ./results
phi scores
```

### Upload existing structures and score
```bash
phi upload ./designs/
phi filter --preset default --wait --out ./results
phi scores
```

### Relax thresholds for novel de novo binders
```bash
phi filter --preset relaxed --msa-tool single_sequence --wait
```
