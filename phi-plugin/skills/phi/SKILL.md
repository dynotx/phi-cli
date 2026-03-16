---
name: phi
description: "Run phi CLI commands for the dyno protein design platform: fetch structures, design binders (RFDiffusion3, BoltzGen), run folding/inverse-folding pipelines (ESMFold, AlphaFold2, ProteinMPNN), filter and score candidates, download results, and run research queries. Use when the user asks about phi commands, binder design, uploading PDB/CIF files, or scoring designs."
argument-hint: "[command] [options]"
allowed-tools: Bash(phi *)
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

## End-to-end workflow

```
research → fetch → design → filter → download
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
phi research --question "..." --stream

# Append notes to a local file and cloud storage
phi research --question "..." --notes-file ./research.md --dataset-id <ID>

# View accumulated notes for a dataset
phi notes <DATASET_ID>
```

---

## 2. Fetch and prepare structures

`phi fetch` downloads a structure from RCSB PDB or the AlphaFold DB, optionally crops
it by chain or residue range, and uploads it to a new dataset ready for design.

```bash
# Fetch a PDB entry (all chains)
phi fetch --pdb 6M0J

# Fetch a specific chain and residue range
phi fetch --pdb 6M0J --chain A --start 1 --end 200

# Fetch an AlphaFold DB prediction and trim low-confidence regions (pLDDT < 70)
phi fetch --uniprot P12345 --plddt-cutoff 70

# Upload the prepared structure to a new dataset for use with phi design
phi fetch --pdb 6M0J --chain A --upload
```

After `--upload`, a new `dataset_id` is cached. The output prints the full `phi design`
command ready to run.

---

## 3. Design binders

### RFDiffusion3 (backbone diffusion — recommended default)

```bash
# Alias: phi design
phi design --fasta target.fasta --hotspots A25,A30,A35 --num-designs 50 --wait

# Specify binder length range
phi design --fasta target.fasta --binder-min-length 60 --binder-max-length 100 --num-designs 50 --wait

# Use a structure file from a cached dataset
phi design --dataset-id <ID> --hotspots A25,A30 --num-designs 50 --wait

# Specify partial diffusion steps (for motif scaffolding)
phi rfdiffusion3 --fasta target.fasta --hotspots A25,A30 --partial-diffusion-steps 10 --num-designs 50 --wait
```

### BoltzGen (all-atom diffusion — for high-quality production runs)

```bash
# Full pipeline from a design YAML
phi boltzgen --yaml design.yaml --protocol protein-anything --num-designs 50 --wait

# Specify a budget (final diversity-filtered set size)
phi boltzgen --yaml design.yaml --num-designs 1000 --budget 50 --wait

# Run only specific pipeline steps
phi boltzgen --yaml design.yaml --boltzgen-steps "design inverse_folding" --num-designs 50 --wait

# Inverse folding only (resequence an existing backbone from a YAML spec)
phi boltzgen --yaml backbone.yaml --only-inverse-fold --inverse-fold-num-sequences 10 --wait

# Use a structure already uploaded to cloud storage
phi boltzgen --yaml design.yaml --yaml-gcs gs://bucket/design.yaml --structure-gcs gs://bucket/target.cif --wait
```

**BoltzGen protocols:**

| Protocol | Use |
|---|---|
| `protein-anything` | Design proteins to bind proteins or peptides (default) |
| `peptide-anything` | Design (cyclic) peptides to bind proteins |
| `protein-small_molecule` | Design proteins to bind small molecules |
| `antibody-anything` | Design antibody CDRs |
| `nanobody-anything` | Design nanobody CDRs |
| `protein-redesign` | Redesign or optimize existing proteins |

**BoltzGen pipeline steps** (pass to `--boltzgen-steps`):
`design`, `inverse_folding`, `folding`, `design_folding`, `affinity`, `analysis`, `filtering`

---

## 4. Upload an existing design set

```bash
# Upload a directory of PDB / CIF files (auto-expands directories)
phi upload ./designs/

# Upload specific files
phi upload binder_001.cif binder_002.pdb

# Cache the dataset ID for subsequent commands
phi use <DATASET_ID>
```

---

## 5. Run the filter pipeline

`phi filter` scores binder candidates end-to-end:
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

## 6. Run individual tools

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

## 7. Download results

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

## 8. View scores

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
`phi use <DATASET_ID>` writes to `.phi-state.json` in the current directory.
`phi filter` also caches `last_job_id`, so `phi download` and `phi scores` work
without arguments after a filter run.

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
| `design` | RFDiffusion3 | Backbone diffusion (default) |

---

## Common workflows

### Research-guided design
```bash
phi research \
  --question "Which residues on EGFR domain III are critical for antibody binding?" \
  --target EGFR --structures --notes-file ./research.md

phi fetch --pdb 1IVO --chain A --start 300 --end 500 --upload

phi design --hotspots A310,A315,A320 --num-designs 100 --wait

phi filter --preset relaxed --msa-tool single_sequence --wait --out ./results
phi scores
```

### Upload existing designs and score
```bash
phi upload ./designs/
phi filter --preset default --wait --out ./results
phi scores
```

### Relax thresholds for novel de novo binders
```bash
phi filter --preset relaxed --msa-tool single_sequence --wait
```

### BoltzGen inverse folding only (resequence a backbone)
```bash
phi boltzgen \
  --yaml backbone_spec.yaml \
  --only-inverse-fold \
  --inverse-fold-num-sequences 10 \
  --wait
```

### Full BoltzGen production run
```bash
phi boltzgen \
  --yaml design_spec.yaml \
  --protocol protein-anything \
  --num-designs 10000 \
  --budget 100 \
  --wait
```
