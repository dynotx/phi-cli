# Phi CLI Reference

**`phi`** is the command-line interface for the dyno protein design platform.
Submit and monitor computational biology jobs, manage datasets, run structure
prediction and inverse-folding pipelines, and download results — all from your
terminal.

**Version:** 0.1.0 · **Package:** `dyno-phi` · **Requires:** Python ≥ 3.11

---

## Contents

- [Installation & authentication](#installation--authentication)
- [Global flags](#global-flags)
- [State caching](#state-caching)
- [Command index](#command-index)
- [Detailed reference](#detailed-reference)
  - [phi login](#phi-login)
  - [phi fetch](#phi-fetch)
  - [phi upload](#phi-upload)
  - [phi use](#phi-use)
  - [phi datasets](#phi-datasets)
  - [phi dataset](#phi-dataset)
  - [phi ingest-session](#phi-ingest-session)
  - [phi design / rfdiffusion3](#phi-design--rfdiffusion3)
  - [phi boltzgen](#phi-boltzgen)
  - [phi folding / esmfold](#phi-folding--esmfold)
  - [phi complex_folding / alphafold](#phi-complex_folding--alphafold)
  - [phi inverse_folding / proteinmpnn](#phi-inverse_folding--proteinmpnn)
  - [phi esm2](#phi-esm2)
  - [phi boltz](#phi-boltz)
  - [phi filter](#phi-filter)
  - [phi status](#phi-status)
  - [phi jobs](#phi-jobs)
  - [phi logs](#phi-logs)
  - [phi cancel](#phi-cancel)
  - [phi scores](#phi-scores)
  - [phi download](#phi-download)
  - [phi research](#phi-research)
  - [phi notes](#phi-notes)
- [Filter presets](#filter-presets)
- [Workflows](#workflows)

---

## Installation & authentication

```bash
pip install dyno-phi
```

For local biomodal development (deploying Modal GPU apps):

```bash
pip install "dyno-phi[biomodals]"
```

**Set your API key** (obtain from Settings → API keys in the dyno web app):

```bash
export DYNO_API_KEY=ak_...
```

Optionally override the API base URL:

```bash
export DYNO_API_BASE_URL=https://api.dynotx.com
```

Verify your connection:

```bash
phi login
```

---

## Global flags

These flags apply to all commands:

| Flag | Default | Description |
|---|---|---|
| `--poll-interval S` | `5` | Seconds between status-poll requests |
| `--version` | — | Print version and exit |
| `--help` | — | Show help for any command |

---

## State caching

`phi` caches the most recently used **dataset ID** and **job ID** in
`.phi-state.json` in the current directory. This means you don't need to pass
`--dataset-id` or `job_id` repeatedly in a session.

```bash
phi use d7c3a1b2-...         # set active dataset
phi filter --preset default  # uses cached dataset
phi scores                   # uses cached job from last filter/model run
phi download --out ./results # uses cached job
```

The footer line printed after every command shows the current active IDs:

```
Active: dataset [d7c3a1b2-...] · job [cb4553f5-...]
```

---

## Command index

| Command | Aliases | Description |
|---|---|---|
| `phi login` | — | Verify API key and print connection + identity |
| `phi fetch` | — | Download a structure from RCSB PDB or AlphaFold DB, crop, optionally upload |
| `phi upload` | — | Upload PDB/CIF/FASTA files or a directory → create a dataset |
| `phi use` | — | Set the active dataset ID |
| `phi datasets` | — | List your datasets |
| `phi dataset` | — | Show details for a single dataset |
| `phi ingest-session` | — | Check the status of an ingest session |
| `phi design` | `rfdiffusion3` | Backbone generation — binder design, de novo, motif scaffolding |
| `phi boltzgen` | — | All-atom generative design from a YAML spec |
| `phi folding` | `esmfold` | Fast single-sequence structure prediction (ESMFold) |
| `phi complex_folding` | `alphafold` | Monomer or multimer structure prediction (AlphaFold2) |
| `phi inverse_folding` | `proteinmpnn` | Sequence design via inverse folding (ProteinMPNN) |
| `phi esm2` | — | Language model log-likelihood scoring and perplexity |
| `phi boltz` | — | Biomolecular complex prediction — proteins, DNA, RNA (Boltz-1) |
| `phi filter` | — | Full filter pipeline: inverse folding → folding → complex folding → score |
| `phi status` | — | Get the status of a job |
| `phi jobs` | — | List recent jobs |
| `phi logs` | — | Print log stream URL for a job |
| `phi cancel` | — | Cancel a running job |
| `phi scores` | — | Display scoring metrics table for a completed filter job |
| `phi download` | — | Download output files for a completed job |
| `phi research` | — | Run a biological research query with citations |
| `phi notes` | — | View accumulated research notes for a dataset |

---

## Detailed reference

### phi login

Verify your API key and print your identity and connection details.

```
phi login [--json]
```

| Flag | Description |
|---|---|
| `--json` | Print raw JSON response |

**Example:**
```bash
phi login
```

---

### phi fetch

Download a structure from **RCSB PDB** or the **AlphaFold Database**, optionally
crop it, save it locally, and optionally upload it to the dyno cloud.

```
phi fetch (--pdb ID | --uniprot ID) [crop options] [output options]
```

**Source (pick one):**

| Flag | Description |
|---|---|
| `--pdb ID` | RCSB PDB ID (e.g., `4ZQK`) |
| `--uniprot ID` | UniProt accession — downloads from AlphaFold DB (e.g., `Q9NZQ7`) |

**Cropping (optional):**

| Flag | Description |
|---|---|
| `--chain CHAIN` | Extract a single chain (e.g., `A`) |
| `--residues START-END` | Keep only residues in this range (e.g., `56-290`) |
| `--trim-low-confidence PLDDT` | Remove residues with pLDDT below this threshold. AlphaFold DB structures store pLDDT in the B-factor column. Typical value: `70` |

**Output:**

| Flag | Description |
|---|---|
| `--out FILE` | Output PDB path (default: `{ID}[_{chain}].pdb` in current dir) |
| `--upload` | Upload to dyno cloud storage after saving — creates a dataset and prints the GCS URI for use with `phi design --target-pdb-gcs` |
| `--name NAME` | Dataset name label when using `--upload` |

**Examples:**
```bash
# Fetch and crop PDB 4ZQK chain A, residues 56–290
phi fetch --pdb 4ZQK --chain A --residues 56-290 --out target.pdb

# Fetch from AlphaFold DB, trim low-confidence tails, upload to cloud
phi fetch --uniprot Q9NZQ7 --trim-low-confidence 70 --upload

# Fetch and upload with a custom dataset name
phi fetch --pdb 7XKJ --chain B --upload --name pd-l1-target
```

---

### phi upload

Upload PDB, CIF, or FASTA files (or a directory) to the dyno platform and
create a dataset for batch processing.

```
phi upload [FILE ...] [--dir DIR] [options]
```

| Flag | Description |
|---|---|
| `FILE ...` | One or more files to upload (positional) |
| `--dir DIR` | Upload all matching files in this directory |
| `--file-type TYPE` | Override auto-detected file type: `pdb`, `cif`, `fasta`, `csv`. When omitted, type is inferred from file extensions |
| `--run-id ID` | Label for this ingest session |
| `--wait` | Poll until the dataset is `READY` (default: on) |
| `--no-wait` | Return after finalizing without polling |

**Examples:**
```bash
# Upload a directory of PDB files
phi upload --dir ./designs/ --file-type pdb

# Upload specific files
phi upload binder1.pdb binder2.pdb binder3.pdb

# Upload and return immediately (poll with phi ingest-session)
phi upload --dir ./designs/ --no-wait
```

---

### phi use

Set the active dataset ID. Saved to `.phi-state.json` and used as the default
`--dataset-id` for subsequent commands.

```
phi use DATASET_ID
```

**Example:**
```bash
phi use d7c3a1b2-4f3e-11ef-9ab7-0242ac120002
```

---

### phi datasets

List your datasets.

```
phi datasets [--limit N] [--json]
```

| Flag | Default | Description |
|---|---|---|
| `--limit N` | `20` | Number of datasets to show |
| `--json` | — | Print raw JSON |

---

### phi dataset

Show details for a single dataset, including artifact count and sample files.

```
phi dataset DATASET_ID [--json]
```

| Flag | Description |
|---|---|
| `--json` | Print raw JSON |

---

### phi ingest-session

Check the status of a background ingest session (useful after `phi upload --no-wait`).

```
phi ingest-session SESSION_ID [--json]
```

| Flag | Description |
|---|---|
| `--json` | Print raw JSON |

---

### phi design / rfdiffusion3

Generate protein backbones using **RFdiffusion3**. Supports binder design
(targeting a receptor), de novo backbone generation, and motif scaffolding.
Runtime: ~2–5 min per design.

```
phi design [mode options] [binder options] [generation options] [job options]
```

**Design mode (pick one):**

| Flag | Description |
|---|---|
| `--target-pdb FILE` | Target PDB for binder design |
| `--target-pdb-gcs URI` | Cloud storage URI to target PDB (`gs://…`) |
| `--length N` | Backbone length for de novo generation (no target) |
| `--motif-pdb FILE` | Motif PDB for scaffolding |
| `--motif-pdb-gcs URI` | Cloud storage URI to motif PDB (`gs://…`) |

**Binder design options:**

| Flag | Description |
|---|---|
| `--target-chain CHAIN` | Target chain ID (e.g., `A`) |
| `--hotspots A45,A67` | Comma-separated hotspot residues for interface design |
| `--motif-residues 10-20,45-55` | Comma-separated motif residue ranges |

**Generation parameters:**

| Flag | Default | Description |
|---|---|---|
| `--num-designs N` | `10` | Number of backbone designs to generate |
| `--steps N` | `50` | Diffusion inference steps — higher improves quality |
| `--contigs STR` | — | Contig specification string for advanced control |
| `--symmetry C3` | — | Symmetry specification (e.g., `C3`, `D2`, `C5`) |

**Job options** (shared with all model commands):

| Flag | Default | Description |
|---|---|---|
| `--run-id ID` | — | Optional run label |
| `--wait` | on | Poll until job completes |
| `--no-wait` | — | Return immediately after submission |
| `--out DIR` | — | Download results to this directory when done |
| `--json` | — | Output raw JSON |

**Examples:**
```bash
# Binder design targeting a receptor
phi design --target-pdb target.pdb --hotspots A45,A67 --num-designs 50

# With uploaded target (GCS URI from phi fetch --upload)
phi design --target-pdb-gcs gs://bucket/target.pdb --hotspots A45,A67 --num-designs 100

# De novo backbone generation
phi design --length 80 --num-designs 20

# Motif scaffolding
phi design --motif-pdb motif.pdb --motif-residues 10-20,45-55 --num-designs 30

# Symmetric design
phi design --length 120 --symmetry C3 --num-designs 10
```

---

### phi boltzgen

All-atom generative binder design using **BoltzGen**. Takes a YAML design
specification and runs diffusion + inverse folding. Supports proteins, peptides,
antibodies, nanobodies, and small molecule binders.
Runtime: ~10–20 min.

```
phi boltzgen (--yaml FILE | --yaml-gcs URI) [options]
```

**Input (pick one):**

| Flag | Description |
|---|---|
| `--yaml FILE` | Local YAML design specification file |
| `--yaml-gcs URI` | Cloud storage URI to YAML file (`gs://…`) |
| `--structure-gcs URI` | Cloud storage URI to a structure file referenced in the YAML |

**Generation parameters:**

| Flag | Default | Description |
|---|---|---|
| `--protocol PROTOCOL` | `protein-anything` | Design protocol. Choices: `protein-anything`, `peptide-anything`, `protein-small_molecule`, `antibody-anything`, `nanobody-anything`, `protein-redesign` |
| `--num-designs N` | `10` | Intermediate designs to generate. Use `10,000–60,000` for production campaigns |
| `--budget N` | `num_designs // 10` | Final diversity-optimized design count |
| `--boltzgen-steps STEPS` | — | Specific pipeline steps, space-separated (e.g., `design inverse_folding folding`). Omit to run full pipeline |

**Inverse folding only:**

| Flag | Description |
|---|---|
| `--only-inverse-fold` | Run inverse folding on an existing structure YAML — skips backbone design |
| `--inverse-fold-num-sequences N` | Sequences per design when using `--only-inverse-fold` (default: `2`) |

**Examples:**
```bash
# Full protein binder design pipeline
phi boltzgen --yaml design.yaml --protocol protein-anything --num-designs 10

# Peptide binder design
phi boltzgen --yaml peptide.yaml --protocol peptide-anything --num-designs 50

# Production-scale campaign
phi boltzgen --yaml binder.yaml --num-designs 20000 --budget 200

# Run only inverse folding on existing designs
phi boltzgen --yaml structures.yaml --only-inverse-fold --inverse-fold-num-sequences 4
```

---

### phi folding / esmfold

Fast single-sequence structure prediction using **ESMFold**.
Runtime: ~1 min per sequence.

```
phi folding (--fasta FILE | --fasta-str FASTA | --dataset-id ID) [options]
```

**Input (pick one):**

| Flag | Description |
|---|---|
| `--fasta FILE` | FASTA file to submit |
| `--fasta-str FASTA` | FASTA content as a string (for scripting) |
| `--dataset-id ID` | Pre-ingested dataset ID (batch mode, 100–50,000 sequences) |

**Options:**

| Flag | Default | Description |
|---|---|---|
| `--recycles N` | `3` | Recycling iterations |
| `--no-confidence` | — | Skip per-residue pLDDT extraction |
| `--fasta-name NAME` | — | Name label for output files (single-sequence mode only) |

**Examples:**
```bash
# Single file
phi folding --fasta sequences.fasta

# Batch (dataset)
phi folding --dataset-id d7c3a1b2-... --wait --out ./results/

# Inline sequence
phi folding --fasta-str ">binder1\nMKTAYIAKQRQISFVKS..."
```

---

### phi complex_folding / alphafold

Monomer or multimer structure prediction using **AlphaFold2** (ColabFold
pipeline). Automatically detects multimer mode from `:` separators in the FASTA.
Runtime: ~8–15 min.

```
phi complex_folding (--fasta FILE | --fasta-str FASTA | --dataset-id ID) [options]
```

**Input (pick one):**

| Flag | Description |
|---|---|
| `--fasta FILE` | FASTA file — use `:` as chain separator for multimer (e.g., `>binder:target`) |
| `--fasta-str FASTA` | FASTA content as a string |
| `--dataset-id ID` | Pre-ingested dataset ID (batch mode) |

**Options:**

| Flag | Default | Description |
|---|---|---|
| `--models 1,2,3` | `1,2,3` | Model numbers to run |
| `--model-type TYPE` | `auto` | `auto` (picks `ptm` for monomers, `multimer_v3` for complexes), `ptm`, `multimer_v1`, `multimer_v2`, `multimer_v3` |
| `--msa-tool TOOL` | `mmseqs2` | MSA algorithm: `mmseqs2` or `jackhmmer` |
| `--msa-databases DB` | `uniref_env` | Database set for MSA: `uniref_env` or `uniref_only` |
| `--template-mode MODE` | `none` | Template lookup: `none` or `pdb70` |
| `--pair-mode MODE` | `unpaired_paired` | MSA pairing for complexes: `unpaired_paired`, `paired`, `unpaired` |
| `--recycles N` | `6` | Recycling iterations |
| `--num-seeds N` | `3` | Number of model seeds |
| `--amber` | — | Run AMBER force-field relaxation to remove stereochemical violations |

**Examples:**
```bash
# Complex prediction (binder + target, colon-separated FASTA)
phi complex_folding --fasta binder_target.fasta

# Monomer with AMBER relaxation
phi complex_folding --fasta monomer.fasta --amber

# Batch over a dataset, download on completion
phi complex_folding --dataset-id d7c3a1b2-... --wait --out ./af2_results/
```

---

### phi inverse_folding / proteinmpnn

Design sequences for a protein backbone using **ProteinMPNN**.
Runtime: ~1–2 min.

```
phi inverse_folding (--pdb FILE | --pdb-gcs URI | --dataset-id ID) [options]
```

**Input (pick one):**

| Flag | Description |
|---|---|
| `--pdb FILE` | PDB structure file |
| `--pdb-gcs URI` | Cloud storage URI to PDB (`gs://…`) |
| `--dataset-id ID` | Pre-ingested dataset ID (batch mode) |

**Options:**

| Flag | Default | Description |
|---|---|---|
| `--num-sequences N` | `10` | Sequences to design |
| `--temperature T` | `0.1` | Sampling temperature 0–1. Lower values are more conservative |
| `--fixed A52,A56` | — | Comma-separated residue positions to fix (not redesign) |

**Examples:**
```bash
# Design 20 sequences for a backbone
phi inverse_folding --pdb design.pdb --num-sequences 20

# Fix key interface residues
phi inverse_folding --pdb binder.pdb --num-sequences 10 --fixed A52,A56,A63

# Higher diversity sampling
phi inverse_folding --pdb design.pdb --num-sequences 50 --temperature 0.3

# Batch
phi inverse_folding --dataset-id d7c3a1b2-... --num-sequences 4 --wait
```

---

### phi esm2

Language model scoring with **ESM2** — computes pseudo-log-likelihood (PLL)
scores and perplexity. Useful for filtering designs by sequence plausibility.

```
phi esm2 (--fasta FILE | --fasta-str FASTA | --dataset-id ID) [options]
```

**Input (pick one):**

| Flag | Description |
|---|---|
| `--fasta FILE` | FASTA file |
| `--fasta-str FASTA` | FASTA content as a string |
| `--dataset-id ID` | Pre-ingested dataset ID (batch mode) |

**Options:**

| Flag | Description |
|---|---|
| `--mask 5,10,15` | Comma-separated positions to mask for scoring |

**Example:**
```bash
phi esm2 --fasta designed_sequences.fasta --wait
```

---

### phi boltz

Biomolecular complex structure prediction using **Boltz-1** (open-source).
Supports proteins, DNA, and RNA. Good alternative to AlphaFold2.

```
phi boltz (--fasta FILE | --fasta-str FASTA | --dataset-id ID) [options]
```

**Input (pick one):**

| Flag | Description |
|---|---|
| `--fasta FILE` | FASTA file |
| `--fasta-str FASTA` | FASTA content as a string |
| `--dataset-id ID` | Pre-ingested dataset ID (batch mode) |

**Options:**

| Flag | Default | Description |
|---|---|---|
| `--recycles N` | `3` | Recycling iterations |
| `--no-msa` | — | Disable MSA for faster, lower-accuracy prediction |

**Example:**
```bash
phi boltz --fasta complex.fasta --wait
```

---

### phi filter

Run the full **binder design validation pipeline** on a dataset:

1. **ProteinMPNN** — inverse folding to generate candidate sequences
2. **ESMFold** — fast structure prediction of each candidate
3. **AlphaFold2** — complex structure prediction (binder + target)
4. **Scoring** — compute pLDDT, pTM, ipTM, iPAE, RMSD; apply thresholds

```
phi filter [--dataset-id ID] [--preset NAME] [threshold flags] [options]
```

**Input:**

| Flag | Description |
|---|---|
| `--dataset-id ID` | Dataset of PDB/CIF designs (default: cached from last upload or `phi use`) |

**Preset:**

| Flag | Description |
|---|---|
| `--preset default\|relaxed` | Apply a named threshold preset. Individual flags override preset values |

**Threshold overrides** (all optional — defaults come from the preset):

| Flag | Preset default | Description |
|---|---|---|
| `--plddt-threshold F` | `0.80` | ESMFold binder pLDDT lower bound |
| `--ptm-threshold F` | `0.55` | AlphaFold2 complex pTM lower bound |
| `--iptm-threshold F` | `0.50` | AlphaFold2 interface pTM lower bound |
| `--ipae-threshold F` | `10.85` Å | AlphaFold2 interface PAE upper bound |
| `--rmsd-threshold F` | `3.5` Å | Binder backbone RMSD upper bound |

**Pipeline options:**

| Flag | Default | Description |
|---|---|---|
| `--num-sequences N` | `4` | ProteinMPNN sequences per design |
| `--num-recycles N` | `3` | AlphaFold2 recycling iterations |
| `--msa-tool TOOL` | `single_sequence` | MSA algorithm for AF2: `single_sequence` (recommended for de novo binders), `mmseqs2`, `jackhmmer` |

**Job options:**

| Flag | Description |
|---|---|
| `--run-id ID` | Optional custom run ID |
| `--wait` | Poll until pipeline completes |
| `--out DIR` | Download results when done |
| `--all` | When `--out` is set, download all artifact types including MSA files and archives |

**Examples:**
```bash
# Default pipeline (uses cached dataset)
phi filter --preset default --wait

# Custom thresholds
phi filter --dataset-id d7c3a1b2-... --plddt-threshold 0.75 --iptm-threshold 0.45 --wait

# Relaxed preset with download
phi filter --preset relaxed --wait --out ./results/

# Run and download everything
phi filter --preset default --wait --out ./results/ --all
```

---

### phi status

Get the status of a job.

```
phi status JOB_ID [--json]
```

| Flag | Description |
|---|---|
| `--json` | Print raw JSON |

---

### phi jobs

List recent jobs.

```
phi jobs [--limit N] [--status STATUS] [--job-type TYPE] [--json]
```

| Flag | Default | Description |
|---|---|---|
| `--limit N` | `20` | Number of jobs to show |
| `--status STATUS` | — | Filter by status: `pending`, `running`, `completed`, `failed`, `cancelled` |
| `--job-type TYPE` | — | Filter by job type (e.g., `esmfold`, `design_pipeline`) |
| `--json` | — | Print raw JSON |

**Examples:**
```bash
phi jobs
phi jobs --status running
phi jobs --limit 50 --job-type design_pipeline
```

---

### phi logs

Print the log stream URL for a job. Useful for monitoring long-running jobs.

```
phi logs JOB_ID [--follow]
```

| Flag | Description |
|---|---|
| `--follow` | Stream logs continuously |

---

### phi cancel

Cancel a running job.

```
phi cancel JOB_ID
```

---

### phi scores

Display the scoring metrics table for a completed filter job.

```
phi scores [JOB_ID] [--top N] [--out FILE] [--json]
```

| Flag | Default | Description |
|---|---|---|
| `JOB_ID` | cached | Job ID (default: last cached job) |
| `--top N` | `20` | Show top-N candidates ranked by score |
| `--out FILE` | — | Save scores CSV to file |
| `--json` | — | Output raw JSON |

**Examples:**
```bash
# Show scores for the last filter job
phi scores

# Show top 50 and save CSV
phi scores --top 50 --out scores.csv
```

---

### phi download

Download all output files for a completed job — structures, scores CSV, and
raw score JSONs.

```
phi download [JOB_ID] [--out DIR] [--all]
```

| Flag | Default | Description |
|---|---|---|
| `JOB_ID` | cached | Job ID (default: last cached job) |
| `--out DIR` | `./results` | Output directory |
| `--all` | — | Download all artifact types including MSA files, zip archives, and scripts |

**Examples:**
```bash
phi download --out ./results/

# Download everything including MSA files
phi download --out ./results/ --all

# Download a specific job
phi download cb4553f5-... --out ./run-42/
```

---

### phi research

Run a biological research query against the platform. Searches PubMed, UniProt,
and PDB, then synthesizes a report with citations.
Runtime: ~2–5 min.

```
phi research --question QUESTION [options]
```

| Flag | Default | Description |
|---|---|---|
| `--question QUESTION` | (required) | Research question (e.g., `"What are known binding hotspots for PD-L1?"`) |
| `--target TARGET` | — | Protein or gene name to focus the search (e.g., `PD-L1`, `KRAS`) |
| `--databases LIST` | `pubmed,uniprot,pdb` | Comma-separated databases to query |
| `--max-papers N` | `20` | Maximum papers to retrieve from PubMed |
| `--structures` | — | Include related PDB structures in the report |
| `--context TEXT` | — | Additional context for the research query |
| `--context-file FILE` | — | Path to a prior `research.md` file — prepended as context |
| `--dataset-id ID` | — | Associate notes with a dataset and sync to cloud storage |
| `--notes-file FILE` | `./research.md` | Local append-only notes file |
| `--no-save` | — | Skip saving the report to the local notes file |
| `--stream` | — | Stream results live via SSE (skips job tracking) |

**Examples:**
```bash
phi research --question "What are the known binding hotspots for PD-L1?"

phi research \
  --question "What is the structure and function of EGFR domain III?" \
  --target EGFR \
  --structures \
  --dataset-id d7c3a1b2-...

# Build on a prior research session
phi research \
  --question "Which of these hotspots are most druggable?" \
  --context-file ./research.md
```

---

### phi notes

View the accumulated research notes for a dataset.

```
phi notes DATASET_ID [--out PATH] [--json]
```

| Flag | Description |
|---|---|
| `--out PATH` | Save notes to a `.md` file or directory (saves as `research.md`) instead of printing |
| `--json` | Output raw JSON |

**Example:**
```bash
phi notes d7c3a1b2-... --out ./campaign-notes.md
```

---

## Filter presets

`phi filter --preset` applies a named set of quality-control thresholds across
the full validation pipeline.

| Metric | `default` | `relaxed` | Description |
|---|---|---|---|
| pLDDT | ≥ 0.80 | ≥ 0.80 | ESMFold per-residue confidence (0–1) |
| pTM | ≥ 0.55 | ≥ 0.45 | Global TM-score proxy from ESMFold |
| ipTM | ≥ 0.50 | ≥ 0.50 | Interface pTM from AF2 multimer (0–1) |
| iPAE | ≤ 10.85 Å | ≤ 12.4 Å | AF2 interface predicted aligned error |
| RMSD | ≤ 3.5 Å | ≤ 4.5 Å | Backbone RMSD vs. reference design |

Override any individual threshold alongside a preset:

```bash
phi filter --preset default --plddt-threshold 0.75 --iptm-threshold 0.45
```

The `single_sequence` MSA mode (default for `phi filter`) is recommended for
de novo designed binders — they have no natural homologs, so MSA adds noise
rather than signal and results in better-calibrated confidence scores.

---

## Workflows

### Binder design — full pipeline

```bash
# 1. Fetch and prepare target
phi fetch --pdb 4ZQK --chain A --residues 56-290 --out target.pdb

# 2. Generate backbones
phi design --target-pdb target.pdb --hotspots A45,A67 --num-designs 50

# 3. Upload backbones for batch validation
phi upload --dir ./rfdiffusion_outputs/ --file-type pdb

# 4. Run full filter pipeline
phi filter --preset default --wait --out ./results/

# 5. Review scores
phi scores --top 30
```

### BoltzGen binder design

```bash
# 1. Fetch target and upload to get GCS URI
phi fetch --uniprot Q9NZQ7 --trim-low-confidence 70 --upload

# 2. Create YAML spec referencing the GCS URI, then run
phi boltzgen --yaml design.yaml --protocol protein-anything --num-designs 10000

# 3. Download top designs
phi download --out ./boltzgen_results/
```

### Validate a batch of existing sequences

```bash
# Upload FASTA sequences
phi upload sequences.fasta

# Structure prediction
phi folding --dataset-id d7c3a1b2-... --wait

# Score with ESM2
phi esm2 --dataset-id d7c3a1b2-... --wait

# Download results
phi download --out ./validation/
```

### Research-guided campaign

```bash
# Research the target
phi research \
  --question "What are the binding hotspots of PD-L1 for therapeutic binders?" \
  --target PD-L1 \
  --structures \
  --dataset-id d7c3a1b2-...

# View notes later
phi notes d7c3a1b2-...
```
