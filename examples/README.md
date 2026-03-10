# phi CLI examples

Quick-start data for testing the phi design and scoring pipeline.

---

## binders/

Five de novo binder CIF files (`binder_XXX.cif`).
Use these to test the upload → filter → download workflow end-to-end.

```bash
# Upload and cache the dataset
phi upload examples/binders/pdl1/
# → prints DATASET_ID and caches it

# Run the full scoring pipeline
phi filter --preset default --wait --out ./results

# Inspect results
phi scores
```

---

## fixtures/pdl1/

PD-L1 target structures for testing design jobs.

| File | Description |
|---|---|
| `5O45.pdb` | PD-L1 crystal structure (PDB: 5O45) |
| `pdl1_pdb_19_132.pdb` | PD-L1 chain A, residues 19–132 (binding domain) |

### Upload target and run backbone design

```bash
# Upload the prepared PD-L1 binding domain as a new dataset
phi upload examples/fixtures/pdl1/pdl1_pdb_19_132.pdb

# Design binders against PD-L1 hotspot residues
phi design \
  --dataset-id <DATASET_ID> \
  --hotspots A56,A58,A61,A73 \
  --num-designs 50 \
  --wait
```

### Fetch and prepare from PDB directly

```bash
# Fetch PD-L1 chain A from RCSB and crop to residues 19–132
phi fetch --pdb 5O45 --chain A --start 19 --end 132 --upload
# → creates and caches a dataset ready for phi design
```

---

## Typical research → design → filter workflow

```bash
# 1. Research the target
phi research \
  --question "What are the key binding hotspots on PD-L1 for antibody design?" \
  --target PD-L1 \
  --notes-file ./research.md

# 2. Fetch and prepare the target structure
phi fetch --pdb 5O45 --chain A --start 19 --end 132 --upload

# 3. Design binders
phi design --hotspots A56,A58,A61,A73 --num-designs 100 --wait

# 4. Score and filter
phi filter --preset relaxed --msa-tool single_sequence --wait --out ./results

# 5. Inspect results
phi scores
phi download
```
