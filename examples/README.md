# phi CLI examples

Quick-start data for testing the phi upload, scoring, and download pipeline.

---

## binders/

Five binder CIF files (`binder_XXX.cif`).
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

PD-L1 target structures.

| File | Description |
|---|---|
| `5O45.pdb` | PD-L1 crystal structure (PDB: 5O45) |
| `pdl1_pdb_19_132.pdb` | PD-L1 chain A, residues 19–132 (binding domain) |

### Fetch and prepare from PDB directly

```bash
# Fetch PD-L1 chain A from RCSB and crop to residues 19–132
phi fetch --pdb 5O45 --chain A --start 19 --end 132 --upload
```

---

## Typical research → upload → filter workflow

```bash
# 1. Research the target
phi research \
  --question "What are the key binding hotspots on PD-L1 for antibody design?" \
  --target PD-L1 \
  --notes-file ./research.md

# 2. Fetch and prepare the target structure
phi fetch --pdb 5O45 --chain A --start 19 --end 132 --upload

# 3. Upload structures for scoring
phi upload ./structures/

# 4. Score and filter
phi filter --preset relaxed --msa-tool single_sequence --wait --out ./results

# 5. Inspect results
phi scores
phi download
```
