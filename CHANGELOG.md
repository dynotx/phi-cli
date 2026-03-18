# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.1] - 2026-03-17

### Added
- `phi tutorial` — downloads example PD-L1 binder structures from the API and prints a step-by-step scoring walkthrough

### Fixed
- GCS internal paths (`gs://...`) no longer shown in status output, research notes, or `phi fetch` results
- `phi design` / `phi boltzgen` feature-flagged behind `DYNO_ENABLE_DESIGN=1` env var
- macOS SSL certificate error when downloading tutorial files via signed URLs

## [0.2.0] - 2026-03-17

### Added
- Dashboard URL (`design.dynotx.com/dashboard/datasets/<dataset_id>`) printed after every command that sets an active dataset
- API key cached to `.phi/state.json` on first use — no need to re-export `DYNO_API_KEY` in subsequent sessions
- `phi filter` dataset ready panel now shows a prominent **Next step: `phi filter`** above detailed job options

### Changed
- Python compatibility widened from 3.11-only to **3.9–3.13**
- Filter jobs now named `job-filter_pipeline-<hash>` (was `job-design_pipeline-<hash>`)
- `phi login` panel no longer displays `endpoint`, `user_id`, or `org_id`
- `DYNO_API_KEY` missing error message now points to `https://design.dynotx.com/dashboard/settings`
- Removed generative model commands (`phi design`, `phi boltzgen`) from user-facing documentation
- Removed Biomodals section from README
- Fixed repository URLs (`dyno-tx` → `dynotx`)

## [0.1.0] - 2024-03-01

### Added
- Initial public release of the `phi` CLI
- `phi upload` — upload PDB/CIF files or a directory as a dataset
- `phi fetch` — fetch a structure from RCSB PDB or AlphaFold DB
- `phi datasets` / `phi dataset` — list and inspect datasets
- `phi use` — set the active dataset (cached to `.phi-state.json`)
- `phi design` (`rfdiffusion3`) — backbone diffusion via RFDiffusion3
- `phi boltzgen` — all-atom binder design via BoltzGen
- `phi folding` (`esmfold`) — single-sequence structure prediction
- `phi complex_folding` (`alphafold`) — multi-chain complex prediction
- `phi inverse_folding` (`proteinmpnn`) — sequence design via inverse folding
- `phi esm2` — ESM2 sequence embeddings and scoring
- `phi boltz` — Boltz-1 structure prediction
- `phi filter` — full quality-control pipeline with configurable thresholds
- `phi status` / `phi jobs` / `phi logs` / `phi cancel` — job management
- `phi scores` — display scored results table
- `phi download` — download job artifacts (structures, scores, raw JSONs)
- `phi research` — research query against the platform
- `phi notes` — manage dataset research notes
- `biomodals/` — Modal GPU apps for AlphaFold2, ESMFold, ProteinMPNN, Boltz,
  BoltzGen, BindCraft, Chai-1, RFdiffusion3, RF3, LigandMPNN, AF2Rank, RSO,
  TM-score, structure alignment
- Claude Code skill (`skills/phi/SKILL.md`)
