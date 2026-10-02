# References

- `papers.bib` — BibTeX bibliography
- `claims.yaml` — claim ID → BibTeX key mapping (drives report citations)
- `notes/` — per-paper markdown summaries (one file per BibTeX key)
- `expert/` — curated expert documents (PDFs, lab notes, expert review docs) registered via `workspace.yaml.expert_docs`. Add via the dashboard "Expert knowledge" panel or `POST /api/expert-doc`.
- `basement-membrane-v1-spec.md` — the AICS-Lumenoids v1 spec (Matt Akamatsu; Roam export, currently **2026-09-30**). Updated when Matt sends a new export.

## Upstream source code

The authors' LAMMPS input scripts for the Meadowcroft/Barrientos model are now
public: [github.com/Billie1717/BasementMembraneTurnoverSims](https://github.com/Billie1717/BasementMembraneTurnoverSims),
archived at Zenodo [10.5281/zenodo.20719515](https://doi.org/10.5281/zenodo.20719515)
(v1.0.0, CC-BY 4.0; cloned at commit `ab4c004`). Stage folders (Assembly,
Equilibration, BondSampling, Stretch, Growth) each ship a `build_*.py` that
writes that stage's `collagen.in`. The `viva_lumenoid` build is still a
**clean-room transcription from the spec**, not a port of these scripts, but the
scripts are now the ground truth to cross-check its parameters against
(bibliography key `barrientos2026code`).
