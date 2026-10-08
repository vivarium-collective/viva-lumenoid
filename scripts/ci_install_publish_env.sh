#!/usr/bin/env bash
# Resilient install of the read-only-dashboard publish environment.
#
# WHY THIS EXISTS: the ecosystem rebrand (pbg-* → viva-*) left some published
# packages requesting deps under their OLD names — notably `vivarium-workbench`
# and `viva-superpowers` both require `pbg-basic-processes`, whose repo now
# publishes metadata name `viva-basic-processes`. A normal `uv sync` aborts on
# that name mismatch ("Package metadata name `viva-basic-processes` does not
# match given name `pbg-basic-processes`"), which is why the template's
# `uv sync --extra dev` publish step fails.
#
# We sidestep the broken transitive resolution: install every leaf dependency
# by its CORRECT name from PyPI, then install the viva-* siblings + the workbench
# + this workspace with `--no-deps` (so their stale pbg-* requirements are never
# resolved). Only the static-bundle publish path is needed — LAMMPS/Smoldyn are
# not required to render the SPA (build_core swallows their import errors).
#
# Usage: scripts/ci_install_publish_env.sh   (expects an active venv / uv env)
set -euo pipefail

ORG="https://github.com/vivarium-collective"
gitdep() { echo "$1 @ git+${ORG}/$2.git@main"; }

# 1) Leaf runtime deps — all on PyPI, resolved normally.
#    filelock + ruamel.yaml are required by the dashboard SERVE path
#    (viva_superpowers.workspace_catalog imports filelock unconditionally, which
#    vivarium_workbench.api.app pulls in at import) — the reports-publish workflow
#    serves the dashboard headlessly, unlike the static-snapshot dashboard build.
uv pip install \
  "process-bigraph>=1.8.4" "bigraph-schema>=1.4.3" "bigraph-viz>=2.0.3" \
  fastapi "uvicorn>=0.29" "pydantic>=2" jinja2 "pyyaml>=6.0" numpy \
  "jsonschema[format-nongpl]>=4.21" "pypdf>=4.0" "boto3>=1.34" \
  "xarray>=2024.0" "zarr>=2.17" pyarrow polars plotly \
  filelock "ruamel.yaml>=0.18"

# 2) viva-* siblings + workbench, from git@main, --no-deps (correct names, so
#    the pbg-* name mismatch never triggers). Order does not matter with --no-deps.
uv pip install --no-deps \
  "$(gitdep viva-superpowers        viva-superpowers)" \
  "$(gitdep viva-basic-processes    viva-basic-processes)" \
  "$(gitdep viva-catalog            viva-catalog)" \
  "$(gitdep viva-workspace          viva-workspace)" \
  "$(gitdep viva-emitters           viva-emitters)" \
  "$(gitdep investigation-contracts investigation-contracts)" \
  "$(gitdep vivarium-workbench      vivarium-workbench)"

# 3) This workspace's own package (for build_core registration), --no-deps.
uv pip install --no-deps -e .

# Sanity: both publish paths must import — the static-bundle CLI AND the FastAPI
# app (the SERVE path the reports-publish workflow drives headlessly).
python -c "import vivarium_workbench, investigation_contracts, viva_superpowers" \
  && echo "publish env OK: vivarium-workbench-publish ready"
python -c "from vivarium_workbench.api.app import app" \
  && echo "serve env OK: dashboard app importable (reports-publish path)"
