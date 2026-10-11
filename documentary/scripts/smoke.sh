#!/usr/bin/env bash
# Verified smoke test: unit tests, then a full-length end-to-end smoke render gated by automated QC.
set -euo pipefail
cd "$(dirname "$0")/.."
python3 -m unittest discover -s tests -v
REMIX_E2E=1 python3 -m unittest tests.test_pipeline.EndToEndSmoke -v
