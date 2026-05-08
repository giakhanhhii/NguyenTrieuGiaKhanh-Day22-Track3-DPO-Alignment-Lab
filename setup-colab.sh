#!/usr/bin/env bash
# Idempotent Colab setup. Safe to rerun after a partial install or after you
# pull new code from GitHub.

set -euo pipefail

python scripts/colab_resume.py "$@"
