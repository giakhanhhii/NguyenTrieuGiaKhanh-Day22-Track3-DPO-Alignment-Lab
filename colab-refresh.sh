#!/usr/bin/env bash
# Pull the latest lab code, then refresh only the pieces that changed.

set -euo pipefail

python scripts/colab_resume.py --pull "$@"
