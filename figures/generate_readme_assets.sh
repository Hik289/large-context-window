#!/usr/bin/env bash
#
# generate_readme_assets.sh
#
# README figures are stored under ../assets. The default artifact already ships
# with assets/pipeline.png. Regenerate or replace it with your preferred local
# image workflow, then re-run README visual checks before release.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ASSET="$ROOT/assets/pipeline.png"

if [ -f "$ASSET" ]; then
  echo "Found README pipeline asset: $ASSET"
else
  echo "Missing README pipeline asset: $ASSET"
  echo "Create assets/pipeline.png before releasing the repository."
  exit 1
fi
