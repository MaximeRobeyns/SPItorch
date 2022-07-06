#!/usr/bin/env bash
set -euo pipefail

sphinx-autobuild docs/source docs/build/html --open-browser --watch ../spt
