#!/bin/bash

set -e
# Historical inputs/outputs belong in the artifact directory, not beside this script.
ARTIFACT_DIR="${1:-${SECURITY_ARTIFACT_DIR:-$PWD}}"
cd -- "$ARTIFACT_DIR"
cat start.txt > circuits1.txt
cat recent_circuit.txt > circuits2.txt
echo "Circuits updated: circuits1.txt and circuits2.txt written successfully."
