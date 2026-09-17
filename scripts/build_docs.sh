#!/usr/bin/env bash
# Build the current-method and history PDFs with their shared navigation rules.
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: bash scripts/build_docs.sh [--output-dir DIRECTORY]

Build local_mixing_documentation.pdf and local_mixing_history.pdf.
The default output directory is the repository's docs/ directory.
A relative --output-dir is resolved from the caller's working directory.

Requires pandoc, xelatex, and rsvg-convert on PATH. On Debian/Ubuntu,
rsvg-convert is provided by librsvg2-bin. SVG circuit figures remain vector
graphics, explicit section anchors are preserved, and links between the two
documents point to the companion PDF.
EOF
}

doc_script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
doc_repo_root="$(cd -- "$doc_script_dir/.." && pwd)"
doc_output_dir="$doc_repo_root/docs"

while (($#)); do
  case "$1" in
    -h|--help) usage; exit 0 ;;
    --output-dir)
      if (($# < 2)) || [[ -z "$2" ]]; then
        printf '%s\n' 'build_docs.sh: --output-dir requires a directory' >&2
        exit 2
      fi
      doc_output_dir="$2"
      shift 2
      ;;
    *) printf 'build_docs.sh: unknown argument: %s\n' "$1" >&2; usage >&2; exit 2 ;;
  esac
done

doc_missing=()
for doc_program in pandoc xelatex rsvg-convert; do
  command -v "$doc_program" >/dev/null 2>&1 || doc_missing+=("$doc_program")
done
if ((${#doc_missing[@]})); then
  printf 'build_docs.sh: missing required program(s): %s\n' "${doc_missing[*]}" >&2
  printf '%s\n' 'Install Pandoc, XeLaTeX, and rsvg-convert; see README.md.' >&2
  exit 127
fi

doc_build_dir="$(mktemp -d -t local-mixing-docs.XXXXXX)"
trap 'rm -rf -- "$doc_build_dir"' EXIT
doc_names=(local_mixing_documentation local_mixing_history)

for doc_name in "${doc_names[@]}"; do
  printf 'Building %s.pdf\n' "$doc_name"
  pandoc "$doc_repo_root/docs/$doc_name.md" \
    --from=markdown \
    --resource-path="$doc_repo_root/docs" \
    --lua-filter="$doc_script_dir/pdf_links.lua" \
    --pdf-engine=xelatex \
    --fail-if-warnings \
    --output="$doc_build_dir/$doc_name.pdf"
done

# Keep the existing PDFs until both documents have rendered successfully.
mkdir -p -- "$doc_output_dir"
for doc_name in "${doc_names[@]}"; do
  cp -- "$doc_build_dir/$doc_name.pdf" "$doc_output_dir/$doc_name.pdf"
  printf 'Wrote %s/%s.pdf\n' "$doc_output_dir" "$doc_name"
done
