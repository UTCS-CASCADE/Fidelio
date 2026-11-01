#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REV=3bfe2c7c07069260786752fdd43cc355eb33187b
if [[ "${1:-}" == --help ]]; then
  echo "Usage: ./bootstrap_folly_edit_rag.sh"
  echo "Registers or initializes the folly_edit_rag Git submodule."
  exit 0
fi
cd "${REPO_ROOT}"
if [[ "$(git rev-parse --show-toplevel 2>/dev/null || true)" != "${REPO_ROOT}" ]]; then
  echo "Fidelio must be a Git repository with writable metadata. Run git init first." >&2
  exit 1
fi
if git ls-files --stage -- folly_edit_rag | grep -q '^160000 '; then
  # Existing repositories use their recorded gitlink revision.
  git submodule update --init --recursive -- folly_edit_rag
else
  git submodule add https://github.com/Alan-S-Andrade/folly_edit_rag.git folly_edit_rag
  git -C folly_edit_rag checkout --detach "${REV}"
  git -C folly_edit_rag submodule update --init --recursive
  git add .gitmodules folly_edit_rag
fi
test -f folly_edit_rag/scripts/local_patch_utils.py
echo "folly_edit_rag submodule ready."
