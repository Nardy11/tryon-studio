#!/usr/bin/env bash
# Build the website for GitHub Pages and publish it on the gh-pages branch.
#   bash web/deploy-pages.sh            (from the repo root, in Git Bash)
# The site is then at https://nardy11.github.io/tryon-studio/ (about a minute later).
set -euo pipefail
cd "$(dirname "$0")"
VITE_BASE=/tryon-studio/ TRYON_HTTP=1 pnpm build
remote="$(git remote get-url origin)"
tmp="$(mktemp -d)"
cp -r dist/. "$tmp"
touch "$tmp/.nojekyll"   # serve files as they are
cd "$tmp"
git init -q -b gh-pages
git add -A
git commit -q -m "Publish website $(date -u +%Y-%m-%dT%H:%MZ)"
git push -q -f "$remote" gh-pages
echo "Published. https://nardy11.github.io/tryon-studio/"
