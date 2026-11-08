#!/usr/bin/env bash
# Build the website for GitHub Pages and publish it on the gh-pages branch.
#   bash web/deploy-pages.sh            (from the repo root, in Git Bash)
# The site is then at https://nardy11.github.io/tryon-studio/ (about a minute later).
set -euo pipefail
cd "$(dirname "$0")"
# No leading slash: Git Bash would turn "/tryon-studio/" into a Windows path.
MSYS_NO_PATHCONV=1 VITE_BASE=tryon-studio TRYON_HTTP=1 pnpm build
# Refuse to publish a build that points anywhere but /tryon-studio/.
grep -q 'src="/tryon-studio/assets/' dist/index.html || { echo "Wrong base path in dist/index.html"; exit 1; }
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
