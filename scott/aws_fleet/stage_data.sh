#!/usr/bin/env bash
# Stage everything the workers need into S3 (run once locally; re-run after code/substrate
# changes). Uploads: a code tarball of the current working tree, the substrate files, and
# config.env + bootstrap.sh (which the workers fetch at boot).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${FLEET_CONFIG:-$HERE/config.env}"
REPO_ROOT="$(cd "$HERE/../.." && pwd)"
cd "$REPO_ROOT"

echo "Staging to $S3_URI (region $AWS_REGION)"

# 1. code tarball — current working tree (tracked + untracked), respecting .gitignore so
#    connectomes/, outputs/, .venv/, data/ are excluded. Captures uncommitted changes.
TARBALL="/tmp/pathint_code.tar.gz"
git ls-files -co --exclude-standard > /tmp/pathint_filelist.txt
if [ -n "${STAGE_EXCLUDE_REGEX:-}" ]; then
  grep -vE "$STAGE_EXCLUDE_REGEX" /tmp/pathint_filelist.txt > /tmp/pathint_filelist.staged.txt || true
  mv /tmp/pathint_filelist.staged.txt /tmp/pathint_filelist.txt
fi
tar -czf "$TARBALL" -T /tmp/pathint_filelist.txt
echo "  code.tar.gz  ($(du -h "$TARBALL" | cut -f1), $(wc -l < /tmp/pathint_filelist.txt) files)"
aws s3 cp "$TARBALL" "$S3_URI/code.tar.gz" --region "$AWS_REGION" --only-show-errors

# 2. substrate files (small training inputs only)
for f in $SUBSTRATE_FILES; do
  [ -f "$f" ] || { echo "ERROR: substrate file missing: $f (build it first)"; exit 1; }
  echo "  substrate: $f ($(du -h "$f" | cut -f1))"
  aws s3 cp "$f" "$S3_URI/substrates/$f" --region "$AWS_REGION" --only-show-errors
done

# 3. config + bootstrap (workers download these at boot)
aws s3 cp "${FLEET_CONFIG:-$HERE/config.env}" "$S3_URI/config.env"  --region "$AWS_REGION" --only-show-errors
aws s3 cp "$HERE/bootstrap.sh" "$S3_URI/bootstrap.sh" --region "$AWS_REGION" --only-show-errors

echo "done. Staged code + substrates + config/bootstrap under $S3_URI/"
