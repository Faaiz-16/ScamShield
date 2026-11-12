#!/usr/bin/env bash
# Update a running ScamShield server to the latest code on GitHub.
#
#   bash deploy/update.sh            pull, then rebuild and restart if anything changed
#   bash deploy/update.sh --force    rebuild and restart even if nothing changed
#
# Keeps .env and the database. Stops without changing anything if files on the
# server were edited by hand, so local changes are never overwritten.
set -euo pipefail
cd "$(dirname "$0")/.."

before=$(git rev-parse --short HEAD)
git pull --ff-only
after=$(git rev-parse --short HEAD)

if [ "$before" = "$after" ] && [ "${1:-}" != "--force" ]; then
    echo "Already up to date ($after). Use --force to rebuild anyway."
    exit 0
fi

if [ "$before" != "$after" ]; then
    echo "Updating $before -> $after:"
    git log --oneline "$before..$after"
fi

docker compose up -d --build
docker image prune -f > /dev/null
docker compose ps
