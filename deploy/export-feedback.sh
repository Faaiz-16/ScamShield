#!/usr/bin/env bash
# Export new user ratings from the server's database as labelled rows for review.
#
#   bash deploy/export-feedback.sh                  every new rating
#   bash deploy/export-feedback.sh --min-votes 2    only messages rated by 2+ users
#
# Runs ml.export_feedback inside the api container and saves the result as
# ~/feedback-YYYY-MM-DD.csv. Download it with DOWNLOAD FILE in the SSH window.
# Messages already saved in data/labels/ (as of the last update) are skipped.
set -euo pipefail
cd "$(dirname "$0")/.."

name="feedback-$(date +%F).csv"
docker compose exec -T api python -m ml.export_feedback --output "/tmp/$name" "$@"

if docker compose exec -T api test -f "/tmp/$name"; then
    docker compose cp "api:/tmp/$name" "$HOME/$name"
    docker compose exec -T api rm -f "/tmp/$name"
    echo
    echo "Saved. To download it, click DOWNLOAD FILE and enter: $HOME/$name"
fi
