#!/bin/bash
# Ingest a local folder of documents into a remote server.
# Usage: bash ingest_remote.sh <folder> <server-url>
# Example: bash ingest_remote.sh ./docs https://abc123.trycloudflare.com

set -euo pipefail

FOLDER="${1:-}"
SERVER="${2:-}"

if [ -z "$FOLDER" ] || [ -z "$SERVER" ]; then
    echo "Usage: bash ingest_remote.sh <folder> <server-url>"
    echo "Example: bash ingest_remote.sh ./docs https://abc123.trycloudflare.com"
    exit 1
fi

SERVER="${SERVER%/}"  # strip trailing slash

FILES=("$FOLDER"/*.pdf "$FOLDER"/*.txt)
TOTAL=0
SUCCESS=0
FAILED=0

for f in "${FILES[@]}"; do
    [ -f "$f" ] || continue
    TOTAL=$((TOTAL + 1))
    NAME=$(basename "$f")
    printf "Ingesting %-50s ... " "$NAME"

    RESPONSE=$(curl -s -w "\n%{http_code}" \
        -X POST "$SERVER/ingest" \
        -F "file=@$f")

    HTTP_CODE=$(echo "$RESPONSE" | tail -1)
    BODY=$(echo "$RESPONSE" | head -1)

    if [ "$HTTP_CODE" = "200" ]; then
        CHUNKS=$(echo "$BODY" | grep -o '"chunks":[0-9]*' | grep -o '[0-9]*')
        echo "OK (${CHUNKS} chunks)"
        SUCCESS=$((SUCCESS + 1))
    else
        echo "FAILED (HTTP $HTTP_CODE: $BODY)"
        FAILED=$((FAILED + 1))
    fi
done

echo ""
echo "Done: $SUCCESS/$TOTAL ingested, $FAILED failed."
