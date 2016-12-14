#!/usr/bin/env bash
# Requires gdrive to be set up. If you have access to this download, you will be
# prompted for an access token.

rm -rf data
mkdir -p data
gdrive download -r 0B7MwQ1GfoP1rZFpucGZVNXlZMkE
mv 440data/* data/
