#!/usr/bin/env bash

mkdir -p data/{l3,library,tink,thwing,stairs}
mv data/inside/TinkCafeteriaIndoorsOpenSpace* data/tink/
mv data/inside/Thwing* data/thwing/
mv data/inside/L3* data/l3/
mv data/inside/Library* data/library/
mv data/inside/olin* data/stairs
rm -r data/inside
