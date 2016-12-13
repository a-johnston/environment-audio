#!/usr/bin/env bash

mkdir -p data/{l3,library,tink,thwing}
mv data/inside/TinkCafeteriaIndoorsOpenSpace* data/tink/
mv data/inside/Thwing* data/thwing/
mv data/inside/L3* data/l3/
mv data/inside/Library* data/library/
rm -r data/inside
