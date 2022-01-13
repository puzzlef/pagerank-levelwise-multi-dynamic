#!/usr/bin/env bash
src="kitchen-sink"
out="/home/resources/Documents/subhajit/$src.log"
ulimit -s unlimited
printf "" > "$out"

# Download program
rm -rf $src
git clone https://github.com/puzzlef/$src
cd $src

# Run
nvcc -std=c++17 -Xcompiler -fopenmp -lnvgraph -O3 main.cu
stdbuf --output=L ./a.out ~/data/webbase-2001.mtx    2>&1 | tee -a "$out"
stdbuf --output=L ./a.out ~/data/indochina-2004.mtx  2>&1 | tee -a "$out"
stdbuf --output=L ./a.out ~/data/arabic-2005.mtx     2>&1 | tee -a "$out"
stdbuf --output=L ./a.out ~/data/GAP-twitter.mtx     2>&1 | tee -a "$out"
stdbuf --output=L ./a.out ~/data/GAP-road.mtx        2>&1 | tee -a "$out"
