#!/usr/bin/env bash
src="pagerank-multi-monolithic-vs-levelwise"
out="/home/resources/Documents/subhajit/$src.log"
ulimit -s unlimited
printf "" > "$out"

# Download program
rm -rf $src
git clone https://github.com/puzzlef/$src
cd $src

# Run
nvcc -std=c++17 -Xcompiler -fopenmp -lnvgraph -O3 main.cu
stdbuf --output=L ./a.out ~/data/web-Stanford.mtx      2>&1 | tee -a "$out"
stdbuf --output=L ./a.out ~/data/web-Google.mtx        2>&1 | tee -a "$out"
stdbuf --output=L ./a.out ~/data/indochina-2004.mtx    2>&1 | tee -a "$out"
stdbuf --output=L ./a.out ~/data/soc-Epinions1.mtx     2>&1 | tee -a "$out"
stdbuf --output=L ./a.out ~/data/soc-LiveJournal1.mtx  2>&1 | tee -a "$out"
stdbuf --output=L ./a.out ~/data/wiki-Talk.mtx         2>&1 | tee -a "$out"
stdbuf --output=L ./a.out ~/data/cit-Patents.mtx       2>&1 | tee -a "$out"
stdbuf --output=L ./a.out ~/data/coPapersDBLP.mtx      2>&1 | tee -a "$out"
stdbuf --output=L ./a.out ~/data/amazon-2008.mtx       2>&1 | tee -a "$out"
stdbuf --output=L ./a.out ~/data/italy_osm.mtx         2>&1 | tee -a "$out"
stdbuf --output=L ./a.out ~/data/Linux_call_graph.mtx  2>&1 | tee -a "$out"
