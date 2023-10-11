#!/usr/bin/env bash
src="pagerank-levelwise-multi-dynamic"
out="$HOME/Logs/$src$1.log"
ulimit -s unlimited
printf "" > "$out"

# Download program
if [[ "$DOWNLOAD" != "0" ]]; then
  rm -rf $src
  git clone https://github.com/puzzlef/$src
  cd $src
fi

# Run
nvcc -std=c++17 -Xcompiler -fopenmp -O3 main.cu
stdbuf --output=L ./a.out ~/Data/arabic-2005.mtx       2>&1 | tee -a "$out"
stdbuf --output=L ./a.out ~/Data/uk-2005.mtx           2>&1 | tee -a "$out"
stdbuf --output=L ./a.out ~/Data/it-2004.mtx           2>&1 | tee -a "$out"
stdbuf --output=L ./a.out ~/Data/soc-Epinions1.mtx     2>&1 | tee -a "$out"
stdbuf --output=L ./a.out ~/Data/soc-LiveJournal1.mtx  2>&1 | tee -a "$out"
stdbuf --output=L ./a.out ~/Data/wiki-Talk.mtx         2>&1 | tee -a "$out"
stdbuf --output=L ./a.out ~/Data/cit-Patents.mtx       2>&1 | tee -a "$out"
stdbuf --output=L ./a.out ~/Data/coPapersDBLP.mtx      2>&1 | tee -a "$out"
stdbuf --output=L ./a.out ~/Data/amazon-2008.mtx       2>&1 | tee -a "$out"
stdbuf --output=L ./a.out ~/Data/italy_osm.mtx         2>&1 | tee -a "$out"
stdbuf --output=L ./a.out ~/Data/Linux_call_graph.mtx  2>&1 | tee -a "$out"

# Signal completion
curl -X POST "https://maker.ifttt.com/trigger/puzzlef/with/key/${IFTTT_KEY}?value1=$src$1"
