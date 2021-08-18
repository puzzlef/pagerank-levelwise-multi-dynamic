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
g++ -std=c++17 -O3 main.cxx
stdbuf --output=L ./a.out data/min1c1l.txt 1 2 2  2>&1 | tee -a "$out"
stdbuf --output=L ./a.out data/min2c1l.txt 1 5 2  2>&1 | tee -a "$out"
stdbuf --output=L ./a.out data/min3c3l.txt 1 13 5 2>&1 | tee -a "$out"
stdbuf --output=L ./a.out data/min8c1l.txt 1 8 5  2>&1 | tee -a "$out"
