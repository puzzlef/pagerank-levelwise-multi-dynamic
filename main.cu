#include <cmath>
#include <string>
#include <vector>
#include <sstream>
#include <cstdio>
#include <iostream>
#include <utility>
#include "src/main.hxx"

using namespace std;




template <class G, class T>
void printRow(const G& x, const PagerankResult<T>& a, const PagerankResult<T>& b, const char *tec) {
  auto e = l1Norm(b.ranks, a.ranks);
  print(x); printf(" [%09.3f ms; %03d iters.] [%.4e err.] %s\n", b.time, b.iterations, e, tec);
}

void runPagerankBatch(const string& data, int repeat, int skip, int batch) {
  vector<float> r0, s0;
  vector<float> *init = nullptr;
  PagerankOptions<float> o = {repeat};

  DiGraph<> xo;
  stringstream s(data);
  while (true) {
    // Skip some edges (to speed up execution)
    if (skip>0 && !readSnapTemporal(xo, s, skip)) break;
    auto x  = selfLoop(xo, [&](int u) { return isDeadEnd(xo, u); });
    auto xt = transposeWithDegree(x);
    auto ksOld = vertices(x);
    auto a0 = pagerankNvgraph(x, xt, init, o);
    auto r0 = move(a0.ranks);

    // Read edges for this batch.
    auto yo = copy(xo);
    if (!readSnapTemporal(yo, s, batch)) break;
    auto y  = selfLoop(yo, [&](int u) { return isDeadEnd(yo, u); });
    auto yt = transposeWithDegree(y);
    auto ks = vertices(y);
    s0.resize(y.span());

    // Adjust ranks.
    adjustRanks(s0, r0, ksOld, ks, 0.0f, float(ksOld.size())/ks.size(), 1.0f/ks.size());

    // Find nvGraph-based pagerank.
    auto b0 = pagerankNvgraph(y, yt, init, o);
    printRow(y, b0, b0, "pagerankNvgraph (static)");
    auto c0 = pagerankNvgraph(y, yt, &s0, o);
    printRow(y, b0, c0, "pagerankNvgraph (incremental)");

    // Find sequential Monolithic pagerank.
    auto b1 = pagerankMonolithicSeq(y, yt, init, o);
    printRow(y, b0, b1, "pagerankMonolithicSeq (static)");
    auto c1 = pagerankMonolithicSeq(y, yt, &s0, o);
    printRow(y, b0, c1, "pagerankMonolithicSeq (incremental)");
    auto d1 = pagerankMonolithicSeqDynamic(x, xt, y, yt, &s0, o);
    printRow(y, b0, d1, "pagerankMonolithicSeq (dynamic)");

    // Find OpenMP-based Monolithic pagerank.
    auto b2 = pagerankMonolithicOmp(y, yt, init, o);
    printRow(y, b0, b2, "pagerankMonolithicOmp (static)");
    auto c2 = pagerankMonolithicOmp(y, yt, &s0, o);
    printRow(y, b0, c2, "pagerankMonolithicOmp (incremental)");
    auto d2 = pagerankMonolithicOmpDynamic(x, xt, y, yt, &s0, o);
    printRow(y, b0, d2, "pagerankMonolithicOmp (dynamic)");

    // Find CUDA-based Monolithic pagerank.
    auto b3 = pagerankMonolithicCuda(y, yt, init, o);
    printRow(y, b0, b3, "pagerankMonolithicCuda (static)");
    auto c3 = pagerankMonolithicCuda(y, yt, &s0, o);
    printRow(y, b0, c3, "pagerankMonolithicCuda (incremental)");
    auto d3 = pagerankMonolithicCudaDynamic(x, xt, y, yt, &s0, o);
    printRow(y, b0, d3, "pagerankMonolithicCuda (dynamic)");

    // Find sequential Levelwise pagerank.
    auto b4 = pagerankLevelwiseSeq(y, yt, init, o);
    printRow(y, b0, b4, "pagerankLevelwiseSeq (static)");
    auto c4 = pagerankLevelwiseSeq(y, yt, &s0, o);
    printRow(y, b0, c4, "pagerankLevelwiseSeq (incremental)");
    auto d4 = pagerankLevelwiseSeqDynamic(x, xt, y, yt, &s0, o);
    printRow(y, b0, d4, "pagerankLevelwiseSeq (dynamic)");

    // Find OpenMP-based Levelwise pagerank.
    auto b5 = pagerankLevelwiseOmp(y, yt, init, o);
    printRow(y, b0, b5, "pagerankLevelwiseOmp (static)");
    auto c5 = pagerankLevelwiseOmp(y, yt, &s0, o);
    printRow(y, b0, c5, "pagerankLevelwiseOmp (incremental)");
    auto d5 = pagerankLevelwiseOmpDynamic(x, xt, y, yt, &s0, o);
    printRow(y, b0, d5, "pagerankLevelwiseOmp (dynamic)");

    // Find CUDA-based Levelwise pagerank.
    // auto b6 = pagerankLevelwiseCuda(y, yt, init, o);
    // printRow(y, b0, b6, "pagerankLevelwiseCuda (static)");
    // auto c6 = pagerankLevelwiseCuda(y, yt, &s0, o);
    // printRow(y, b0, c6, "pagerankLevelwiseCuda (incremental)");
    // auto d6 = pagerankLevelwiseCudaDynamic(x, xt, y, yt, &s0, o);
    // printRow(y, b0, d6, "pagerankLevelwiseCuda (dynamic)");

    // New graph is now old.
    xo = move(yo);
  }
}


void runPagerank(const string& data, int repeat) {
  int M = countLines(data), steps = 100;
  printf("Temporal edges: %d\n", M);
  for (int batch=10, i=0; batch<M; batch*=i&1? 2:5, i++) {
    int skip = max(M/steps - batch, 0);
    printf("\n# Batch size %.0e\n", (double) batch);
    runPagerankBatch(data, repeat, skip, batch);
  }
}


int main(int argc, char **argv) {
  char *file = argv[1];
  int repeat = argc>2? stoi(argv[2]) : 5;
  printf("Using graph %s ...\n", file);
  string d = readFile(file);
  runPagerank(d, repeat);
    printf("\n");
  return 0;
}
