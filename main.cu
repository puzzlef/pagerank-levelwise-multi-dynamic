#include <cmath>
#include <random>
#include <vector>
#include <cstdio>
#include <iostream>
#include "src/main.hxx"

using namespace std;




#define REPEAT 5

template <class G, class H>
void runPagerankBatch(const G& x, const H& xt, const vector<float>& ranksOld, int batch) {
  int span = int(1.1 * x.span());
  vector<float> ranksAdj;
  vector<float> *initStatic  = nullptr;
  vector<float> *initDynamic = &ranksAdj;
  random_device dev;
  default_random_engine rnd(dev());

  // Add random edges
  auto y = copy(x);
  for (int i=0; i<batch; i++)
    addRandomEdgeByDegree(y, rnd, span);
  loopDeadEnds(y);
  auto yt = transposeWithDegree(y);

  // Adjust ranks for dynamic pagerank
  auto ksOld = vertices(x);
  auto ks    = vertices(y);
  ranksAdj.resize(y.span());
  adjustRanks(ranksAdj, ranksOld, ksOld, ks, 0.0f, float(ksOld.size())/ks.size(), 1.0f/ks.size());

  // Find static pagerank using nvGraph.
  auto a1 = pagerankNvgraph(yt, initStatic, {REPEAT});
  auto e1 = l1Norm(a1.ranks, a1.ranks);
  print(y); printf(" [%09.3f ms; %03d iters.] [%.4e err.] pagerankNvgraph [static]\n", a1.time, a1.iterations, e1);

  // Find dynamic pagerank using nvGraph.
  auto a2 = pagerankNvgraph(yt, initDynamic, {REPEAT});
  auto e2 = l1Norm(a2.ranks, a1.ranks);
  print(y); printf(" [%09.3f ms; %03d iters.] [%.4e err.] pagerankNvgraph [dynamic]\n", a2.time, a2.iterations, e2);

  // Find static pagerank using standard algorithm.
  auto a3 = pagerankMonolithic(yt, initStatic, {REPEAT});
  auto e3 = l1Norm(a3.ranks, a1.ranks);
  print(y); printf(" [%09.3f ms; %03d iters.] [%.4e err.] pagerankMonolithic [static]\n", a3.time, a3.iterations, e3);

  // Find dynamic pagerank using standard algorithm.
  auto a4 = pagerankMonolithic(yt, initDynamic, {REPEAT});
  auto e4 = l1Norm(a4.ranks, a1.ranks);
  print(y); printf(" [%09.3f ms; %03d iters.] [%.4e err.] pagerankMonolithic [dynamic]\n", a4.time, a4.iterations, e4);

  // Find static pagerank using levelwise algorithm.
  auto a5 = pagerankLevelwise(y, yt, initStatic, {REPEAT});
  auto e5 = l1Norm(a5.ranks, a1.ranks);
  print(y); printf(" [%09.3f ms; %03d iters.] [%.4e err.] pagerankLevelwise [static]\n", a5.time, a5.iterations, e5);

  // Find dynamic pagerank using levelwise algorithm.
  auto a6 = pagerankLevelwise(x, xt, y, yt, initDynamic, {REPEAT});
  auto e6 = l1Norm(a6.ranks, a1.ranks);
  print(y); printf(" [%09.3f ms; %03d iters.] [%.4e err.] pagerankLevelwise [dynamic]\n", a6.time, a6.iterations, e6);
}


template <class G, class H>
void runPagerank(const G& x, const H& xt, bool show) {
  vector<float> *init = nullptr;

  // Find pagerank using standard algorithm.
  auto a1 = pagerankMonolithic(xt, init, {REPEAT});
  auto e1 = l1Norm(a1.ranks, a1.ranks);
  printf("[%09.3f ms; %03d iters.] [%.4e err.] pagerankMonolithic\n", a1.time, a1.iterations, e1);
  if (show) println(a1.ranks);

  // Find pagerank for different batch sizes.
  for (int batch=1, i=0; batch<x.size(); batch*=i&1? 2:5, i++) {
    printf("\n# Batch size %.0e\n", (double) batch);
    for (int repeat=0; repeat<5; repeat++)
      runPagerankBatch(x, xt, a1.ranks, batch);
  }
}


int main(int argc, char **argv) {
  char *file = argv[1];
  bool  show = argc > 2;
  printf("Loading graph %s ...\n", file);
  auto x  = readMtx(file); println(x);
  loopDeadEnds(x); print(x); printf(" (loopDeadEnds)\n");
  auto xt = transposeWithDegree(x); print(xt); printf(" (transposeWithDegree)\n");
  runPagerank(x, xt, show);
  printf("\n");
  return 0;
}
