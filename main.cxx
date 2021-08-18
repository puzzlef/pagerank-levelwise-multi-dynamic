#include <cmath>
#include <string>
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

void runPagerankBatch(const string& data, int repeat, int original, int update) {
  vector<float> r0, r1, r2, r3;
  vector<float> s0, s1, s2, s3;
  vector<float> *init = nullptr;
  PagerankOptions<float> o = {repeat};

  DiGraph<> x;
  stringstream s(data);
  // Read original edges.
  if (!readSnapTemporal(x, s, original)) return;
  auto ksOld = vertices(x);
  auto a0 = pagerankTeleport(x, init, o);
  auto a1 = pagerankLoop(x, init, o);
  auto a2 = pagerankLoopAll(x, init, o);
  auto a3 = pagerankRemove(x, init, o);
  r0 = move(a0.ranks);
  r1 = move(a1.ranks);
  r2 = move(a2.ranks);
  r3 = move(a3.ranks);

  // Read edges for this update.
  auto y = copy(x);
  if (!readSnapTemporal(y, s, update)) return;
  auto ks = vertices(y);
  s0.resize(y.span());
  s1.resize(y.span());
  s2.resize(y.span());
  s3.resize(y.span());

  // Adjust ranks.
  adjustRanks(s0, r0, ksOld, ks, 0.0f, float(ksOld.size())/ks.size(), 1.0f/ks.size());
  adjustRanks(s1, r1, ksOld, ks, 0.0f, float(ksOld.size())/ks.size(), 1.0f/ks.size());
  adjustRanks(s2, r2, ksOld, ks, 0.0f, float(ksOld.size())/ks.size(), 1.0f/ks.size());
  adjustRanks(s3, r3, ksOld, ks, 0.0f, float(ksOld.size())/ks.size(), 1.0f/ks.size());

  // Find pagerank by teleporting to a random vertex from every dead end.
  // printf("r0: "); print(r0); printf("\n");
  // printf("s0: "); print(s0); printf("\n");
  auto b0 = pagerankTeleport(y, init, o);
  printRow(y, b0, b0, "pagerankTeleport (static)");
  auto c0 = pagerankTeleport(y, &s0, o);
  printRow(y, b0, c0, "pagerankTeleport (incremental)");
  auto d0 = pagerankTeleportDynamic(x, y, &s0, o);
  printRow(y, b0, d0, "pagerankTeleport (dynamic)");
  // printf("c0: "); print(c0.ranks); printf("\n");
  // printf("d0: "); print(d0.ranks); printf("\n");

  // Find pagerank by self-looping dead ends.
  // printf("r1: "); print(r1); printf("\n");
  // printf("s1: "); print(s1); printf("\n");
  auto b1 = pagerankLoop(y, init, o);
  printRow(y, b1, b1, "pagerankLoop (static)");
  auto c1 = pagerankLoop(y, &s1, o);
  printRow(y, b1, c1, "pagerankLoop (incremental)");
  auto d1 = pagerankLoopDynamic(x, y, &s1, o);
  printRow(y, b1, d1, "pagerankLoop (dynamic)");

  // Find pagerank by self-looping all vertices.
  // printf("r2: "); print(r2); printf("\n");
  // printf("s2: "); print(s2); printf("\n");
  auto b2 = pagerankLoopAll(y, init, o);
  printRow(y, b2, b2, "pagerankLoopAll (static)");
  auto c2 = pagerankLoopAll(y, &s2, o);
  printRow(y, b2, c2, "pagerankLoopAll (incremental)");
  auto d2 = pagerankLoopAllDynamic(x, y, &s2, o);
  printRow(y, b2, d2, "pagerankLoopAll (dynamic)");

  // Find pagerank by removing dead ends initially, and calculating their ranks after convergence.
  // printf("r3: "); print(r3); printf("\n");
  // printf("s3: "); print(s3); printf("\n");
  auto b3 = pagerankRemove(y, init, o);
  printRow(y, b3, b3, "pagerankRemove (static)");
  auto c3 = pagerankRemove(y, &s3, o);
  printRow(y, b3, c3, "pagerankRemove (incremental)");
  auto d3 = pagerankRemoveDynamic(x, y, &s3, o);
  printRow(y, b3, d3, "pagerankRemove (dynamic)");

  // New graph is now old.
  x = move(y);
}


void runPagerank(const string& data, int repeat, int original, int update) {
  int M = countLines(data);
  printf("Temporal edges: %d\n", M);
  runPagerankBatch(data, repeat, original, update);
}


int main(int argc, char **argv) {
  char *file = argv[1];
  int repeat   = stoi(argv[2]);
  int original = stoi(argv[3]);
  int update   = stoi(argv[4]);
  printf("Using graph %s ...\n", file);
  string d = readFile(file);
  runPagerank(d, repeat, original, update);
    printf("\n");
  return 0;
}
