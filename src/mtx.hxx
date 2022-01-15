#pragma once
#include <string>
#include <istream>
#include <sstream>
#include <fstream>
#include <algorithm>
#include "_main.hxx"
#include "DiGraph.hxx"

using std::string;
using std::istream;
using std::stringstream;
using std::ofstream;
using std::getline;
using std::max;




// READ-MTX
// --------

template <class G, class FS>
void readMtx(G& a, istream& s, FS fs) {
  string ln, h0, h1, h2, h3, h4;

  // read header
  while (1) {
    getline(s, ln);
    if (ln.find('%')!=0) break;
    if (ln.find("%%")!=0) continue;
    stringstream ls(ln);
    ls >> h0 >> h1 >> h2 >> h3 >> h4;
  }
  if (h1!="matrix" || h2!="coordinate") return;
  bool sym = h4=="symmetric" || h4=="skew-symmetric";

  // read rows, cols, size
  int r, c, sz;
  stringstream ls(ln);
  ls >> r >> c >> sz;
  int n = max(r, c);
  for (int u=1; u<=n; u++)
    a.addVertex(u);

  // read edges (from, to)
  int events = 0;
  while (getline(s, ln)) {
    int u, v;
    ls = stringstream(ln);
    if (!(ls >> u >> v)) break;
    a.addEdgeUnchecked(u, v);
    if (sym) a.addEdgeUnchecked(v, u);
    if (++events<=1000000) continue;
    fs(a.size()/float(sz)); events = 0;
  }
  a.correct();
}

template <class FS>
auto readMtx(istream& s, FS fs) {
  DiGraph<> a; readMtx(a, s, fs);
  return a;
}


template <class G, class FS>
void readMtx(G& a, const char *pth, FS fs) {
  string buf = readFile(pth);
  stringstream s(buf);
  return readMtx(a, s, fs);
}

template <class FS>
auto readMtx(const char *pth, FS fs) {
  DiGraph<> a; readMtx(a, pth, fs);
  return a;
}




// WRITE-MTX
// ---------

template <class G>
void writeMtx(ostream& a, const G& x) {
  a << "%%MatrixMarket matrix coordinate real asymmetric\n";
  a << x.order() << " " << x.order() << " " << x.size() << "\n";
  for (int u : x.vertices()) {
    for (int v : x.edges(u))
      a << u << " " << v << " " << x.edgeData(u) << "\n";
  }
}

template <class G>
void writeMtx(string pth, const G& x) {
  string s0; stringstream s(s0);
  writeMtx(s, x);
  ofstream f(pth);
  f << s.rdbuf();
  f.close();
}
