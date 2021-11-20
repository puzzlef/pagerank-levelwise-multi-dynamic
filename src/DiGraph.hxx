#pragma once
#include <vector>
#include <ostream>
#include <iostream>
#include "_main.hxx"
#include "Bitset.hxx"

using std::vector;
using std::ostream;
using std::cout;




// DI-GRAPH
// --------

template <class V=NONE, class E=NONE>
class DiGraph {
  template <class T>
  using Bitset = BitsetSorted<T>;

  public:
  using TVertex = V;
  using TEdge   = E;

  private:
  Bitset<E>    none;
  vector<bool> vex;
  vector<V>    vdata;
  vector<Bitset<E>> edata;
  int N = 0, M = 0;

  // Read operations
  public:
  int span()  const { return vex.size(); }
  int order() const { return N; }
  int size()  const { return M; }

  bool hasVertex(int u)      const { return u < span() && vex[u]; }
  bool hasEdge(int u, int v) const { return u < span() && edata[u].has(v); }
  auto edges(int u)          const { return u < span()? edata[u].keys() : none.keys(); }
  int degree(int u)          const { return u < span()? edata[u].size() : 0; }
  auto vertices()     const { return filterIter(rangeIter(span()), [&](int u) { return  vex[u]; }); }
  auto nonVertices()  const { return filterIter(rangeIter(span()), [&](int u) { return !vex[u]; }); }
  auto inEdges(int v) const { return filterIter(rangeIter(span()), [&](int u) { return edata[u].has(v); }); }
  int inDegree(int v) const { return    countIf(rangeIter(span()), [&](int u) { return edata[u].has(v); }); }

  V vertexData(int u)   const { return hasVertex(u)? vdata[u] : V(); }
  void setVertexData(int u, V d) { if (hasVertex(u)) vdata[u] = d; }
  E edgeData(int u, int v)   const { return hasEdge(u, v)? edata[u].get(v) : E(); }
  void setEdgeData(int u, int v, E d) { if (hasEdge(u, v)) edata[u].set(v, d); }

  // Write operations
  public:
  void clear() {
    vex.clear();
    vdata.clear();
    edata.clear();
  }

  void addVertex(int u, V d=V()) {
    if (hasVertex(u)) return;
    if (u >= span()) {
      vex.resize(u+1);
      vdata.resize(u+1);
      edata.resize(u+1);
    }
    vex[u] = true;
    vdata[u] = d;
    N++;
  }

  void addEdge(int u, int v, E d=E()) {
    if (hasEdge(u, v)) return;
    addVertex(u);
    addVertex(v);
    edata[u].add(v, d);
    M++;
  }

  void removeEdge(int u, int v) {
    if (!hasEdge(u, v)) return;
    edata[u].remove(v);
    M--;
  }

  void removeEdges(int u) {
    if (!hasVertex(u)) return;
    M -= degree(u);
    edata[u].clear();
  }

  void removeInEdges(int v) {
    if (!hasVertex(v)) return;
    for (int u : inEdges(v))
      removeEdge(u, v);
  }

  void removeVertex(int u) {
    if (!hasVertex(u)) return;
    removeEdges(u);
    removeInEdges(u);
    vex[u] = false;
    N--;
  }
};




// DI-GRAPH PRINT
// --------------

template <class V, class E>
void write(ostream& a, const DiGraph<V, E>& x, bool all=false) {
  a << "order: " << x.order() << " size: " << x.size();
  if (!all) { a << " {}"; return; }
  a << " {\n";
  for (int u : x.vertices()) {
    a << "  " << u << " ->";
    for (int v : x.edges(u))
      a << " " << v;
    a << "\n";
  }
  a << "}";
}

template <class V, class E>
ostream& operator<<(ostream& a, const DiGraph<V, E>& x) {
  write(a, x);
  return a;
}
