#pragma once
#include "components.hxx"




template <class H, class G, class C>
void blockgraph(H& a, const G& x, const C& comps) {
  auto c = componentIds(x, comps);
  for (int u : x.vertices()) {
    a.addVertex(c[u]);
    for (int v : x.edges(u))
      if (c[u] != c[v]) a.addEdge(c[u], c[v]);
  }
}

template <class G, class C>
auto blockgraph(const G& x, const C& comps) {
  G a; blockgraph(a, x, comps);
  return a;
}
