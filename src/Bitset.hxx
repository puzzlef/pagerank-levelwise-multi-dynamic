#pragma once
#include <utility>
#include <vector>
#include <algorithm>
#include "_main.hxx"

using std::pair;
using std::vector;
using std::iter_swap;
using std::find_if;
using std::lower_bound;
using std::sort;
using std::unique;




// BITSET (UNSORTED)
// -----------------

template <class T=NONE>
class BitsetUnsorted {
  vector<pair<int, T>> ids;

  // Cute helpers
  private:
  auto lookup(int id) const {
    auto fn = [&](const auto& e) { return e.first == id; };
    return find_if(ids.begin(), ids.end(), fn);
  }

  // Read as iterable.
  public:
  auto entries() const { return transformIter(ids, [](const auto& e) { return e; }); }
  auto keys()    const { return transformIter(ids, [](const auto& e) { return e.first; }); }
  auto values()  const { return transformIter(ids, [](const auto& e) { return e.second; }); }

  // Read operations.
  public:
  size_t size()      const { return ids.size(); }
  bool   has(int id) const { return lookup(id) != ids.end(); }
  T      get(int id) const { auto it = lookup(id); return it == ids.end()? T() : (*it).second; }

  // Write operations
  public:
  void clear() {
    ids.clear();
  }

  void set(int id, T v) {
    auto it = lookup(id);
    if (it == ids.end()) return;
    (*it).second = v;
  }

  void add(int id, T v=T()) {
    if (!has(id)) ids.push_back({id, v});
  }

  void remove(int id) {
    auto it = lookup(id);
    if (it == ids.end()) return;
    iter_swap(it, ids.end()-1);
    ids.pop_back();
  }
};




// BITSET (SORTED)
// --------------

template <class T=NONE>
class BitsetSorted {
  vector<pair<int, T>> ids;

  // Cute helpers
  private:
  auto where(int id) const {
    auto fc = [](const auto& e, int id) { return e.first < id; };
    return lower_bound(ids.begin(), ids.end(), id, fc);
  }

  auto lookup(int id) const {
    auto it = where(id);
    return it != ids.end() && (*it).first == id? it : ids.end();
  }

  // Read as iterable.
  public:
  auto entries() const { return transformIter(ids, [](const auto& e) { return e; }); }
  auto keys()    const { return transformIter(ids, [](const auto& e) { return e.first; }); }
  auto values()  const { return transformIter(ids, [](const auto& e) { return e.second; }); }

  // Read operations.
  public:
  size_t size()      const { return ids.size(); }
  bool   has(int id) const { return lookup(id) != ids.end(); }
  T      get(int id) const { auto it = lookup(id); return it == ids.end()? T() : (*it).second; }

  // Write operations
  public:
  void correct() {
    auto fl = [](const auto& a, const auto& b) { return a.first <  b.first; };
    auto fe = [](const auto& a, const auto& b) { return a.first == b.first; };
    sort(ids.begin(), ids.end(), fl);
    auto it = unique(ids.begin(), ids.end(), fe);
    ids.resize(it - ids.begin());
  }

  void clear() {
    ids.clear();
  }

  void set(int id, T v) {
    auto it = lookup(id);
    if (it == ids.end()) return;
    (*it).second = v;
  }

  void add(int id, T v=T()) {
    ids.push_back({id, v});
  }

  void addChecked(int id, T v=T()) {
    auto it = where(id);
    if (it != ids.end() && (*it).first == id) return;
    ids.insert(it, {id, v});
  }

  void remove(int id) {
    auto it = lookup(id);
    if (it == ids.end()) return;
    ids.erase(it);
  }
};
