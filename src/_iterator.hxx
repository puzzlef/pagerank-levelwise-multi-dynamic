#pragma once
#include <cstddef>
#include <iterator>
#include <unordered_map>
#include <algorithm>

using std::ptrdiff_t;
using std::input_iterator_tag;
using std::output_iterator_tag;
using std::forward_iterator_tag;
using std::bidirectional_iterator_tag;
using std::random_access_iterator_tag;
using std::iterator_traits;
using std::unordered_map;
using std::distance;
using std::max;




// ITERATOR-*
// ----------
// Helps create iterators.

#ifndef ITERATOR_USING
#define ITERATOR_USING(cat, dif, val, ref, ptr) \
  using iterator_category = cat; \
  using difference_type   = dif; \
  using value_type = val; \
  using reference  = ref; \
  using pointer    = ptr;

#define ITERATOR_USING_I(I) \
  using iterator_category = typename I::iterator_category; \
  using difference_type   = typename I::difference_type; \
  using value_type = typename I::value_type; \
  using reference  = typename I::reference; \
  using pointer    = typename I::pointer;

#define ITERATOR_USING_IVR(I, val, ref) \
  using iterator_category = typename I::iterator_category; \
  using difference_type   = typename I::difference_type; \
  using value_type = val; \
  using reference  = ref; \
  using pointer    = value_type*;
#endif


#ifndef ITERATOR_DEREF
#define ITERATOR_DEREF(I, i, se, be, ae) \
  reference operator*() { return se; } \
  reference operator[](difference_type i) { return be; } \
  pointer operator->() { return ae; }
#endif


#ifndef ITERATOR_NEXT
#define ITERATOR_NEXTP(I, ie)  \
  I& operator++() { ie; return *this; }  \
  I operator++(int) { I a = *this; ++(*this); return a; }

#define ITERATOR_NEXTN(I, de) \
  I& operator--() { de; return *this; }  \
  I operator--(int) { I a = *this; --(*this); return a; }

#define ITERATOR_NEXT(I, ie, de) \
  ITERATOR_NEXTP(I, ie) \
  ITERATOR_NEXTN(I, de)
#endif


#ifndef ITERATOR_ADVANCE
#define ITERATOR_ADVANCEP(I, i, fe) \
  I& operator+=(difference_type i) { fe; return *this; }

#define ITERATOR_ADVANCEN(I, i, be) \
  I& operator-=(difference_type i) { be; return *this; }

#define ITERATOR_ADVANCE(I, i, fe, be) \
  ITERATOR_ADVANCEP(I, i, fe) \
  ITERATOR_ADVANCEN(I, i, be)
#endif


#ifndef ITERATOR_ARITHMETICP
#define ITERATOR_ARITHMETICP(I, a, b, ...)  \
  friend I operator+(const I& a, difference_type b) { return I(__VA_ARGS__); } \
  friend I operator+(difference_type b, const I& a) { return I(__VA_ARGS__); }
#endif


#ifndef ITERATOR_ARITHMETICN
#define ITERATOR_ARITHMETICN(I, a, b, ...) \
  friend I operator-(const I& a, difference_type b) { return I(__VA_ARGS__); } \
  friend I operator-(difference_type b, const I& a) { return I(__VA_ARGS__); }
#endif


#ifndef ITERATOR_COMPARISION
#define ITERATOR_COMPARISION(I, a, b, ae, be)  \
  friend bool operator==(const I& a, const I& b) { return ae == be; } \
  friend bool operator!=(const I& a, const I& b) { return ae != be; } \
  friend bool operator>=(const I& a, const I& b) { return ae >= be; } \
  friend bool operator<=(const I& a, const I& b) { return ae <= be; } \
  friend bool operator>(const I& a, const I& b) { return ae > be; } \
  friend bool operator<(const I& a, const I& b) { return ae < be; }
#endif


#ifndef ITERABLE_SIZE
#define ITERABLE_SIZE(se) \
  size_t size() { return se; } \
  bool empty() { return size() == 0; }
#endif




// ITERABLE
// --------

template <class I>
class Iterable {
  const I ib, ie;

  public:
  Iterable(I ib, I ie) : ib(ib), ie(ie) {}
  auto begin() const { return ib; }
  auto end() const   { return ie; }
};


template <class I>
auto iterable(I ib, I ie) {
  return Iterable<I>(ib, ie);
}

template <class J>
auto iterable(const J& x) {
  using I = decltype(x.begin());
  return Iterable<I>(x.begin(), x.end());
}




// SIZED-ITERABLE
// --------------

template <class I>
class SizedIterable : public Iterable<I> {
  const size_t N;

  public:
  SizedIterable(I ib, I ie, size_t N) : Iterable<I>(ib, ie), N(N) {}
  ITERABLE_SIZE(N)
};


template <class I>
auto sizedIterable(I ib, I ie, int N) {
  return SizedIterable<I>(ib, ie, N);
}

template <class I>
auto sizedIterable(I ib, I ie) {
  return SizedIterable<I>(ib, ie, distance(ib, ie));
}

template <class J>
auto sizedIterable(const J& x, int N) {
  using I = decltype(x.begin());
  return Iterable<I>(x.begin(), x.end(), N);
}

template <class J>
auto sizedIterable(const J& x) {
  using I = decltype(x.begin());
  return Iterable<I>(x.begin(), x.end());
}




// SIZE
// ----

template <class T>
int size(const vector<T>& x) {
  return x.size();
}

template <class I>
int size(const SizedIterable<I>& x) {
  return x.size();
}

template <class J>
int size(const J& x) {
  return distance(x.begin(), x.end());
}




// CSIZE
// -----
// Compile-time size.

template <class T>
int csize(const vector<T>& x) {
  return x.size();
}

template <class I>
int csize(const SizedIterable<I>& x) {
  return x.size();
}

template <class J>
int csize(const J& x) {
  return -1;
}




// SLICE
// -----

template <class J>
auto slice(const J& x, int i) {
  auto ib = x.begin(), ie = x.end();
  return sizedIterable(ib+i<ie? ib+i : ie, ie);
}

template <class J>
auto slice(const J& x, int i, int I) {
  auto ib = x.begin(), ie = x.end();
  return sizedIterable(ib+i<ie? ib+i : ie, ib+I<ie? ib+I : ie, I-i);
}



// DEFAULT
// -------
// Return default value of type, always.

template <class T>
class DefaultIterator {
  public:
  using iterator = DefaultIterator;
  using iterator_category = random_access_iterator_tag;
  using difference_type   = ptrdiff_t;
  using value_type = T;
  using reference  = const T&;
  using pointer    = const T*;

  public: // base
  DefaultIterator() {}
  iterator& operator++() { return *this; }
  value_type operator*() const { return T(); }
  friend void swap(iterator& l, iterator& r) {}

  public: // input
  iterator operator++(int) { return *this; }
  pointer operator->() const { return NULL; }
  friend bool operator==(const iterator& l, const iterator& r) { return true; }
  friend bool operator!=(const iterator& l, const iterator& r) { return false; }

  public: // bidirectional
  iterator& operator--() { return *this; }
  iterator operator--(int) { return *this; }

  public: // random access
  friend bool operator<(const iterator& l, const iterator& r) { return false; }
  friend bool operator>(const iterator& l, const iterator& r) { return false; }
  friend bool operator<=(const iterator& l, const iterator& r) { return true; }
  friend bool operator>=(const iterator& l, const iterator& r) { return true; }

  iterator& operator+=(size_t n) { return *this; }
  friend iterator operator+(const iterator& l, size_t n) { return l; }
  friend iterator operator+(size_t n, const iterator& r) { return r; }
  iterator& operator-=(size_t n) { return *this; }
  friend iterator operator-(const iterator& l, size_t n) { return l; }
  friend difference_type operator-(const iterator& l, const iterator& r) { return 0; }

  reference operator[](size_t i) const { return T(); }
};

template <class T>
auto defaultIterator(const T& _) {
  return DefaultIterator<T>();
}


// DEFAULT (REF)
// -------------
// Return default value of type, always.

template <class T>
class DefaultRefIterator : public DefaultIterator<T> {
  const T x;

  public:
  using iterator = DefaultRefIterator;
  using reference  = const T&;
  using pointer    = const T*;

  public:
  DefaultRefIterator() : x() {}
  reference operator*() const { return x; }
  pointer operator->() const { return &x; }
  reference operator[](size_t i) const { return x; }
};

template <class T>
auto defaultRefIterator(const T& _) {
  return DefaultRefIterator<T>();
}




// SELECT (BASE)
// -------------
// Select iterator by index.

template <class I0, class I1>
class Select2BaseIterator {
  public:
  using iterator = Select2BaseIterator;
  using difference_type   = typename I0::difference_type;
  using value_type = typename I0::value_type;
  using reference  = typename I0::reference;
  using pointer    = typename I0::pointer;

  protected:
  using ID = DefaultIterator<value_type>;

  public:
  const int s;
  I0 i0;
  I1 i1;
  ID i2;
  ID i3;
  ID i4;
  ID i5;
  ID i6;
  ID i7;
  ID id;

  public:
  Select2BaseIterator(int s, I0 i0, I1 i1)
  : s(s), i0(i0), i1(i1), i2(), i3(), i4(), i5(), i6(), i7(), id() {}

  iterator& operator++() {
    switch (s) {
      default: break;
      case 0: ++i0; break;
      case 1: ++i1; break;
    }
    return *this;
  }

  reference operator*() {
    switch (s) {
      default: return *id;
      case 0: return *i0;
      case 1: return *i1;
    }
  }

  friend void swap(iterator& l, iterator& r) {
    // Does this do the right thing?
    if (l.s != r.s) return;
    switch (l.s) {
      default: break;
      case 0: swap(l.i0, r.i0); break;
      case 1: swap(l.i1, r.i1); break;
    }
  }
};


template <class I0, class I1>
auto selectBaseIterator(int s, I0 i0, I1 i1) {
  return Select2BaseIterator<I0, I1>(s, i0, i1);
}

template <class J0, class J1>
auto selectBaseIter(int s, const J0& x0, const J1& x1) {
  auto ib = selectBaseIterator(s, x0.begin(), x1.begin());
  auto ie = selectBaseIterator(s, x0.end(), x1.end());
  return iterable(ib, ie);
}




// SELECT (INPUT)
// --------------

template <class I0, class I1>
class Select2InputIterator : public Select2BaseIterator<I0, I1> {
  public:
  using iterator = Select2InputIterator;
  using iterator_category = input_iterator_tag;
  using value_type = typename Select2BaseIterator<I0, I1>::value_type;
  using pointer    = typename Select2BaseIterator<I0, I1>::pointer;

  protected:
  using ID = DefaultIterator<value_type>;

  public:
  Select2InputIterator(int s, I0 i0, I1 i1)
  : Select2BaseIterator<I0, I1>(s, i0, i1) {}

  iterator& operator++() { this->Select2BaseIterator<I0, I1>::operator++(); return *this; }
  iterator operator++(int) { auto it = *this; ++(*this); return it; }
  // value_type operator*() const {}

  pointer operator->() {
    switch (this->s) {
      default: return this->id.ID::operator->();
      case 0: return this->i0.I0::operator->();
      case 1: return this->i1.I1::operator->();
    }
  }

  friend bool operator==(const iterator& l, const iterator& r) {
    if (l.s != r.s) return false;
    switch (l.s) {
      default: return true;
      case 0: return l.i0 == r.i0;
      case 1: return l.i1 == r.i1;
    }
  }

  friend bool operator!=(const iterator& l, const iterator& r) {
    if (l.s != r.s) return true;
    switch (l.s) {
      default: return false;
      case 0: return l.i0 != r.i0;
      case 1: return l.i1 != r.i1;
    }
  }
};


template <class I0, class I1>
auto selectInputIterator(int s, I0 i0, I1 i1) {
  return Select2InputIterator<I0, I1>(s, i0, i1);
}

template <class J0, class J1>
auto selectInputIter(int s, const J0& x0, const J1& x1) {
  auto ib = selectInputIterator(s, x0.begin(), x1.begin());
  auto ie = selectInputIterator(s, x0.end(), x1.end());
  return iterable(ib, ie);
}




/*
  reference operator[](difference_type i) {
    switch (s) {
      default:
      case 0: return i0[i];
      case 1: return i1[i];
      case 2: return i2[i];
      case 3: return i3[i];
      case 4: return i4[i];
      case 5: return i5[i];
      case 6: return i6[i];
      case 7: return i7[i];
    }
  }
*/


// TRANSFORM
// ---------

template <class I, class F>
class TransformIterator {
  I it;
  const F fn;

  public:
  ITERATOR_USING_IVR(I, decltype(fn(*it)), value_type)
  TransformIterator(I it, F fn) : it(it), fn(fn) {}
  ITERATOR_DEREF(TransformIterator, i, fn(*it), fn(it[i]), NULL)
  ITERATOR_NEXT(TransformIterator, ++it, --it)
  ITERATOR_ADVANCE(TransformIterator, i, it += i, it -= i)
  ITERATOR_ARITHMETICP(TransformIterator, a, b, a.it+b)
  ITERATOR_ARITHMETICN(TransformIterator, a, b, a.it-b)
  ITERATOR_COMPARISION(TransformIterator, a, b, a.it, b.it)
};


template <class I, class F>
auto transform(I ib, I ie, F fn) {
  auto b = TransformIterator<I, F>(ib, fn);
  auto e = TransformIterator<I, F>(ie, fn);
  return iterable(b, e);
}

template <class J, class F>
auto transform(const J& x, F fn) {
  auto b = x.begin();
  auto e = x.end();
  return transform(b, e, fn);
}




// FILTER
// ------

template <class I, class F>
class FilterIterator {
  I it;
  const I ie;
  const F fn;

  public:
  ITERATOR_USING_I(I);
  FilterIterator(I ix, I ie, F fn) : it(ix), ie(ie), fn(fn) { while (it!=ie && !fn(*it)) ++it; }
  ITERATOR_DEREF(FilterIterator, i, *it, it[i], NULL)
  ITERATOR_NEXTP(FilterIterator, do { ++it; } while (it!=ie && !fn(*it)))
  ITERATOR_ADVANCEP(FilterIterator, i, for (; i>0; i--) ++it)
  ITERATOR_ARITHMETICP(FilterIterator, a, b, a.it+b)
  ITERATOR_COMPARISION(FilterIterator, a, b, a.it, b.it)
};


template <class I, class F>
auto filter(I ib, I ie, F fn) {
  auto b = FilterIterator<I, F>(ib, ie, fn);
  auto e = FilterIterator<I, F>(ie, ie, fn);
  return iterable(b, e);
}

template <class J, class F>
auto filter(const J& x, F fn) {
  return filter(x.begin(), x.end(), fn);
}




// RANGE
// -----

template <class T>
int rangeSize(T v, T V, T DV=1) {
  return max(0, (int) ceilDiv(V-v, DV));
}

template <class T>
int rangeLast(T v, T V, T DV=1) {
  return v + DV*(rangeSize(v, V, DV) - 1);
}


template <class T>
class RangeIterator {
  T n;

  public:
  ITERATOR_USING(random_access_iterator_tag, T, T, T, const T*)
  RangeIterator(T n) : n(n) {}
  ITERATOR_DEREF(RangeIterator, i, n, n+i, &n)
  ITERATOR_NEXT(RangeIterator, ++n, --n)
  ITERATOR_ADVANCE(RangeIterator, i, n += i, n -= i)
  ITERATOR_ARITHMETICP(RangeIterator, a, b, a.n+b)
  ITERATOR_ARITHMETICN(RangeIterator, a, b, a.n-b)
  ITERATOR_COMPARISION(RangeIterator, a, b, a.n, b.n)
};


template <class T>
auto rangeIter(T V) {
  auto ib = RangeIterator<T>(0);
  auto ie = RangeIterator<T>(V);
  return iterable(ib, ie);
}

template <class T>
auto rangeIter(T v, T V, T DV=1) {
  auto x = rangeIter(rangeSize(v, V, DV));
  return transform(x, [=](int n) { return v+DV*n; });
}
