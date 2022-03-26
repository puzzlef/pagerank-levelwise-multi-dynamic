Performance of **Monolithic** PageRank with vertices split by components
vs **Levelswise** PageRank with topologically-ordered levels of components
([pull], [CSR]).

This experiment was for comparing performance between:
1. Find dynamic pagerank of updated graph using Monolithic PageRank.
2. Find dynamic pagerank of updated graph using Levelwise PageRank.
3. Find dynamic pagerank of updated graph using pure CPU/GPU [HyPR].
4. Find static pagerank of updated graph using plain [STIC-D PageRank] (CPU only).
5. Find incremental pagerank of updated using [nvGraph] PageRank (GPU only).

> - Affected vertices are grouped together by SCC. [report][r1]
> - Small SCCs are combined together. [report][r1]
> - Details of CUDA implementation. [report][r2]
> - L2-norm converges slower that L∞-norm. [report][r3]

[r1]: https://gist.github.com/wolfram77/12e5a19ff081b2e3280d04331a9976ca
[r2]: https://gist.github.com/wolfram77/4ef16ab9699ac03a617b8731dd240e1f
[r3]: https://gist.github.com/wolfram77/6dc740392d2f4e713fafdaea4ec1eba2


This study was carried out to extend the Levelwise strategy of PageRank computation in the [STIC-D paper] to perform dynamic PageRank on the CPU as well as the GPU. This levelwise computation of PageRank is computationally more efficient because it processes SCCs in topological order, avoiding unnecessary recomputation of SCCs that are dependent upon ranks of vertices in other SCCs which have not yet converged. We have compared it to our monolithic CPU/GPU implementations, along with other state-of-the art implementations, such as nvGraph and HyPR in batch sizes of 500, 1000, 2000, 5000, and 10000 edges. We also contrasted the performance of a batched update to a series of single edge updates (cumulative).

Results indicate that Levelwise approach is in general faster than Monolithic PageRank on the CPU, and the opposite is true on the GPU. This likely has to do with the fact that processing a large number of small levels is inefficient. Hence with Levelwise PageRank, smaller levels/components should be combined and processed at a time in order to help improve GPU usage efficiency.

All outputs are saved in [out](out/) and a small part of the output is listed
here. The input data used for this experiment is available at the
[SuiteSparse Matrix Collection]. This experiment was done with guidance
from [Prof. Dip Sankar Banerjee] and [Prof. Kishore Kothapalli].

<br>

```bash
$ nvcc -std=c++17 -Xcompiler -fopenmp -lnvgraph -O3 main.cu
$ ./a.out ~/data/arabic-2005.mtx
$ ./a.out ~/data/uk-2005.mtx
$ ...

# Loading graph /home/subhajit/data/arabic-2005.mtx ...
# order: 22744080 size: 639999458 {}
# ...
```

![](https://i.imgur.com/P1mlTU5.png)
![](https://i.imgur.com/1XdHMjw.png)

<br>
<br>


## References

- [STIC-D: algorithmic techniques for efficient parallel pagerank computation on real-world graphs][STIC-D paper]
- [PageRank Algorithm, Mining massive Datasets (CS246), Stanford University][this lecture]
- [SuiteSparse Matrix Collection]

<br>
<br>

[![](https://i.imgur.com/89cRRdY.jpg)](https://www.youtube.com/watch?v=iMdq5_5eib0)

[Prof. Dip Sankar Banerjee]: https://sites.google.com/site/dipsankarban/
[Prof. Kishore Kothapalli]: https://www.iiit.ac.in/people/faculty/kkishore/
[SuiteSparse Matrix Collection]: https://suitesparse-collection-website.herokuapp.com
[STIC-D paper]: https://gist.github.com/wolfram77/bb09968cc0e592583c4b180243697d5a
[STIC-D PageRank]: https://gist.github.com/wolfram77/bb09968cc0e592583c4b180243697d5a
[HyPR]: https://gist.github.com/wolfram77/50224c1bf5585a719b1c87113e95d074
[nvGraph]: https://github.com/rapidsai/nvgraph
[pull]: https://github.com/puzzlef/pagerank-push-vs-pull
[CSR]: https://github.com/puzzlef/pagerank-class-vs-csr
[this lecture]: https://www.youtube.com/watch?v=ke9g8hB0MEo
[puzzlef]: https://puzzlef.github.io
