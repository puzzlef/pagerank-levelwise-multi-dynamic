Comparision of *OpenMP* and *CUDA-based*, *Monolithic* and *Levelwise* **Dynamic**
*PageRank algorithms*.

<br>


### With CUDA implementation on Fixed graphs

This experiment ([cuda-on-fixed]) was for comparing performance between:
1. Find **static** pagerank of updated graph using [nvGraph][pr-nvgraph].
2. Find **dynamic** pagerank of updated graph using [nvGraph][pr-nvgraph].
3. Find **static** [monolithic CUDA] based pagerank of updated graph.
4. Find **dynamic** [monolithic CUDA] based pagerank of updated graph.
5. Find **static** [levelwise CUDA] based pagerank of updated graph.
6. Find **dynamic** [levelwise CUDA] based pagerank of updated graph.

Each approach was attempted on a number of graphs, running each with multiple
batch sizes (`1`, `5`, `10`, `50`, ...). Each batch size was run with 5
different updates to graph, and each specific update was run 5 times for each
approach to get a good time measure. **Levelwise** pagerank is the
[STIC-D algorithm], without **ICD** optimizations. Indeed, **dynamic levelwise**
pagerank is **faster** than the *static* approach for many batch sizes. In
order to measure error, [nvGraph] pagerank is taken as a reference.

[cuda-on-fixed]: https://github.com/puzzlef/pagerank-levelwise-multi-dynamic/tree/cuda-on-fixed
[pr-nvgraph]: https://github.com/puzzlef/pagerank-nvgraph-dynamic
[monolithic CUDA]: https://github.com/puzzlef/pagerank-levelwise-cuda
[levelwise CUDA]: https://github.com/puzzlef/pagerank-levelwise-cuda
[skip-teleport]: https://github.com/puzzlef/pagerank-levelwise
[compute-5M]: https://github.com/puzzlef/pagerank-levelwise-cuda
[skip-comp]: https://github.com/puzzlef/pagerank-levelwise-dynamic
[scaled-fill]: https://github.com/puzzlef/pagerank-dynamic

<br>


### Comparing with Monolithic approach

This experiment ([compare-monolithic]) was for comparing performance between:
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


This study was carried out to extend the Levelwise strategy of PageRank
computation in the [STIC-D paper] to perform dynamic PageRank on the CPU as well
as the GPU. This levelwise computation of PageRank is computationally more
efficient because it processes SCCs in topological order, avoiding unnecessary
recomputation of SCCs that are dependent upon ranks of vertices in other SCCs
which have not yet converged. We have compared it to our monolithic CPU/GPU
implementations, along with other state-of-the art implementations, such as
nvGraph and HyPR in batch sizes of 500, 1000, 2000, 5000, and 10000 edges. We
also contrasted the performance of a batched update to a series of single edge
updates (cumulative).

Results indicate that Levelwise approach is in general faster than Monolithic
PageRank on the CPU, and the opposite is true on the GPU. This likely has to do
with the fact that processing a large number of small levels is inefficient.
Hence with Levelwise PageRank, smaller levels/components should be combined and
processed at a time in order to help improve GPU usage efficiency.

![](https://i.imgur.com/P1mlTU5.png)
![](https://i.imgur.com/1XdHMjw.png)

[compare-monolithic]: https://github.com/puzzlef/pagerank-levelwise-multi-dynamic/tree/compare-monolithic

<br>


### Other experiments

- [approach-combine-levels](https://github.com/puzzlef/pagerank-levelwise-multi-dynamic/tree/approach-combine-levels)
- [adjust-batch-size](https://github.com/puzzlef/pagerank-levelwise-multi-dynamic/tree/adjust-batch-size)
- [kitchen-sink](https://github.com/puzzlef/pagerank-levelwise-multi-dynamic/tree/kitchen-sink)

<br>
<br>


## References

- [STIC-D: algorithmic techniques for efficient parallel pagerank computation on real-world graphs][STIC-D paper]
- [PageRank Algorithm, Mining massive Datasets (CS246), Stanford University][this lecture]
- [SuiteSparse Matrix Collection]

<br>
<br>


[![](https://i.imgur.com/89cRRdY.jpg)](https://www.youtube.com/watch?v=iMdq5_5eib0)
![](https://ga-beacon.deno.dev/G-KD28SG54JQ:hbAybl6nQFOtmVxW4if3xw/github.com/puzzlef/pagerank-levelwise-multi-dynamic)

[Prof. Dip Sankar Banerjee]: https://sites.google.com/site/dipsankarban/
[Prof. Kishore Kothapalli]: https://www.iiit.ac.in/people/faculty/kkishore/
[SuiteSparse Matrix Collection]: https://suitesparse-collection-website.herokuapp.com
[STIC-D paper]: https://gist.github.com/wolfram77/bb09968cc0e592583c4b180243697d5a
[STIC-D PageRank]: https://gist.github.com/wolfram77/bb09968cc0e592583c4b180243697d5a
[HyPR]: https://gist.github.com/wolfram77/50224c1bf5585a719b1c87113e95d074
[nvGraph]: https://github.com/rapidsai/nvgraph
[this lecture]: https://www.youtube.com/watch?v=ke9g8hB0MEo
[puzzlef]: https://puzzlef.github.io
