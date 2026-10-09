# V1DD input preparation profile

Compute job 24492129, node1612, 4 allocated CPUs, 16 crop readers and 16 chunk readers. Three cells, first 128 Morton-ordered nodes each; fresh node-local chunk cache, then disk and RAM reuse. All nine trials completed. Raw measurements: timings.json and shared_io.json.

| Phase | Wall seconds per 128 inputs |
|---|---:|
| Cold remote chunks | 24.23–34.10 |
| Cached disk chunks | 2.44–2.98 |
| Cached RAM chunks | 1.72–1.74 |

Remote object retrieval dominates cold waiting. Root membership masking is the largest CPU cost. Chunk decode and lock waits are smaller. Concurrent/nested phase timings overlap and must not be added as elapsed-time percentages. CloudFiles.get includes transport decompression and metadata calls; it is not a pure wire transfer measurement. Each phase creates fresh crop-reader threads, causing volume metadata requests even for warm trials; persistent production readers amortize those calls. This is a small sample without 64-worker aggregate contention.

The separate shared-storage probe read 128 image files (32 MiB) in 0.084 seconds and 128 packed-mask files (32 MiB) in 0.100 seconds. Repeated reads took 0.017–0.018 seconds. These are sampled reads, potentially cached by the filesystem, not a sustained aggregate bandwidth measurement.

Shared production writes fail with errno 122, Disk quota exceeded. Historical CPU array 24490429 failed with this same error. Therefore production disk write throughput cannot currently be measured. Node-local prepared batch writes were approximately 0.14 seconds for two arrays in warm trials; this does not measure shared scratch write throughput.

Priorities: resolve the production account quota; reduce and overlap remote object requests; optimize root membership masking; reuse volume clients across reader lifetimes. More CPU jobs alone previously improved aggregate throughput only about 15 percent.
