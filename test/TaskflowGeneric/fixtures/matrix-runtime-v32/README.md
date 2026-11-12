These fixtures retain actual extracted AIE and its source-bound selected-graph runtime contract. `constant-feed` is configuration 0 of 64x64 BF16 GEMM + FP32 ReLU, tile 16x16x64, after the AIR launch requests `air.preserve_shim_dma_order`. Its repeated A feed remains a constant two-iteration loop. `circuit-relu` is the last, scalar ReLU-only configuration of the 96x96 tile16x32x48 program.

The runtime realizer independently checks exact selected host addresses, occurrence counts, storage/dtype, direct routing and core DMA FIFO order. It reconstructs a flat runtime and checks it again. This regression neither compiles external kernels nor executes hardware, and does not certify core arithmetic or its full lock protocol. Source/graph identities are preserved byte-for-byte in the checked contracts and IR. File hashes and original producer paths are in `provenance.json`; those original paths are provenance only and are unnecessary to run the test.

Run `check-taskflow-matrix-runtime-static` in an AIR-enabled CPU build. The Python driver extracts SSA IDs from the fixtures and requires the expected runtime-guard diagnostic for every negative; parser failures do not count as successful rejection.

The original constant two-iteration feed is now a known unsafe configuration
shape after the r7 64x64 hardware timeout and output-lock lifetime audit. The
driver requires its explicit repeated-input-region rejection; it is no longer
a positive execution certificate. Dynamic/IV-dependent/side-effect variants
must still reach their specific finite-inventory guards, not parser failures.
The actual fixture and selected contract remain unchanged as failure evidence.
