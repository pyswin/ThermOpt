# ATMPlace WL-only Summary

Reference: ATMPlace arXiv 2511.17319 Table VI, ATMPlace WL-driven TWL/m.

| Case | Method | Ours TWL/m | Paper ATMPlace TWL/m | Delta % | Runtime/s | Selected phase |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| Case1 | atplace | 11.606 | 11.973 | -3.1% | 9.0 | clump_milp |
| Case1 | atmplace | 11.606 | 11.973 | -3.1% | 7.2 | milp_seed |
| Case2 | atplace | 16.868 | 15.408 | 9.5% | 9.9 | clump_milp |
| Case2 | atmplace | 16.868 | 15.408 | 9.5% | 9.8 | milp_seed |
| Case3 | atplace | 31.366 | 32.350 | -3.0% | 25.9 | clump_milp |
| Case3 | atmplace | 31.366 | 32.350 | -3.0% | 25.5 | milp_seed |
| Case4 | atmplace | 167.184 | 46.982 | 255.8% | 24.3 | legalization |
| Case5 | atmplace | 68.595 | 48.265 | 42.1% | 24.2 | legalization |
| Case6 | atmplace | 66.437 | 29.482 | 125.3% | 22.5 | legalization |
| Case7 | atmplace | 16.832 | 12.523 | 34.4% | 16.5 | legalization |
| Case8 | atmplace | 25.669 | 9.001 | 185.2% | 20.4 | legalization |

Notes:

- Case1-3: `atmplace` reuses the MILP seed on small cases, so it matches `atplace`.
- Case4-8: current `atmplace` uses a fast spectral/clump seed instead of full pairwise MILP; it is faster but not yet consistently competitive.
- Thermal and warpage objectives are intentionally disabled in this WL-only run.
- ATPlace_pub core placement code is PyArmor encrypted, so this repo keeps a readable in-house implementation and treats the official package as an external baseline runner.
