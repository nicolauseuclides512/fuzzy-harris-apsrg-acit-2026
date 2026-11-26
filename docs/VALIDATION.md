# Local reproduction validation

A research-team local run used a clean Python 3.14.6 virtual environment and the 21 exact package versions in requirements-historical.txt. The report and accompanying logs were reviewed; experiments were not independently repeated by an external reviewer.

| Check | Recorded outcome |
|---|---|
| Isolated package and core AST/token equivalence | PASS |
| Clean environment, install, dependency check, compile and import | PASS |
| Unit tests | 45 passed |
| Statistics | 56 contrast-metric rows; four primary tests; CSV comparisons PASS |
| DRIVE working files | 100 SHA-256 fingerprint checks PASS |
| Ablation | 180 observations; CSV comparison PASS |
| Sensitivity | 260 observations; 13 scenarios and 16 shared summary columns compared |
| New CPU timing | 60 timed executions; inventory and prediction checks PASS |
| Qualitative and prediction | 20 training failure rows, 20 test disagreement rows, 40 local masks; comparisons PASS |

All 17 recorded steps passed. Six table comparisons reported 2,955 compared cells (including the repeated primary subset), with numerical tolerances of 1e-9 relative and 1e-10 absolute. Evidence was the local report and logs, not new raw CSVs or images inspected independently. New timing uses a two-method schedule; no equality of wall-clock times is claimed. Test predictions were not scored against test ground truth.

A pandas deprecation warning for DataFrame.set_index(verify_integrity=True) was observed; the run passed with pandas 3.0.5. Source remains unchanged.

Documentation and the historical version list were added after the tested run without modifying its scientific artifacts. Independent replication, other platforms and future dependency compatibility remain unverified.
