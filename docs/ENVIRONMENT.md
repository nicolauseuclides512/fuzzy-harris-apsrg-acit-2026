# Environment and dependencies

requirements.txt is a range-based dependency list. requirements-historical.txt records 21 exact versions from the historical environment and the successful local clean-environment run using Python 3.14.6 on Windows. This is not a wheel-hash lock. Use an isolated environment and record installed versions.

The archived CPU timing protocol records the following versions (not a complete environment lock, and not independently verified as installable on every platform):

| Component | Recorded value |
|---|---|
| Python | 3.14.6 |
| NumPy | 2.5.1 |
| pandas | 3.0.5 |
| SciPy | 1.18.0 |
| OpenCV | 5.0.0 |
| scikit-image | 0.26.0 |
| OS | Windows 11 |
| Processor | Intel64 Family 6 Model 186 Stepping 2 |
| Logical CPUs | 12 |

See RUNTIME_PROTOCOL.json for historical timing conditions. The reviewed local report and logs confirm all 21 versions matched, pip check passed, and 45 tests passed. The historical version list includes recorded transitive dependencies but does not pin the OS, wheel hashes, CPU, package index or future availability. Linux/macOS and other Python versions are unverified.

For new experiments capture:

```text
python --version
python -m pip check
python -m pip freeze
```

The runtime runner sets numerical thread environment variables before imports, sets OpenCV to one thread and disables OpenCL. This does not enforce OS CPU affinity. Record actual hardware, decoder support and execution conditions.
