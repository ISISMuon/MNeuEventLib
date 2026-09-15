# MNeuEventLib
MNeuEventLib is an event data filtering library for ISIS Neutron and Muon Source.

## Install from PyPI
MNeuEventLib can be installed as a pre-compiled binary [from PyPI](https://pypi.org/project/MNeuEventLib/) using
```
pip install MNeuEventLib
```


## Build from source
MNeuEventLib is built with [`maturin`](https://www.maturin.rs/). Compilation also requires the [`rustup` toolchain](https://rustup.rs/).

To compile locally, set up a Python virtual environment with `maturin` installed and run
```
maturin develop --release
```

## CI/testing
This package does unit testing, linting, and formatting using `cargo`.
The Github Actions tests check there are no `cargo clippy` suggestions,
and that the code has had `cargo fmt` applied.

To run the tests:
```
cargo test
```

To lint:
```
cargo clippy --all-targets --all-features -- -D warnings
```

To format:
```
cargo fmt
```
