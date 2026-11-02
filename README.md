# Numerical certificates for random q-ary lattices

Companion code for **Fast Provable Algorithms for Exact SVP and Approximate
CVP on Random q-ary Lattices**, by Jiseung Kim, Junhyuk Kwon, Changmin Lee,
and JeongHwan Lee.

This repository contains the numerical verification programs and recorded
certificates used in Appendices F and G. It verifies numerical inequalities
for the parameter choices and parameter families analyzed in the paper.
It does not contain implementations of the lattice algorithms themselves.

## Requirements

Python 3 with the standard library; the recorded runs and reproduction checks
used Python 3.12.14. No third-party packages are required. Run without `-O`,
`-OO`, or `PYTHONOPTIMIZE`, because the verification uses assertions.

## Reproduce the main upper-bound certificates

Run from the repository root:

```sh
mkdir -p results/reproduced
python3 scripts/parameter_selection.py --certify > results/reproduced/parameter_certificate.json
python3 scripts/cvp_density_extension.py --output results/reproduced/cvp_density_extension.json
python3 scripts/svp_adaptive_upper.py --output results/reproduced/svp_adaptive_upper.json
python3 scripts/svp_density_extension.py --output results/reproduced/svp_density_extension.json
```

See [scripts/README.md](scripts/README.md) for the exact correspondence with
the paper, expected bounds, additional lower-bound verification commands,
and instructions for checking completion and file integrity.

## Contents

- `scripts/`: eight verification and supporting Python programs, with detailed
  reproduction instructions.
- `results/`: six recorded certificate outputs from the completed runs of
  October 31, 2026.
- `SHA256SUMS`: checksums of the verification programs, detailed instructions,
  and recorded outputs in this repository.
- `README.txt`: the original archive note.

The verification programs and recorded certificate values are those supplied
with the paper. `SHA256SUMS` identifies the listed repository files; it does
not cover this repository README or `.gitignore`.

`svp_global_upper.py` is included for the shared functions imported by the
other verifiers. Its optional standalone mode requires a candidate JSON file
that is not included in this certificate archive. Use the commands above
for the upper bounds in the main results. Search and plotting programs are
not included in this repository.

The lower-bound certificates concern the parameter families defined in the
paper; they are not lower bounds for arbitrary SVP or CVP algorithms.
