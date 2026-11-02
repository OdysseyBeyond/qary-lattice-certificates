# Numerical verification

These programs verify numerical inequalities used in Appendices F and G.
They use Python 3 and its standard library; no external packages are required.
The verification environment used Python 3.12.14.
Run without `-O`, `-OO`, or `PYTHONOPTIMIZE`: the verification uses assertions.
Keep all Python files in this directory together, because the programs import
the shared interval arithmetic from one another.

The programs check the numerical parts of the proofs. Their connection to the
lattice algorithms, the analytical reductions, and the elementary branches
are proved in the manuscript. The lower-bound programs concern the parameter
families defined there; they do not establish lower bounds for arbitrary
SVP or CVP algorithms.

## Upper bounds used by the main results

From the archive root (which contains `scripts/` and `results/`), run:

```sh
mkdir -p results/reproduced
python3 scripts/parameter_selection.py --certify > results/reproduced/parameter_certificate.json
python3 scripts/cvp_density_extension.py --output results/reproduced/cvp_density_extension.json
python3 scripts/svp_adaptive_upper.py --output results/reproduced/svp_adaptive_upper.json
python3 scripts/svp_density_extension.py --output results/reproduced/svp_density_extension.json
```

Using `results/reproduced/` preserves the recorded JSON files in `results/`. The last three commands also print progress or a summary.

| Program | Manuscript statement | Expected bounds and coverage |
| --- | --- | --- |
| `parameter_selection.py --certify` | Appendix F, `sec:cvp-feasible-parameters`; parameters `eq:cvp-selected-parameters` | Representation time < 0.26675191, space < 0.262740033; exact-search time < 0.266706; all selected-parameter inequalities |
| `cvp_density_extension.py` | `lem:cvp-high-density-certificate` | Complete time < 0.56 and space < 0.325 on density [1, 3/2]; 829 visited and 415 accepted intervals |
| `svp_adaptive_upper.py` | `lem:svp-adaptive-certificate` | Complete time < 0.27400949 and space < 0.261392024 on density [0, 1], including the analytical exact-search branch below 0.54; 2139 visited and 1070 accepted intervals on [0.54, 1] |
| `svp_density_extension.py` | `lem:svp-density-certificate`, `tab:svp-density-certificate-results` | Six profiles, each verified on [1, D], as listed below |

The six SVP profiles have the following expected outputs. The displayed cost
bounds are rounded upward, as in the manuscript.

| D | Time upper bound | Space upper bound | Accepted intervals |
| --- | --- | --- | --- |
| 1.05 | 0.285998580 | 0.266611111 | 335 |
| 1.10 | 0.297998391 | 0.274778021 | 342 |
| 1.20 | 0.325997508 | 0.284999867 | 351 |
| 1.30 | 0.360999854 | 0.306587328 | 342 |
| 1.40 | 0.407979218 | 0.317854425 | 231 |
| 1.50 | 0.467999492 | 0.328474928 | 223 |

For the selected CVP parameters, the rigorous JSON fields are
`selected_exact_certificate` and `selected_representation_certificate`.
The same output also contains floating-point exploratory and historical
comparisons; those fields are not certificates. The certified intervals are
printed as outward-rounded decimal endpoints.

For the density-interval programs, `time_upper`, `space_upper`, and interval
endpoints are exact rational strings. Fields ending in `_decimal` are only
convenient floating-point displays. Check `partition.exact_coverage_verified`
and the positive `minimum_feasibility_margins` in every certificate. The
programs reconstruct the full subdivision and verify exact endpoint adjacency;
the JSON records its endpoints and summary, rather than every leaf. Rerunning
the programs regenerates and checks the covering partition.

The CVP density certificate checks the entropy surplus and the shifted-theta
condition. The SVP certificates include the stated rounding window, padding
margin, and retained-coordinate margin. Point queries such as
`cvp_density_extension.py --density 1.2` do not replace the full interval run.

## Additional lower bounds for the parameter optimization

These runs establish the additional near-optimality statements. They are
separate from the four upper-bound checks above and are more expensive.

```sh
python3 scripts/global_parameter_certificate.py --output results/reproduced/global_parameter_certificate.json
python3 scripts/svp_global_lower.py --threshold .2740094 --max-boxes 1000000 --output results/reproduced/svp_global_lower_certificate.json
```

| Program | Manuscript statement | Expected result |
| --- | --- | --- |
| `global_parameter_certificate.py` | `lem:cvp-numerical-lower`, `eq:cvp-appendix-global-bounds` | Lower bound 0.26675186; 17,053 splits and 17,054 leaf rectangles, with an exact coverage check |
| `svp_global_lower.py` | `sec:svp-global-certificate`, `thm:svp-global-interval` | Lower bound 0.2740094; `success: true`, `pending: 0`, and 454,329 visited boxes |

The SVP lower-bound program writes a partial result if the box limit is
reached. A zero process exit code alone therefore does not establish the
lower bound. Check completion explicitly:

```sh
python3 -c "import json; r=json.load(open('results/reproduced/svp_global_lower_certificate.json')); assert r['success'] and r['pending'] == 0"
```

Floating-point calculations in these lower-bound programs propose brackets
or choose the subdivision order. Acceptance uses outward rational bounds.
The analytical domain reductions and local convexity arguments are supplied
in Appendices F and G.

## Arithmetic and supporting files

`parameter_selection.py` provides Fraction-based interval arithmetic rounded
outward to a grid of spacing 10^-50, with explicit logarithm, exponential,
and arctangent remainder bounds. `global_parameter_certificate.py` also
provides integer interval arithmetic on a grid of spacing 10^-35. The SVP
and high-density CVP programs reuse these operations; their theta tails
are bounded by geometric series as specified in the manuscript.

`svp_global_upper.py` supplies shared SVP interval functions.
`svp_parameter_selection.py` supplies theta-law enclosures used by the SVP
lower-bound program. Their default standalone runs concern other feasible
witnesses and are not substitutes for `svp_adaptive_upper.py` in the current
selected-parameter check.

Search and plotting programs, and outputs produced with `--search` or
`--depths`, assist parameter selection or illustrate formulas. They do not
certify a continuous density interval or a global infimum.

## File integrity and comparison with recorded runs

The archive's `SHA256SUMS` identifies the supplied source files and recorded
outputs. Verify it from the archive root using the standard library:

```sh
python3 - <<'PY'
import hashlib
from pathlib import Path
for line in Path('SHA256SUMS').read_text().splitlines():
    expected, name = line.split('  ', 1)
    actual = hashlib.sha256(Path(name).read_bytes()).hexdigest()
    if actual != expected:
        raise SystemExit('SHA-256 mismatch: ' + name)
print('All listed SHA-256 hashes match.')
PY
```

For rerun outputs, compare exact rational bounds, constraint margins, and
coverage results. Elapsed-time fields and convenient floating-point displays
are not certificate decisions and need not match byte for byte.
