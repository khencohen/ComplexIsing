# Implicit Binarization via Complex Phase Dynamics in Combinatorial Optimization

This repository contains the reference implementation accompanying the paper:

> **Implicit Binarization via Complex Phase Dynamics in Combinatorial Optimization**

The code reproduces the experiments reported in the paper on three families of
combinatorial-optimization problems:

1. **QUBO** — random Quadratic Unconstrained Binary Optimization instances.
2. **Sparse Coding** — a constrained QUBO instance in which the solution has a
   prescribed cardinality. Sparse coding is treated as a particular case of
   QUBO and is therefore handled by the same driver script.
3. **PQ (Planted-Solution Ising)** — Ising instances with planted ground states
   indexed by two primes `p` and `q`, taken from the dataset shipped under
   `ps_data/`.

For each problem the same continuous relaxation is optimized with four
different parametrizations of the spin variables, all of which converge to
binary configurations through the *implicit binarization* mechanism described
in the paper:

| Name        | Variable space                          |
|-------------|------------------------------------------|
| `Real`      | scalar real angle, `cos(πθ)`             |
| `Complex`   | unit complex phase, `exp(iπθ)`           |
| `Sphere`    | unit vector on the 2-sphere `S²`         |
| `Quaternion`| unit vector on the 3-sphere `S³`         |

The `Complex` parametrization is the central object of study in the paper —
the other three are baselines used for comparison.

---

## Repository layout

```
.
├── main_qubo.py             # Driver for QUBO and Sparse-Coding experiments
├── qubo_run_func.py         # Per-noise-point experiment loop (data + training)
├── qubo_train_func.py       # Per-instance training loop over initial conditions
├── qubo_network_def.py      # Axb problem generator + ComplexNet model (4 parametrizations)
├── qubo_opt_sol.py          # Brute-force optimal solver (used as a sanity check, N ≤ 32)
│
├── main_ps.py               # Driver for the PQ / Planted-Solution Ising experiments
├── ps_data/                 # Planted-solution Ising instances (J_ij, h_i, ground-state energy)
│   └── inst_p<P>_q<Q>_e<E>.txt
└── ps_params/               # Cached metadata (sorted file list, sizes, ground-state energies)
```

### QUBO / Sparse-Coding pipeline

`main_qubo.py` selects a test configuration from the `testparams` table and
calls `run_experiment` in `qubo_run_func.py`. Each test point specifies, among
other things, the problem dimensions `N × M`, the cardinality (`-1` means
unconstrained QUBO; a positive integer triggers the sparse-coding regime), the
number of independent instances (`avg`), the number of random initial
conditions per instance (`nic`), the number of training epochs, and the noise
levels (`noise_std_list`).

For every noise level the driver:

1. Generates (or loads from `datain/`) random `A`, planted message `m`, noise
   `e`, and observation `b = A m + e`.
2. Trains all four `ComplexNet` parametrizations (`qubo_network_def.py`) with
   Adam, optionally in parallel processes (`use_multi_threading`).
3. Saves per-method results under `data/test<id>/` and produces comparison
   plots under `plots/`.

For small instances (`N ≤ 32`) the brute-force optimum from `qubo_opt_sol.py`
can be enabled (`opt_sol_en = 1`) and overlaid on the plots.

### PQ / Planted-Solution pipeline

`main_ps.py` reads the Ising instances in `ps_data/`. Each file
`inst_p<P>_q<Q>_e<E>.txt` is a list of triples:

* `i j J_ij` for `i ≠ j` (couplings)
* `i i h_i`  (local fields)

The integer following `_e` in the filename is the planted ground-state energy
(e.g. `inst_p7_q11_e-70.txt` ⇒ ground-state energy `-70`).

For every enabled instance and every enabled parametrization the script runs
`ntrials` independent optimizations, records the rounded energies reached at
the half-way point and at the end of training, and (when `show_results = 1`)
produces histograms of the achieved minimum energies under `ps_res/`.

---

## Requirements

The code is pure Python and depends on:

* `numpy`
* `torch`
* `tqdm`
* `matplotlib`
* `tensorboard` (only if you re-enable the optional `SummaryWriter` calls)

A recent CPU-only PyTorch build is sufficient to reproduce all experiments in
the paper; no GPU is required.

---

## Running the experiments

### QUBO and Sparse Coding

Pick a configuration in `main_qubo.py` by editing `tests_run_list` (e.g. `[1]`
for the 160 × 160 QUBO row, `[3]`–`[6]` for sparse-coding rows of increasing
size), then run:

```bash
python main_qubo.py
```

`run_mode` controls behaviour: `0` runs only, `1` plots only from cached
results, `2` does both. Output artifacts are written to `data/`, `datain/`,
`plots/`, and `tmp/`, which are created automatically on first run.

### PQ / Planted-Solution Ising

Edit `files_enable` and `model_type_enable` at the top of `main_ps.py` to
choose which instances and which parametrizations to run, then:

```bash
python main_ps.py
```

Set `multi_proc_num > 1` to launch additional worker processes in parallel.
Energy traces are stored in `ps_res/` and successful runs that reach the
planted ground state save the recovered spin configuration to the same folder.

---

## Reference to the paper

The four parametrizations, the diagonal-regularization schedule controlled by
`reg_coeff` / `RegCoeffArray` / `diag_reg_factor`, and the two-stage training
procedure (`epochs // 2` schedule switch) implement the *implicit binarization
via complex phase dynamics* mechanism analysed in the paper. The QUBO,
Sparse-Coding and PQ experiments in this repository correspond to the
respective sections of the paper and can be used to reproduce the figures
reported there.

## Citation

@misc{cohen2026implicitbinarizationcomplexphase,
      title={Implicit Binarization via Complex Phase Dynamics in Combinatorial Optimization},
      author={Khen Cohen and Mark Glass and Meir Feder and Yaron Oz},
      year={2026},
      eprint={2605.24502},
      archivePrefix={arXiv},
      primaryClass={cond-mat.stat-mech},
      url={https://arxiv.org/abs/2605.24502}, 
}


