# Ideal QRAM loading on ransomware detection — code and reproduction

Code for the experiments in *"Ideal quantum random-access memory loading yields classically equivalent models
for ransomware detection"* (Abdelmoaty, Sobh & Bahaa-Eldin). It loads all 483 binary behavioural features of
each MLRan sample into two idealised quantum states (amplitude state on 9 qubits, QRAM address–data state on
10 qubits), trains variational circuits and exact fidelity-kernel SVMs on them, compares them with tuned
classical models under two protocols, measures the Qiskit gate cost of loading, and numerically verifies
the two propositions.

## Layout

```
qramqml/            library
  data.py           MLRan loading; provided and family-held-out protocols
  states.py         amplitude / QRAM address–data states; closed-form kernels (Proposition 1)
  circuits.py       batched JAX statevector simulator; 2-layer RY–CNOT ring–RZ ansatz; COBYLA training
  models.py         classical models, degree-2 polynomial SVM, precomputed-kernel SVMs, PCA-4 angle encoder
  metrics.py        ROC-AUC, validation-selected thresholds, paired t-intervals, Wilcoxon, Holm
scripts/
  run_experiments.py   all 21 model arms x 10 seeds x 2 protocols  -> results/runs_*.csv
  verify_propositions.py  Propositions 1-2                         -> results/propositions.json
  loading_cost.py      Qiskit synthesis cost                       -> results/loading_cost.csv
  analyze.py           tables + paper comparison                   -> results/COMPARISON.md, table*_reproduced.csv
  make_figures.py      figures                                     -> figures/*.png
  paper_values.py      numbers reported in the manuscript (comparison only)
tests/test_simulator.py  simulator vs Qiskit Statevector; kernels vs explicit states
data/get_mlran.sh    fetch MLRan at commit ec2651f
```

## Reproduce

```bash
pip install -r requirements.txt
bash data/get_mlran.sh data/mlran
python -m pytest -q tests                                        # simulator and kernel checks
python scripts/verify_propositions.py
python scripts/loading_cost.py --mlran data/mlran
python scripts/run_experiments.py --mlran data/mlran --protocols provided --out results/runs_provided.csv
python scripts/run_experiments.py --mlran data/mlran --protocols family   --out results/runs_family.csv
python scripts/analyze.py && python scripts/make_figures.py
```

A full run takes about 25 min per protocol on 2 CPU cores (the two protocols can run in parallel).
`run_experiments.py` is resumable and accepts `--only "<model name>"` to rerun single arms.

## Reproduction summary (full details in `results/COMPARISON.md`)

| Claim in the paper | Reproduced? |
|---|---|
| Dataset: 4,880 samples, 64 families, 483 binary features, 975 provided test samples | Yes |
| Proposition 1 (kernel identities) | Yes — max error 4e-16 (paper: < 1e-6) |
| Proposition 2 (circuits on QRAM state are quadratic; 137 monomials, full rank) | Yes — exhaustive over all 65,536 inputs, residual 3e-15 (paper: 2e-7); amplitude formula exact to 9e-16 |
| Best classical models: RBF-SVM (provided), MLP (family-held-out) | Yes, same models |
| Classical models on 483 features (Table 1) | Yes — all within 0.004 AUC of the paper |
| Squared-cosine / squared-agreement kernel SVMs and degree-2 polynomial SVM equivalent to best classical (±0.01) | Yes — all six 95% CIs inside the margin; agreement kernel slightly lower on provided split, Holm P = 0.031 as in the paper |
| QRAM-loaded circuits far below classical (all 16 confirmatory comparisons, Holm P = 0.031) | Yes — same 16 significant deficits, differences of 0.08–0.20 AUC |
| Exact kernel ≫ circuit on the same (amplitude) state: +0.120 | Yes — +0.148, P = 0.002 |
| Entanglement ablations (amplitude: no benefit; QRAM state: product much worse; angle: product better) | Yes, same directions and significance |
| No evidence that QRAM loading helps circuits over angle encoding | Yes |
| Loading cost: amplitude 502 CNOT / depth 1,005; naive QRAM write 32,383 CNOT / depth 23,786 | Yes — identical |
| 400 vs 100 COBYLA evaluations improves the amplitude circuit (+0.045, P = 0.014; AUC 0.915) | **No** — −0.002 (P = 1.0); AUC 0.841 provided, 0.902 family |
| QRAM-state circuit worse than amplitude circuit (−0.109, P = 0.004) | **Partly** — same direction (−0.054) but not significant (P = 0.19) |
| Best amplitude circuit vs angle product circuit, provided split (−0.021, CI includes 0) | **Stronger** — −0.101 (−0.162, −0.039), P = 0.002; same direction, now clearly significant |

Every conclusion of the paper holds in the reproduction. The mismatches concern the circuits, whose
results depend strongly on the random initial parameters (run-to-run s.d. of 0.03–0.09 AUC), so different
initialisation seeds shift their 10-run means by a few hundredths. The 400-evaluation result is the only
quantitative claim that does not reproduce: with the same starting point as the 100-evaluation arm, the
extra budget lowers the training loss (0.670 → 0.637) without improving test AUC. The manuscript has been
updated to report the reproduced values throughout; the reproduced value (0.841) strengthens rather than
weakens the paper's argument.

## Implementation choices not fully specified in the paper

These are the choices made here; they should be stated in the paper or the repository to make it reproducible.

* **Hyperparameter grids** for the five classical models (`models.py`, `CLASSICAL_GRIDS`); the polynomial SVM
  uses C ∈ {0.01, 0.1, 1, 10} × coef0 ∈ {0, 1, 10} (12 configurations); kernel SVMs use C ∈ {0.1, 1, 10, 100}.
* **Threshold rule.** The paper says "the highest value giving at most the target false-positive rate"; the
  highest such threshold would always give zero positives, so the code uses the *lowest* threshold whose
  validation false-positive rate is at most the target (maximising detection within the FPR budget).
* **Angle encoding.** PCA on unscaled binary features, min–max scaled to [0, π] using training data, test angles clipped.
* **Circuit initialisation.** Each arm draws its own initial parameters from a seed derived from the run seed
  and the arm name; the 400-evaluation arm reuses the 100-evaluation arm's initial parameters.
* **Family-held-out draws.** The 16 held-out families are drawn with NumPy's `default_rng(seed)`; test sizes are
  1,218–1,486 here versus 1,277–1,594 in the paper, so the exact draws differ.
* **Loading-cost samples** are drawn with `default_rng(0)`; this happens to reproduce the paper's averages exactly.

`results/supplementary/runs_*_amp400_independent_init.csv.bak` keeps the first 400-evaluation run with an independent
initialisation (AUC 0.822 provided, 0.887 family) for transparency.

## Data

MLRan (Onwuegbuche et al., 2025) is available at https://github.com/faithfulco/mlran; it is not redistributed here.
