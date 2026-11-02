# Online Extra Trees for Data Streams: An Empirical Study

Does extreme randomization make streaming tree ensembles faster and lighter without hurting accuracy? This project benchmarks the **Online Extra Trees (OXT)** regressor ([Mastelini et al., 2023](https://doi.org/10.1109/tnnls.2022.3212859)) against **Adaptive Random Forest (ARF)** and a **Hoeffding Tree Regressor (HTR)**. It uses synthetic streams with concept drift and a real-world stream.

> Course project for *Data Stream Mining*, MSc in Artificial Intelligence, Faculty of Engineering of the University of Porto (FEUP), 2026. By António Romão and Bruno Moreira.

## What we did

- **Custom prequential (test-then-train) evaluator.** It tracks cumulative and rolling MAE/RMSE, throughput (instances per second) and memory. Memory estimates are corrected with `tracemalloc`, because pickle size underestimated ARF by about 5×.
- **Fair tuning.** A symmetric random search gave each model the same budget of 50 configurations. A Pareto-style score penalized memory use and runtime, mirroring real streaming constraints.
- **Four streams:**
  - Friedman, 100k instances, in three versions: stationary, abrupt drift and gradual drift.
  - Bike Sharing, a real-world stream of 182,470 instances.

## Key results

Cumulative RMSE (lower is better):

| Stream | HTR | OXT | ARF |
|---|---|---|---|
| Friedman: stationary | 2.126 | 2.319 | **1.821** |
| Friedman: abrupt drift | 2.265 | 2.383 | **1.957** |
| Friedman: gradual drift | 2.249 | 2.402 | **1.944** |
| Bike Sharing (real-world) | 7.753 | **7.413** | 7.680 |

Throughput (instances per second):

| Stream | HTR | OXT | ARF |
|---|---|---|---|
| Friedman | 17,051 | 4,186 | 3,831 |
| Bike Sharing | 24,628 | 8,581 | 8,797 |

**Takeaways**

- **Real-world data:** OXT generalizes best on the noisy stream, where its randomization works as a built-in regularizer.
- **Synthetic data:** OXT trails both baselines on the clean Friedman functions.
- **No efficiency gain:** at equal ensemble size, OXT is no faster than ARF and has a heavier, more volatile memory footprint.
- **Concept drift:** ARF recovers fastest thanks to its background trees. OXT recovers more slowly, at about the same pace as the single Hoeffding tree.

## Reproduce

Tested with Python 3.11.

```bash
python3 -m venv dsm_env && source dsm_env/bin/activate
pip install -r requirements.txt
python -m ipykernel install --user --name=dsm_env --display-name "Python (DSM OXT)"
jupyter notebook OXT.ipynb
```

The hyperparameter search takes several minutes. Its results are cached in `checkpoints/`, so the notebook can load them instead of re-running it.

## Repository structure

```
├── OXT.ipynb                     # Full study: setup, tuning, evaluation, analysis
├── checkpoints/                  # Cached hyperparameter-search results
├── OXT_Presentation.pptx         # Presentation slides
├── Practical_Assignment_1-...pdf # Assignment brief
└── requirements.txt
```

**Tech:** Python · River · pandas · NumPy · SciPy · Matplotlib

## References

- Mastelini, S. M., Nakano, F. K., Vens, C., & de Carvalho, A. C. P. L. F. (2023). Online Extra Trees Regressor. *IEEE TNNLS*, 34, 6755–6767.
- Ikonomovska, E., Gama, J., & Džeroski, S. (2010). Learning model trees from evolving data streams. *Data Mining and Knowledge Discovery*, 23, 128–168.
- [River: machine learning for streaming data in Python](https://riverml.xyz)
