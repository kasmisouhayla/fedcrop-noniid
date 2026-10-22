# When Farms Grow Different Crops — code and results

Code and results for the paper *“When Farms Grow Different Crops: An Empirical Study of Aggregation, Normalization and Privacy in Federated Crop Recommendation”* (S. Kasmi, S. Chadli, M. Saber), submitted to AISA'27.

The study compares FedAvg, FedProx, SCAFFOLD, FedBN and FedAvg with local fine-tuning, together with local and centralized baselines, on two agricultural tasks:

- **Crop recommendation** (22 crops) with simulated Dirichlet label skew, per-farm sensor shift and 10 or 50 farms;
- **Yield prediction** naturally partitioned across 78 countries (FAO data).

It also evaluates record-level differential privacy (DP-SGD with a Rényi accountant). Everything is implemented in plain NumPy and runs on a CPU.

## Repository layout

```
src/fl.py               FL simulator for classification (MLP, BN/LN, FedAvg, FedProx, SCAFFOLD, FedBN, fine-tuning, DP-SGD)
src/fl_reg.py           Same simulator for regression (yield prediction, natural country partition)
experiments/run_main.py Crop recommendation: label skew, local epochs, DP, sensor shift, 50 farms (parts A and B)
experiments/run_extra.py Yield prediction, fine-tuning baseline, DP with SCAFFOLD (parts A and B)
analysis/merge_results.py Merges the raw outputs
analysis/summary.py     Prints the paper's tables and statistical tests
analysis/make_figures.py Regenerates the figures
results/                Result files used in the paper
figures/                Figures of the paper (PDF)
data/                   Download instructions for the two public datasets
```

## Reproducing the results

```bash
pip install -r requirements.txt
# 1. download the two CSV files into data/ (see data/README.md)
# 2. run the experiments (about 15 minutes per part on one CPU core; the parts can run in parallel)
python experiments/run_main.py A & python experiments/run_main.py B
python experiments/run_extra.py A & python experiments/run_extra.py B
wait
# 3. merge, print tables and statistics, redraw figures
python analysis/merge_results.py
python analysis/summary.py
python analysis/make_figures.py
```

For a quick test, set `SEEDS=1` (e.g. `SEEDS=1 python experiments/run_main.py A`). The result files shipped in `results/` let you run `analysis/summary.py` and `analysis/make_figures.py` without rerunning the experiments.

## Protocol

Ten seeds per configuration (each seed redraws the partition and the initialization); hyper-parameters were tuned on two separate seeds; differences are tested with the two-sided Wilcoxon signed-rank test paired by seed, with Holm correction.

## Citation

If you use this code, please cite the paper (reference to be added after publication).

## License

MIT (see `LICENSE`). The datasets keep their original licenses and are not redistributed.
