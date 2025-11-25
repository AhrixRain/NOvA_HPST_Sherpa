# Heterogeneous Point Set Transformers for Segmentation of Multiple View Particle Detectors
This tutorial based on the server @tau-neutrino.ps.uci.edu 

This repository valids accessibility of [Sherpa Optimization](https://github.com/sherpa-ai/sherpa) on HPST tunining.

This repository contains code to train an HPST, and baselines like GAT and RCNN on NoVa Data for Multiple-view particle detector Segmentation.

For the basic setup for training, refer to [the original HPST](https://github.com/mrheng9/NOvA-HPST).

# Optimization

## Run Sherpa

> Note that due to sherpa maintainance statue, Bayesian Optimization is not used because numpy removes certain functions in its 2.0+ versions.

```bash
python scripts/optim.py
```

## Post-running

The results will be saved at: `base/sherpa/hpst`.
Artifacts includes: wandB logging files (folder`wandb`). results of every trial (`results.csv`), the best hyperparameter info (`sherpa_summary.json`).
