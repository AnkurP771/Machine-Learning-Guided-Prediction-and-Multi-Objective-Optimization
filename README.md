# Machine Learning-Guided Prediction and Multi-Objective Optimisation of Furfural Hydrogenation to Furfuryl Alcohol (FOL) and Tetrahydrofurfuryl Alcohol (THFA)

B.Tech Project (BTP), Department of Chemical Engineering, **IIT Ropar**
**Author:** Ankur Prasad 

---

## Overview

Furfural (FFR) is a renewable platform chemical obtained from lignocellulosic biomass (corncob, bagasse). Its catalytic hydrogenation gives:

- **Furfuryl alcohol (FOL):** selective reduction of the C=O group
- **Tetrahydrofurfuryl alcohol (THFA):** further hydrogenation of the furan ring

Conversion and product selectivity depend on the metal, support, solvent and operating conditions, and the literature data are scattered across many catalyst systems. This project builds a data-driven workflow that:

1. Curates literature datasets and enriches them with physicochemical descriptors
2. Trains and benchmarks regression models for **Conversion (%)** and **Selectivity (%)**
3. Explains the models with **SHAP**
4. Runs **multi-objective optimisation** (NSGA-II and SPEA2) over the trained models
5. Checks how far the optimiser's candidates can be trusted (independent-model agreement, applicability domain)

> The optimisation outputs are **model-generated candidates, not experimentally validated optima.**

---

## Pipeline

```
Literature data -> Descriptor enrichment -> Cleaning / feature engineering
   -> Regression (tuning, stacking, NNLS blend) -> SHAP
   -> NSGA-II / SPEA2 (conventional and SHAP-guided) -> Credibility checks
   -> Full 41-column reconstruction of Pareto candidates
```

### Datasets

| Dataset | Experiments | Source papers | Columns |
|---|---|---|---|
| FFR -> FOL | 436 | multiple literature sources | 41 |
| FFR -> THFA | 175 | 15 | 41 |

Where a paper gave results only as a graph, values were digitised with GetData Graph Digitizer.

**Descriptors added (novelty of this work)**

- **Metal (up to 2 per catalyst):** Pauling electronegativity, atomic radius, electron affinity, first ionisation energy, d-electron count, d-band filling
- **Solvent:** boiling point, density, dipole moment, dielectric constant, viscosity, relative polarity, H-bond donor/acceptor
- **Support:** pore volume, total acidity (mmol NH3/g)

**Preprocessing:** reaction-time fractions converted to hours; structural gaps (e.g. second metal of monometallic catalysts) zero-filled; unreported values median-imputed (THFA); redundant categorical columns dropped; |rho| > 0.9 descriptor pairs removed; 80/20 train/test split with a fixed seed.

---

## Methods

- **Models compared:** Ridge, Random Forest, Extra Trees, Gradient Boosting, XGBoost, LightGBM, CatBoost, SVR, Kernel Ridge, Gaussian Process, Bayesian Ridge, KNN, PLS, MLP/Keras networks, TabPFN, FLAML AutoML
- **Tuning:** RandomizedSearchCV, GridSearchCV, Optuna (TPE)
- **Feature work:** interaction/ratio terms, squared and log1p terms ("v2" features), PCA (95% variance), RFECV, logit target transform
- **Ensembling:** Ridge stacking, "Mega Stacked", **NNLS-weighted blend** across feature spaces
- **Other:** Isolation Forest (outlier diagnostics on training data only), repeated K-fold CV, learning curves, mutual information
- **Interpretability:** SHAP (beeswarm, mean |SHAP| bars, dependence, waterfall, heatmaps)
- **Optimisation (pymoo):** NSGA-II and SPEA2, multiple seeds, compared by hypervolume (HV), IGD+, front size and runtime

### Two optimisation formulations

1. **Conventional:** continuous operating variables plus catalogue indices for catalyst and medium (only combinations present in the data)
2. **SHAP-guided:** only raw variables behind features with mean |SHAP| > 1 are varied (FOL: 14, THFA: 16), bounded by the 1st-99th percentile; all other variables are fixed at a reference experiment. A third objective (minimise reaction temperature) is added. For THFA this is the three-objective problem.

---

## Results

### Prediction (held-out test set)

| System | Target | Best model | R2 | MAE | RMSE |
|---|---|---|---|---|---|
| FOL | Conversion | CatBoost + v2 features | 0.6091 | 13.16 | 19.08 |
| FOL | Selectivity | NNLS ensemble | 0.7595 | 11.18 | 17.00 |
| THFA | Conversion | Extra Trees | 0.5483 | 10.91 | 18.67 |
| THFA | Selectivity | Gaussian Process | 0.3502 | 17.82 | 24.71 |

Notes:
- FOL conversion models cluster at R2 ~ 0.57-0.61 regardless of algorithm, which points to a data-quality ceiling in pooled literature data.
- THFA selectivity SHAP uses CatBoost (test R2 = 0.2611) because Gaussian Process is not compatible with `TreeExplainer`.
- Repeated 5-fold CV for CatBoost + v2 features on THFA: R2 = 0.6122 +/- 0.1244 (conversion), 0.6116 +/- 0.1378 (selectivity).

### Key SHAP findings

- **FOL selectivity:** reaction temperature dominates; higher temperature lowers selectivity (consistent with ring over-hydrogenation). Solvent dipole moment, surface area and reaction time are secondary.
- **FOL conversion:** diffuse and interaction-driven (reaction time, calcination x reaction temperature, pressure, reduction temperature).
- **THFA:** reaction time is the top feature for both targets.
- SHAP shows model behaviour, **not causality**.

### Multi-objective optimisation

| Study | Algorithm | Mean HV | Mean IGD+ | Front size | Runtime (s) |
|---|---|---|---|---|---|
| FOL conventional | NSGA-II | 9990.67 | 1.7354 | 100 | 31.9 |
| FOL conventional | SPEA2 | 9989.58 | 1.8080 | 100 | 35.3 |
| FOL SHAP-guided | NSGA-II | 99.07 % of ideal | 0.1049 | 100 | 19.8 |
| FOL SHAP-guided | SPEA2 | 99.67 % of ideal | 0.0579 | 100 | 14.9 |
| THFA conventional | NSGA-II | 9637.42 | 0.0232 | 100 | 39.4 |
| THFA conventional | SPEA2 | 9638.20 | 0.0166 | 100 | 33.4 |
| THFA SHAP-guided (3-obj) | NSGA-II | 98.41 % of ideal | 0.2943 | 100 | 21.9 |
| THFA SHAP-guided (3-obj) | SPEA2 | 96.57 % of ideal | 0.5198 | 100 | 17.7 |

HV differences are mostly smaller than the seed-to-seed standard deviation, so the two algorithms are treated as comparable (NSGA-II is clearer in the THFA three-objective case).

**Representative candidates (surrogate-predicted)**

| System | Candidate | Conv (%) | Sel (%) | T (C) |
|---|---|---|---|---|
| FOL conventional | Knee (CuNi/MgAlO, methanol, 5.18 MPa, 3.13 h) | 100.0 | 100.0 | ~150 |
| FOL SHAP-guided | Knee | 96.66 | 94.84 | 30.5 |
| FOL SHAP-guided | Low-T (>=95/95) | 96.07 | 95.04 | 30.24 |
| THFA conventional | Knee (Ni1Cu1-Al1 R, 2-butanol, 3.96 MPa, 5.5 h) | 99.98 | 94.75 | ~161 |
| THFA SHAP-guided | Knee | 99.38 | 98.56 | 34.03 |
| THFA SHAP-guided | Low-T (>=95/95) | 99.26 | 100.0 | 30.08 |

The reference experiments already reached 100 % / 100 %, so the value lies in the trade-offs and alternative, milder conditions rather than in exceeding that point.

**Credibility checks** (independent-model agreement + distance to experimental data)

| Front | Pareto solutions | Passed both checks |
|---|---|---|
| FOL conventional | 400 | 399 |
| FOL SHAP-guided | 166 | 116 |
| THFA conventional | 188 | 50 |
| THFA SHAP-guided | 140 | 43 |

A mathematically non-dominated solution is not necessarily a reliable one. Many THFA candidates lie in data-sparse regions.

---

## Repository structure (suggested)

```
.
├── README.md
├── notebooks/
│   ├── BTP_Sem7_MOO_SHAP_NSGA2_SPEA2_best_solution_all_features.ipynb      # FOL pipeline
│   └── BTP_Sem7_Furfural_to_THFA_MOO_SHAP_NSGA2_SPEA2_full_features.ipynb  # THFA pipeline
├── data/                      # Furfural_Furfuryl_Alcohol_dataset_full.xlsx, Furfural_THFA_dataset_full_v2.xlsx
├── report/                    # final BTP report (PDF)
└── slides/                    # mid-sem and end-sem presentations
```

### Notebook layout (both notebooks share the same structure)

| Sections | Content |
|---|---|
| 1-12 | Load, clean, impute, encode, EDA, multicollinearity, train/test split |
| 13-22 | Baselines, RandomizedSearch/GridSearch, stacking, neural networks |
| 23-35 | RFECV, CatBoost, Extra Trees, PLS, Kernel Ridge, GP, Optuna, repeated CV, mega ensemble |
| 36-52 | Noise-floor and learning-curve diagnostics, engineered features, logit target, TabPFN, MI, matminer descriptors, FLAML |
| 53 | PCA, v2 features, outlier-aware training, **NNLS blend**, final comparison |
| 54-56 | SHAP analysis and presentation figures |
| 57 | Conventional MOO (NSGA-II vs SPEA2) |
| 58 | SHAP-guided MOO and credibility checks |
| 59-60 | Full 41-column reconstruction of Pareto solutions (FOL notebook also exports the best solution from the better algorithm) |

---

## Getting started

The notebooks were written for **Google Colab** and read data from and cache models to Google Drive (`/content/drive/MyDrive/BTP_Sem_7/`). To run locally, change the paths in the "Mount Drive" and "Load dataset" cells.

**Main dependencies**

`numpy`, `pandas`, `scipy`, `scikit-learn`, `xgboost`, `lightgbm`, `catboost`, `optuna`, `shap`, `pymoo`, `tensorflow`, `keras-tuner`, `tabpfn`, `flaml`, `matminer`, `pymatgen`, `matplotlib`, `seaborn`, `joblib`, `openpyxl`

```bash
pip install numpy pandas scipy scikit-learn xgboost lightgbm catboost optuna shap pymoo \
            tensorflow keras-tuner tabpfn flaml matminer pymatgen matplotlib seaborn joblib openpyxl
```

Run order: place the dataset `.xlsx` in the configured folder, then run the notebook top to bottom. Trained models are cached as `.pkl` files, so re-runs skip completed fits.

**Outputs written by the notebooks** include `FINAL_BEST_conversion_model.pkl`, `FINAL_BEST_selectivity_model.pkl`, `grand_summary_v4.csv`, `moo_algorithm_comparison.csv`, `moo_pareto_solutions.xlsx`, `moo_shap_pareto_solutions.xlsx` and the full-feature Pareto tables.

---

## Limitations

- Small, heterogeneous literature datasets (different preparation, reactors, analytics)
- Median-imputed and zero-filled values carry different meanings
- Optimisers can exploit surrogate-model errors, especially in sparse regions
- Distance-based applicability-domain checks do not guarantee chemical feasibility
- No experimental validation of the optimised candidates yet

## Future work

Experimental validation of a few credible compromise points; larger and better-documented datasets; uncertainty-aware multi-objective optimisation; additional chemical and process constraints; active learning to choose the next experiments; extension to other biomass-derived platform molecules.

---

## Acknowledgement

Supervised by **Dr. Arghya Banerjee**, Computational Catalysis and Sustainable Chemicals Lab, Department of Chemical Engineering, IIT Ropar.

## License

Add a license of your choice (e.g. MIT) before publishing.

## Citation

```
A. Prasad, "Machine Learning-Guided Prediction and Multi-Objective Optimisation of Furfural
Hydrogenation to Furfuryl Alcohol and Tetrahydrofurfuryl Alcohol," B.Tech Project Report,
Department of Chemical Engineering, IIT Ropar, 2026.
```
