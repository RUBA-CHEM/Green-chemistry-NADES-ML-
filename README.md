# 🧪 NADES-ML: Interactive Machine Learning Workbench
### Natural Deep Eutectic Solvents & Organometallic Drug Discovery Platform

This platform bridges computational chemistry and advanced machine learning to streamline drug discovery. By analyzing the complex interaction profiles between Natural Deep Eutectic Solvents (NADES) and organometallic compounds, the suite automates multi-objective virtual screening to isolate viable therapeutic candidates.

---

## 🎯 Platform Capabilities

* Engineered Feature Crosses: Computes thermodynamic estimates (Excess Gibbs Energy, Hansen Solubility Parameter distance) and metal-specific d-electron group configurations.
* 15+ Benchmarked ML Models: Features systematic cross-validation evaluating standard linear models, advanced gradient boosters (XGBoost, LightGBM, CatBoost), and custom architectures.
* Custom Deep Learning Network: Integrates a PyTorch multi-task deep neural network boosted by a custom feature-level Self-Attention mechanism.
* Automated Multi-Objective Screening: Ranks candidates against a composite pricing, biocompatibility, and drugability score to extract top therapeutic configurations.
* Explainable AI Framework (XAI): Implements localized and global explainability layers using SHAP and LIME diagnostics to map model validation transparently.

---

## 🚀 Virtual Screening & Performance Analytics

The code automatically tracks performance profiles, rendering key data frameworks natively inside the project directory:

### 1. Feature Engineering & Exploratory Data Analysis
* `feature_correlation.png` — Maps linear relationships across target vectors and engineered chemical dimensions.

### 2. Multi-Model Benchmarking & Stacking Performance
* `stacking_ensemble_metrics.png` — Outlines predictive error metrics, baseline algorithms, and final meta-learner improvements.
* `nn_training_metrics.png` — Visualizes learning rates, loss functions, and convergence tracking for the PyTorch Attention network.
* `optuna_optimization.png` — Records the Bayesian parameter exploration history across dense algorithmic fields.

### 3. Trustworthy AI & Interpretability
* `shap_summary.png` / `shap_dependency.png` — Quantifies global feature impacts and isolated directional trends across chemical inputs.
* `lime_explanation.png` / `shap_waterfall.png` — Deconstructs local, case-specific decision pathways prioritizing localized synergy estimations.

### 4. Therapeutic Discoveries
* `drug_discovery_pipeline.png` — Aggregates final candidate matrix groupings, structural cluster projections (t-SNE), and high-performing chemical combinations.

---

## 🛠️ Local Environment Initialization

Replicate the runtime environment locally by deploying the project dependency checklist via your command terminal:

```bash
pip install -r requirements.txt
```
