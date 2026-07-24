# NADES-ML: Interactive Machine Learning Workbench
### Natural Deep Eutectic Solvents & Organometallic Drug Discovery Platform

This platform bridges computational chemistry and advanced machine learning to streamline drug discovery. By analyzing the complex interaction profiles between Natural Deep Eutectic Solvents (NADES) and organometallic compounds, the suite automates multi-objective virtual screening to isolate viable therapeutic candidates.

---

## Platform Capabilities

* Engineered Feature Crosses: Computes thermodynamic estimates (Excess Gibbs Energy, Hansen Solubility Parameter distance) and metal-specific d-electron group configurations.
* 15+ Benchmarked ML Models: Features systematic cross-validation evaluating standard linear models, advanced gradient boosters (XGBoost, LightGBM, CatBoost), and custom architectures.
* Custom Deep Learning Network: Integrates a PyTorch multi-task deep neural network boosted by a custom feature-level Self-Attention mechanism.
* Automated Multi-Objective Screening: Ranks candidates against a composite pricing, biocompatibility, and drugability score to extract top therapeutic configurations.
* Explainable AI Framework (XAI): Implements localized and global explainability layers using SHAP and LIME diagnostics to map model validation transparently.


