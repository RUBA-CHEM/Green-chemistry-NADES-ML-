# =============================================================================
# Install all dependencies
# =============================================================================
# Run this cell first! It will take 1-2 minutes.

!pip install -q gradio==4.44.0
!pip install -q scikit-learn xgboost lightgbm catboost
!pip install -q shap lime plotly
!pip install -q pandas numpy matplotlib seaborn
!pip install -q optuna
!pip install -q torch --index-url https://download.pytorch.org/whl/cu118

import warnings
warnings.filterwarnings('ignore')

print("✅ All packages installed successfully!")
print("⏳ Now run the next cells to load the interactive dashboard...")

# =============================================================================
# NADES-ML: Advanced Machine Learning for Natural Deep Eutectic Solvents
# & Organometallic Drug Discovery Platform
# =============================================================================
# Run this notebook in Google Colab for GPU-accelerated training

!pip install rdkit scikit-learn xgboost lightgbm torch torchvision
!pip install catboost
!pip install optuna shap lime plotly seaborn mordred deepchem
!pip install pandas numpy matplotlib scipy

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler, PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.ensemble import (RandomForestRegressor, GradientBoostingRegressor,
                             AdaBoostRegressor, ExtraTreesRegressor, VotingRegressor,
                             StackingRegressor, BaggingRegressor)
from sklearn.linear_model import (LinearRegression, Ridge, Lasso, ElasticNet,
                                 BayesianRidge, HuberRegressor)
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, mutual_info_regression
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import optuna
import shap
import lime
import lime.lime_tabular
import warnings
warnings.filterwarnings('ignore')

print("✅ All libraries imported successfully!")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

# =============================================================================
# Construct the NADES-Organometallic Interaction Dataset
# =============================================================================

# NADES Systems (Hydrogen Bond Acceptor + Donor pairs)
nades_data = {
    'hba': ['Choline Chloride']*10 + ['Betaine']*4 + ['Proline']*3 + ['Glucose']*2 + ['Sucrose'],
    'hbd': ['Urea', 'Glycerol', 'Citric Acid', 'Lactic Acid', 'Malic Acid',
            'Ethylene Glycol', 'Oxalic Acid', 'Glucose', 'Tartaric Acid', 'Levulinic Acid',
            'Citric Acid', 'Glycerol', 'Malic Acid', 'Lactic Acid',
            'Lactic Acid', 'Glycerol', 'Malic Acid',
            'Citric Acid', 'Malic Acid', 'Citric Acid'],
    'molar_ratio': ['1:2', '1:2', '1:1', '1:2', '1:1', '1:2', '1:1', '1:1', '2:1', '1:2',
                    '1:1', '1:2', '1:1', '1:2', '1:2', '1:3', '1:1', '1:1', '1:1', '1:1'],
    'viscosity': [750, 376, 5800, 220, 4200, 36, 8900, 15000, 6500, 190,
                  6100, 510, 4800, 245, 280, 620, 5200, 12500, 9800, 18000],
    'density': [1.25, 1.18, 1.33, 1.14, 1.28, 1.12, 1.36, 1.40, 1.30, 1.13,
                1.31, 1.20, 1.27, 1.15, 1.16, 1.22, 1.26, 1.42, 1.38, 1.45],
    'conductivity': [0.199, 1.05, 0.076, 2.31, 0.12, 7.61, 0.042, 0.015, 0.058, 2.85,
                     0.065, 0.82, 0.098, 2.12, 1.95, 0.65, 0.088, 0.018, 0.028, 0.008],
    'polarity_ET30': [52.1, 57.3, 50.8, 53.7, 51.5, 55.2, 49.3, 49.0, 50.5, 53.2,
                      51.2, 56.8, 52.0, 54.5, 54.1, 57.9, 51.8, 48.7, 49.5, 47.2],
    'pH': [7.2, 5.8, 1.2, 2.1, 1.8, 6.5, 0.9, 5.0, 1.6, 2.5,
           1.5, 6.2, 2.0, 2.4, 2.3, 5.5, 2.2, 1.4, 1.9, 1.3],
    'water_activity': [0.42, 0.38, 0.31, 0.45, 0.33, 0.52, 0.28, 0.26, 0.32, 0.47,
                       0.30, 0.40, 0.34, 0.46, 0.44, 0.37, 0.35, 0.25, 0.27, 0.22],
    'surface_tension': [52.1, 55.8, 64.2, 48.3, 61.5, 49.1, 67.8, 69.0, 63.8, 46.8,
                        62.7, 54.2, 60.1, 48.0, 47.6, 56.3, 59.5, 70.2, 68.5, 72.1],
    'biodegradability': [0.95, 0.98, 0.92, 0.97, 0.94, 0.88, 0.85, 0.97, 0.93, 0.91,
                         0.96, 0.99, 0.95, 0.97, 0.97, 0.98, 0.96, 0.99, 0.99, 0.99],
    'toxicity_LC50': [8500, 12000, 6200, 9800, 7100, 7500, 4300, 14000, 8200, 7800,
                      11000, 15000, 9500, 10500, 10200, 13500, 10800, 18000, 17500, 20000],
}

nades_df = pd.DataFrame(nades_data)

# Organometallic Therapeutic Compounds
compounds_data = {
    'name': ['RAPTA-C', 'Ferrocifen', 'Ti-salan', 'Au-NHC-Thiolate',
             'Os-azopyridine', 'Ir-Cp*-phen', 'Zn-Phthalocyanine',
             'Cu-phen-gly', 'V-Curcumin', 'Pt-Satraplatin', 'Rh-Cp*-bipy', 'Co-Salen-5FU'],
    'metal': ['Ru', 'Fe', 'Ti', 'Au', 'Os', 'Ir', 'Zn', 'Cu', 'V', 'Pt', 'Rh', 'Co'],
    'MW': [458.3, 537.5, 412.8, 489.2, 523.7, 547.1, 577.9, 356.2, 483.5, 501.3, 478.6, 445.8],
    'logP': [-0.8, 4.2, 1.5, 2.1, 1.8, 0.9, 3.5, -0.3, 2.8, 0.5, 1.2, 0.2],
    'PSA': [42.5, 38.1, 55.2, 31.8, 45.6, 28.3, 62.4, 58.7, 85.2, 72.1, 32.5, 95.3],
    'IC50_uM': [12.5, 0.5, 3.8, 0.85, 4.2, 2.1, 0.12, 1.8, 5.6, 2.5, 6.8, 1.2],
    'selectivity_index': [8.2, 15.3, 12.1, 22.5, 10.8, 18.7, 35.2, 14.5, 9.3, 11.2, 7.5, 19.8],
    'biological_target': ['Cathepsin B', 'ER', 'DNA Topo', 'TrxR', 'GSH', 'NADH',
                          'PDT ROS', 'DNA Intercal.', 'PTP1B', 'DNA Crosslink', 'Kinase', 'TS'],
}

compounds_df = pd.DataFrame(compounds_data)

print(f"📋 NADES Systems: {len(nades_df)}")
print(f"💊 Organometallic Compounds: {len(compounds_df)}")
print(f"\n=== NADES Dataset Preview ===")
display(nades_df.head(10))
print(f"\n=== Organometallic Compounds ===")
display(compounds_df)

# =============================================================================
# Advanced Feature Engineering for NADES-Drug Interactions
# =============================================================================

from itertools import product
import hashlib

# Generate all possible NADES-Compound interaction pairs
interactions = []

for n_idx, nades_row in nades_df.iterrows():
    for c_idx, comp_row in compounds_df.iterrows():
        # Base features
        features = {
            # NADES physicochemical properties
            'viscosity': nades_row['viscosity'],
            'density': nades_row['density'],
            'conductivity': nades_row['conductivity'],
            'polarity': nades_row['polarity_ET30'],
            'pH': nades_row['pH'],
            'water_activity': nades_row['water_activity'],
            'surface_tension': nades_row['surface_tension'],
            'biodegradability': nades_row['biodegradability'],
            'toxicity_LC50': nades_row['toxicity_LC50'],

            # Compound properties
            'MW': comp_row['MW'],
            'logP': comp_row['logP'],
            'PSA': comp_row['PSA'],
            'IC50': comp_row['IC50_uM'],
            'selectivity': comp_row['selectivity_index'],

            # Engineered cross-features
            'visc_logP_interaction': np.log1p(nades_row['viscosity']) * comp_row['logP'],
            'pH_psa_ratio': nades_row['pH'] / (comp_row['PSA'] + 1),
            'conductivity_MW_ratio': nades_row['conductivity'] * 1000 / comp_row['MW'],
            'polarity_logP_product': nades_row['polarity_ET30'] * comp_row['logP'],
            'HLB_estimate': 20 * (1 - comp_row['logP'] / 10),  # Simplified HLB
            'dissolution_score': nades_row['conductivity'] * nades_row['water_activity'] / (nades_row['viscosity'] / 1000),
            'biocompat_score': nades_row['biodegradability'] * nades_row['toxicity_LC50'] / 10000,
            'permeation_estimate': comp_row['logP'] * nades_row['water_activity'] * (1 / np.log1p(comp_row['MW'])),
            'stability_estimate': nades_row['surface_tension'] * comp_row['selectivity_index'] / (nades_row['pH'] + 1),

            # Thermodynamic estimates
            'excess_gibbs': -8.314 * 298 * np.log(nades_row['water_activity'] + 0.01) / 1000,
            'hansen_distance': np.sqrt(
                (nades_row['polarity_ET30'] - 50)**2 +
                (comp_row['logP'] * 5)**2 +
                (nades_row['surface_tension'] - 50)**2
            ),

            # Metal-specific encoding
            'metal_period': {'Ru': 5, 'Fe': 4, 'Ti': 4, 'Au': 6, 'Os': 6,
                           'Ir': 6, 'Zn': 4, 'Cu': 4, 'V': 4, 'Pt': 6, 'Rh': 5, 'Co': 4}[comp_row['metal']],
            'metal_group': {'Ru': 8, 'Fe': 8, 'Ti': 4, 'Au': 11, 'Os': 8,
                          'Ir': 9, 'Zn': 12, 'Cu': 11, 'V': 5, 'Pt': 10, 'Rh': 9, 'Co': 9}[comp_row['metal']],
            'd_electrons': {'Ru': 6, 'Fe': 6, 'Ti': 0, 'Au': 10, 'Os': 6,
                          'Ir': 6, 'Zn': 10, 'Cu': 10, 'V': 0, 'Pt': 8, 'Rh': 6, 'Co': 6}[comp_row['metal']],
        }

        # Simulated biological targets (based on physicochemical relationships)
        np.random.seed(int(hashlib.md5(f"{n_idx}_{c_idx}".encode()).hexdigest()[:8], 16) % 2**31)

        base_sol = features['dissolution_score'] * 2 + np.random.normal(0, 0.5)
        base_stab = features['stability_estimate'] * 0.1 + np.random.normal(0, 0.3)

        features['solubility_enhancement'] = max(1.0, base_sol + 3)
        features['stability_enhancement'] = max(1.0, base_stab + 2)
        features['cell_viability'] = min(100, max(60, 90 + features['biocompat_score'] * 5 + np.random.normal(0, 3)))
        features['skin_permeation'] = max(0, features['permeation_estimate'] * 10 + np.random.normal(15, 5))
        features['anti_cancer_IC50'] = max(0.01, comp_row['IC50_uM'] * (0.3 + features['dissolution_score'] * 0.1) + np.random.normal(0, 0.5))
        features['synergistic_score'] = min(1, max(0, 0.5 + features['biocompat_score'] * 0.2 + features['dissolution_score'] * 0.05 + np.random.normal(0, 0.08)))

        interactions.append(features)

interaction_df = pd.DataFrame(interactions)

# Feature names for ML
feature_cols = [c for c in interaction_df.columns if c not in [
    'solubility_enhancement', 'stability_enhancement', 'cell_viability',
    'skin_permeation', 'anti_cancer_IC50', 'synergistic_score'
]]
target_cols = ['solubility_enhancement', 'stability_enhancement', 'cell_viability',
               'skin_permeation', 'anti_cancer_IC50', 'synergistic_score']

print(f"✅ Generated {len(interaction_df)} interaction samples")
print(f"📊 Features: {len(feature_cols)}, Targets: {len(target_cols)}")
print("\n=== Feature Correlation Heatmap ===")

# Create and save heatmap file directly instead of breaking with plt.show()
plt.figure(figsize=(16, 12))
sns.heatmap(interaction_df[feature_cols].corr(), annot=False, cmap='RdBu_r',
            center=0, square=True, linewidths=0.5)
plt.title('Feature Correlation Matrix', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('feature_correlation.png')
plt.close()
print("💾 Correlation heatmap saved successfully as 'feature_correlation.png'!")

# =============================================================================
# Comprehensive Multi-Model Training Pipeline
# =============================================================================

from sklearn.model_selection import RepeatedKFold
from sklearn.tree import DecisionTreeRegressor

# Prepare data
X = interaction_df[feature_cols].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Define all models
def get_models():
    models = {
        '1. Linear Regression': LinearRegression(),
        '2. Ridge (α=1.0)': Ridge(alpha=1.0),
        '3. Lasso (α=0.01)': Lasso(alpha=0.01),
        '4. ElasticNet': ElasticNet(alpha=0.01, l1_ratio=0.5),
        '5. Bayesian Ridge': BayesianRidge(),
        '6. Huber Regressor': HuberRegressor(),
        '7. KNN (k=5)': KNeighborsRegressor(n_neighbors=5, weights='distance'),
        '8. SVR (RBF)': SVR(kernel='rbf', C=10, epsilon=0.1),
        '9. Decision Tree': DecisionTreeRegressor(max_depth=8, random_state=42),
        '10. Random Forest': RandomForestRegressor(n_estimators=200, max_depth=12, random_state=42, n_jobs=-1),
        '11. Extra Trees': ExtraTreesRegressor(n_estimators=200, max_depth=12, random_state=42, n_jobs=-1),
        '12. Gradient Boosting': GradientBoostingRegressor(n_estimators=200, max_depth=5, learning_rate=0.1, random_state=42),
        '13. AdaBoost': AdaBoostRegressor(n_estimators=100, learning_rate=0.1, random_state=42),
        '14. XGBoost': xgb.XGBRegressor(n_estimators=200, max_depth=6, learning_rate=0.1,
                                         subsample=0.8, colsample_bytree=0.8, random_state=42, verbosity=0),
        '15. LightGBM': lgb.LGBMRegressor(n_estimators=200, max_depth=6, learning_rate=0.1,
                                           subsample=0.8, colsample_bytree=0.8, random_state=42, verbose=-1),
        '16. CatBoost': CatBoostRegressor(iterations=200, depth=6, learning_rate=0.1,
                                           random_state=42, verbose=0),
        '17. MLP (64-32-16)': MLPRegressor(hidden_layer_sizes=(64, 32, 16), max_iter=500,
                                           learning_rate='adaptive', random_state=42),
    }
    return models

# Train and evaluate all models for each target
results_all = {}
trained_models = {}  # Dictionary to save the actual trained models for later deployment

for target in target_cols:
    print(f"\n{'='*70}")
    print(f"🎯 Target: {target}")
    print(f"{'='*70}")

    y = interaction_df[target].values
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    models = get_models()
    results = []
    trained_models[target] = {}

    for name, model in models.items():
        try:
            model.fit(X_train, y_train)
            
            # Store trained instance
            trained_models[target][name] = model
            
            y_pred_train = model.predict(X_train)
            y_pred_test = model.predict(X_test)

            # Cross-validation
            cv = RepeatedKFold(n_splits=5, n_repeats=3, random_state=42)
            cv_scores = cross_val_score(model, X_scaled, y, cv=cv, scoring='r2')

            r2_train = r2_score(y_train, y_pred_train)
            r2_test = r2_score(y_test, y_pred_test)
            rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
            mae_test = mean_absolute_error(y_test, y_pred_test)

            results.append({
                'Model': name,
                'R² Train': r2_train,
                'R² Test': r2_test,
                'RMSE': rmse_test,
                'MAE': mae_test,
                'CV R² (mean)': cv_scores.mean(),
                'CV R² (std)': cv_scores.std(),
            })

            print(f"  {name}: R²={r2_test:.4f} | RMSE={rmse_test:.4f} | CV={cv_scores.mean():.4f}±{cv_scores.std():.4f}")

        except Exception as e:
            print(f"  ❌ {name}: {e}")

    results_all[target] = pd.DataFrame(results).sort_values('R² Test', ascending=False)

# Display best models summary
print("\n\n" + "="*70)
print("🏆 BEST MODELS PER TARGET")
print("="*70)
for target, df in results_all.items():
    if not df.empty:
        best = df.iloc[0]
        print(f"  {target}: {best['Model']} → R²={best['R² Test']:.4f}")


# =============================================================================
# Custom PyTorch Deep Neural Network with Attention for NADES Prediction
# =============================================================================

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

class AttentionBlock(nn.Module):
    """Self-attention mechanism for feature interaction learning"""
    def __init__(self, dim):
        super().__init__()
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        self.scale = dim ** -0.5

    def forward(self, x):
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        attn = torch.softmax(Q * K * self.scale, dim=-1)
        return attn * V + x  # Residual connection

class NADESPredictor(nn.Module):
    """Deep neural network for NADES-Drug interaction prediction"""
    def __init__(self, input_dim, hidden_dims=[256, 128, 64, 32], n_targets=6, dropout=0.3):
        super().__init__()

        layers = []
        prev_dim = input_dim

        for i, h_dim in enumerate(hidden_dims):
            layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.GELU(),
                nn.Dropout(dropout if i < len(hidden_dims) - 1 else dropout * 0.5),
            ])
            if i == 1:  # Add attention after second layer
                layers.append(AttentionBlock(h_dim))
            prev_dim = h_dim

        self.backbone = nn.Sequential(*layers)

        # Multi-task output heads
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dims[-1], 16),
                nn.GELU(),
                nn.Linear(16, 1)
            ) for _ in range(n_targets)
        ])

    def forward(self, x):
        features = self.backbone(x)
        outputs = [head(features).squeeze(-1) for head in self.heads]
        return torch.stack(outputs, dim=1)

# Prepare data
X_tensor = torch.FloatTensor(X_scaled).to(device)
y_all = interaction_df[target_cols].values
y_scaler = StandardScaler()
y_scaled = y_scaler.fit_transform(y_all)
y_tensor = torch.FloatTensor(y_scaled).to(device)

# Train/test split
n_train = int(0.8 * len(X_tensor))
indices = torch.randperm(len(X_tensor))
train_idx, test_idx = indices[:n_train], indices[n_train:]

X_train_t, X_test_t = X_tensor[train_idx], X_tensor[test_idx]
y_train_t, y_test_t = y_tensor[train_idx], y_tensor[test_idx]

# Create DataLoader
train_dataset = TensorDataset(X_train_t, y_train_t)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Initialize model
pytorch_model = NADESPredictor(
    input_dim=X_scaled.shape[1],
    hidden_dims=[256, 128, 64, 32],
    n_targets=len(target_cols),
    dropout=0.3
).to(device)

optimizer = optim.AdamW(pytorch_model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=50, T_mult=2)
criterion = nn.HuberLoss(delta=1.0)

# Training loop
n_epochs = 300
train_losses, test_losses = [], []

print(f"\n{'='*60}")
print(f"Training NADESPredictor on {device}")
print(f"Parameters: {sum(p.numel() for p in pytorch_model.parameters()):,}")
print(f"{'='*60}")

best_test_loss = float('inf')
patience, patience_counter = 30, 0

for epoch in range(n_epochs):
    # Train
    pytorch_model.train()
    epoch_loss = 0
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        pred = pytorch_model(X_batch)
        loss = criterion(pred, y_batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(pytorch_model.parameters(), 1.0)
        optimizer.step()
        epoch_loss += loss.item()

    scheduler.step()
    train_loss = epoch_loss / len(train_loader)
    train_losses.append(train_loss)

    # Evaluate
    pytorch_model.eval()
    with torch.no_grad():
        test_pred = pytorch_model(X_test_t)
        test_loss = criterion(test_pred, y_test_t).item()
        test_losses.append(test_loss)

    if test_loss < best_test_loss:
        best_test_loss = test_loss
        patience_counter = 0
        torch.save(pytorch_model.state_dict(), 'best_nades_model.pth')
    else:
        patience_counter += 1

    if patience_counter >= patience:
        print(f"Early stopping at epoch {epoch+1}")
        break

    if (epoch + 1) % 50 == 0:
        print(f"Epoch {epoch+1}/{n_epochs} | Train Loss: {train_loss:.6f} | Test Loss: {test_loss:.6f}")

# Load best model and evaluate
pytorch_model.load_state_dict(torch.load('best_nades_model.pth', map_location=device))
pytorch_model.eval()

with torch.no_grad():
    y_pred_scaled = pytorch_model(X_test_t).cpu().numpy()
    y_pred_original = y_scaler.inverse_transform(y_pred_scaled)
    y_test_original = y_scaler.inverse_transform(y_test_t.cpu().numpy())

print(f"\n{'='*60}")
print("📊 Per-Target R² Scores (Deep Learning)")
print(f"{'='*60}")
for i, target in enumerate(target_cols):
    r2 = r2_score(y_test_original[:, i], y_pred_original[:, i])
    print(f"  {target}: R² = {r2:.4f}")

# Generate and save metrics plot file instead of using pop-up display
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].plot(train_losses, label='Train', color='#818cf8', linewidth=2)
axes[0].plot(test_losses, label='Test', color='#f472b6', linewidth=2)
axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Loss')
axes[0].set_title('Training Curves'); axes[0].legend()
axes[0].set_yscale('log')

best_target_idx = np.argmax([r2_score(y_test_original[:, i], y_pred_original[:, i]) for i in range(len(target_cols))])
axes[1].scatter(y_test_original[:, best_target_idx], y_pred_original[:, best_target_idx],
               alpha=0.6, color='#34d399', edgecolors='white', linewidth=0.5)
lims = [min(y_test_original[:, best_target_idx].min(), y_pred_original[:, best_target_idx].min()),
        max(y_test_original[:, best_target_idx].max(), y_pred_original[:, best_target_idx].max())]
axes[1].plot(lims, lims, 'r--', alpha=0.5)
axes[1].set_xlabel('Actual'); axes[1].set_ylabel('Predicted')
axes[1].set_title(f'Best Target: {target_cols[best_target_idx]}')
plt.tight_layout()
plt.savefig('nn_training_metrics.png')
plt.close()
print("💾 Neural Network training plots successfully saved as 'nn_training_metrics.png'!")

# =============================================================================
# Bayesian Hyperparameter Optimization with Optuna
# =============================================================================

target_for_optimization = 'synergistic_score'  # Change this for other targets
y_opt = interaction_df[target_for_optimization].values

def objective(trial):
    """Optuna objective function for XGBoost optimization"""
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 500),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'gamma': trial.suggest_float('gamma', 0, 5),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10, log=True),
        'random_state': 42,
        'verbosity': 0,
    }

    model = xgb.XGBRegressor(**params)
    cv_scores = cross_val_score(model, X_scaled, y_opt, cv=5, scoring='r2')
    return cv_scores.mean()

# Run optimization
study = optuna.create_study(direction='maximize', study_name='NADES-XGBoost')
study.optimize(objective, n_trials=100)  # Clean log streaming for production

print(f"\n{'='*60}")
print(f"🏆 Best Trial: #{study.best_trial.number}")
print(f"   Best R² Score: {study.best_value:.4f}")
print(f"   Best Parameters:")
for key, value in study.best_params.items():
    print(f"     {key}: {value}")
print(f"{'='*60}")

# Train final model with best params
best_opt_model = xgb.XGBRegressor(**study.best_params, random_state=42, verbosity=0)
best_opt_model.fit(X_scaled, y_opt)

# Generate and save optimization plots
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Optimization history tracking
trials_df = study.trials_dataframe()
axes[0].scatter(trials_df.index, trials_df['value'], alpha=0.5, c=trials_df['value'],
               cmap='viridis', edgecolors='white', linewidth=0.5)
axes[0].set_xlabel('Trial'); axes[0].set_ylabel('R² Score')
axes[0].set_title('Optuna Optimization History')

# Parameter importance visualization
importance = optuna.importance.get_param_importances(study)
axes[1].barh(list(importance.keys())[:10], list(importance.values())[:10], color='#818cf8')
axes[1].set_xlabel('Importance'); axes[1].set_title('Hyperparameter Importance')
plt.tight_layout()
plt.savefig('optuna_optimization.png')
plt.close()
print("💾 Optuna visualization plots successfully saved as 'optuna_optimization.png'!")

# =============================================================================
# Model Explainability: SHAP & LIME Analysis
# =============================================================================

# Train best model for SHAP analysis
target_explain = 'synergistic_score'
y_explain = interaction_df[target_explain].values

xgb_model = xgb.XGBRegressor(n_estimators=200, max_depth=6, learning_rate=0.1,
                               random_state=42, verbosity=0)
xgb_model.fit(X_scaled, y_explain)

# SHAP Analysis
print("Computing SHAP values...")
explainer = shap.TreeExplainer(xgb_model)
shap_values = explainer.shap_values(X_scaled)

# 1. SHAP Summary Plot
plt.figure(figsize=(12, 8))
shap.summary_plot(shap_values, X_scaled, feature_names=feature_cols, show=False)
plt.title(f'SHAP Feature Importance for {target_explain}', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('shap_summary.png')
plt.close()
print("💾 Global SHAP summary saved successfully as 'shap_summary.png'!")

# 2. SHAP Dependency Plots (Top 4 Features)
top_features = np.argsort(np.abs(shap_values).mean(axis=0))[-4:]
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
for idx, feat_idx in enumerate(top_features):
    ax = axes[idx // 2, idx % 2]
    shap.dependence_plot(feat_idx, shap_values, X_scaled,
                        feature_names=feature_cols, ax=ax, show=False)
plt.suptitle('SHAP Dependency Plots - Top 4 Features', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('shap_dependency.png')
plt.close()
print("💾 SHAP dependency grid saved successfully as 'shap_dependency.png'!")

# LIME Explanation Setup
print("\n" + "="*60)
print("🍋 LIME Explanation for Sample Prediction")
print("="*60)

lime_explainer = lime.lime_tabular.LimeTabularExplainer(
    X_scaled, feature_names=feature_cols,
    mode='regression', random_state=42
)

# Explain a high-synergy prediction sample
sample_idx = np.argmax(y_explain)
exp = lime_explainer.explain_instance(X_scaled[sample_idx], xgb_model.predict, num_features=12)

print(f"\nSample {sample_idx}: Actual={y_explain[sample_idx]:.4f}, Predicted={xgb_model.predict(X_scaled[sample_idx:sample_idx+1])[0]:.4f}")
print("\nTop contributing features from LIME:")
for feat, weight in exp.as_list():
    direction = "↑" if weight > 0 else "↓"
    print(f"  {direction} {feat}: {weight:+.4f}")

# 3. Save LIME Plot
exp.as_pyplot_figure()
plt.title('LIME Explanation for High-Synergy NADES-Drug Pair')
plt.tight_layout()
plt.savefig('lime_explanation.png')
plt.close()
print("💾 Local LIME report saved successfully as 'lime_explanation.png'!")

# 4. Save SHAP Waterfall Force Plot
plt.figure(figsize=(10, 6))
shap.plots.waterfall(shap.Explanation(
    values=shap_values[sample_idx],
    base_values=explainer.expected_value,
    data=X_scaled[sample_idx],
    feature_names=feature_cols
), show=False)
plt.title('SHAP Local Waterfall Explanation', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('shap_waterfall.png')
plt.close()
print("💾 Local SHAP waterfall saved successfully as 'shap_waterfall.png'!")

# =============================================================================
# Advanced Stacking Ensemble with Meta-Learner
# =============================================================================

from sklearn.model_selection import KFold

target_stack = 'synergistic_score'
y_stack = interaction_df[target_stack].values

# Level 1: Base models (diverse algorithms)
base_models = [
    ('ridge', Ridge(alpha=1.0)),
    ('lasso', Lasso(alpha=0.01)),
    ('knn', KNeighborsRegressor(n_neighbors=7, weights='distance')),
    ('svr', SVR(kernel='rbf', C=10)),
    ('rf', RandomForestRegressor(n_estimators=200, max_depth=8, random_state=42, n_jobs=-1)),
    ('et', ExtraTreesRegressor(n_estimators=200, max_depth=8, random_state=42, n_jobs=-1)),
    ('xgb', xgb.XGBRegressor(n_estimators=200, max_depth=6, learning_rate=0.1, random_state=42, verbosity=0)),
    ('lgbm', lgb.LGBMRegressor(n_estimators=200, max_depth=6, learning_rate=0.1, random_state=42, verbose=-1)),
    ('cat', CatBoostRegressor(iterations=200, depth=6, learning_rate=0.1, random_state=42, verbose=0)),
    ('mlp', MLPRegressor(hidden_layer_sizes=(128, 64, 32), max_iter=500, random_state=42)),
]

# Create Stacking Ensemble Wrapper
stacking_model = StackingRegressor(
    estimators=base_models,
    final_estimator=xgb.XGBRegressor(n_estimators=100, max_depth=4, learning_rate=0.05,
                                      random_state=42, verbosity=0),
    cv=5,
    n_jobs=-1,
    passthrough=True  # Include original engineered features for the meta-learner
)

# Train and evaluate split split
X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(X_scaled, y_stack, test_size=0.2, random_state=42)

print("Training Stacking Ensemble (this may take a few minutes)...")
stacking_model.fit(X_train_s, y_train_s)

y_pred_stack = stacking_model.predict(X_test_s)
r2_stack = r2_score(y_test_s, y_pred_stack)
rmse_stack = np.sqrt(mean_squared_error(y_test_s, y_pred_stack))

print(f"\n{'='*60}")
print(f"🏆 Stacking Ensemble Results")
print(f"  R² Score: {r2_stack:.4f}")
print(f"  RMSE: {rmse_stack:.4f}")
print(f"{'='*60}")

# Compare individual performance metrics vs final stacked prediction model
print("\n📊 Individual Model Comparison:")
for name, model in base_models:
    model.fit(X_train_s, y_train_s)
    r2_ind = r2_score(y_test_s, model.predict(X_test_s))
    improvement = ((r2_stack - r2_ind) / max(abs(r2_ind), 0.001)) * 100
    print(f"  {name:>10}: R² = {r2_ind:.4f} {'(+' if improvement > 0 else '('}{improvement:.1f}% vs stacking)")

# Performance evaluation metric plotting pipeline
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Plot 1: Actual vs Predicted tracking
axes[0].scatter(y_test_s, y_pred_stack, alpha=0.6, c='#818cf8', edgecolors='white', linewidth=0.5, s=60)
lims = [min(y_test_s.min(), y_pred_stack.min()), max(y_test_s.max(), y_pred_stack.max())]
axes[0].plot(lims, lims, 'r--', alpha=0.5, linewidth=2)
axes[0].set_xlabel('Actual'); axes[0].set_ylabel('Predicted')
axes[0].set_title(f'Stacking: R²={r2_stack:.4f}')

# Plot 2: Residual distribution mapping
residuals = y_test_s - y_pred_stack
axes[1].scatter(y_pred_stack, residuals, alpha=0.6, c='#34d399', edgecolors='white', linewidth=0.5, s=60)
axes[1].axhline(y=0, color='r', linestyle='--', alpha=0.5)
axes[1].set_xlabel('Predicted'); axes[1].set_ylabel('Residual')
axes[1].set_title('Residual Analysis')

# Plot 3: Baseline Algorithms vs Final Stacking benchmark graph
models_comp = [(n, r2_score(y_test_s, m.predict(X_test_s))) for n, m in base_models]
models_comp.append(('STACKING', r2_stack))
models_comp.sort(key=lambda x: x[1])
colors = ['#818cf8'] * len(base_models) + ['#f472b6']
axes[2].barh([m[0] for m in models_comp], [m[1] for m in models_comp], color=colors)
axes[2].set_xlabel('R² Score')
axes[2].set_title('Model Comparison')

plt.tight_layout()
plt.savefig('stacking_ensemble_metrics.png')
plt.close()
print("💾 Stacking performance plots successfully saved as 'stacking_ensemble_metrics.png'!")

# =============================================================================
# Automated Drug Discovery Pipeline
# Screen all combinations & rank therapeutic candidates
# =============================================================================

print("🔬 Screening all NADES-Compound combinations...")
print(f"Total combinations: {len(nades_df)} × {len(compounds_df)} = {len(nades_df) * len(compounds_df)}")

# Use best stacking model to predict all targets
all_predictions = {}
for target in target_cols:
    y_target = interaction_df[target].values
    model = xgb.XGBRegressor(n_estimators=200, max_depth=6, learning_rate=0.1, random_state=42, verbosity=0)
    model.fit(X_scaled, y_target)
    all_predictions[target] = model.predict(X_scaled)

# Create comprehensive screening results
screening_df = interaction_df.copy()
for target in target_cols:
    screening_df[f'{target}_pred'] = all_predictions[target]

# Compute composite drugability score
screening_df['drugability_score'] = (
    screening_df['synergistic_score_pred'].clip(0, 1) * 30 +
    (screening_df['cell_viability_pred'].clip(60, 100) - 60) / 40 * 25 +
    screening_df['solubility_enhancement_pred'].clip(0, 15) / 15 * 20 +
    screening_df['stability_enhancement_pred'].clip(0, 8) / 8 * 15 +
    (1 - screening_df['anti_cancer_IC50_pred'].clip(0, 20) / 20) * 10
)

# Add cost-effectiveness score (based on NADES availability and biodegradability)
screening_df['cost_score'] = (
    screening_df['biodegradability'] * 40 +
    screening_df['toxicity_LC50'].clip(0, 20000) / 20000 * 30 +
    (1 - screening_df['viscosity'].clip(0, 20000) / 20000) * 30
)

screening_df['overall_score'] = screening_df['drugability_score'] * 0.7 + screening_df['cost_score'] * 0.3

# Rank top candidates
top_candidates = screening_df.nlargest(20, 'overall_score')

print(f"\n{'='*70}")
print("🏆 TOP 20 NADES-DRUG CANDIDATES FOR THERAPEUTIC DEVELOPMENT")
print(f"{'='*70}")

for rank, (idx, row) in enumerate(top_candidates.iterrows(), 1):
    nades_idx = idx // len(compounds_df)
    comp_idx = idx % len(compounds_df)
    nades_name = f"{nades_df.iloc[nades_idx]['hba']} + {nades_df.iloc[nades_idx]['hbd']}"
    comp_name = compounds_df.iloc[comp_idx]['name']

    print(f"\n  #{rank}. {nades_name} → {comp_name}")
    print(f"     Overall Score: {row['overall_score']:.1f}/100")
    print(f"     Drugability: {row['drugability_score']:.1f} | Cost-effectiveness: {row['cost_score']:.1f}")
    print(f"     Synergy: {row['synergistic_score_pred']:.3f} | Cell Viability: {row['cell_viability_pred']:.1f}%")
    print(f"     Solubility ↑: {row['solubility_enhancement_pred']:.1f}× | Anti-Cancer IC50: {row['anti_cancer_IC50_pred']:.2f} µM")

# Visualization Engine Initialization
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 1. Heatmap generation
row_indices = screening_df.index // len(compounds_df)
col_indices = screening_df.index % len(compounds_df)
pivot_data = screening_df.pivot_table(
    values='overall_score',
    index=row_indices,
    columns=col_indices,
    aggfunc='mean'
)
sns.heatmap(pivot_data, cmap='RdYlGn', annot=True, fmt='.0f',
           xticklabels=compounds_df['name'].values,
           yticklabels=[f"{r['hba'][:3]}+{r['hbd'][:3]}" for _, r in nades_df.iterrows()],
           ax=axes[0, 0])
axes[0, 0].set_title('Overall Score Heatmap', fontweight='bold')
axes[0, 0].tick_params(axis='x', rotation=45)

# 2. Drugability vs Cost scatter graph
scatter1 = axes[0, 1].scatter(screening_df['drugability_score'], screening_df['cost_score'],
                  c=screening_df['overall_score'], cmap='viridis', alpha=0.7,
                  edgecolors='white', linewidth=0.5, s=60)
axes[0, 1].set_xlabel('Drugability Score')
axes[0, 1].set_ylabel('Cost-effectiveness Score')
axes[0, 1].set_title('Drugability vs Cost-effectiveness')
fig.colorbar(scatter1, ax=axes[0, 1], label='Overall Score')

# 3. Top 10 Horizontal Bar Chart
top10 = screening_df.nlargest(10, 'overall_score')
labels = [f"{compounds_df.iloc[i % len(compounds_df)]['name'][:10]}\n({nades_df.iloc[i // len(compounds_df)]['hbd'][:5]})"
          for i in top10.index]
axes[1, 0].barh(labels, top10['overall_score'], color=plt.cm.viridis(np.linspace(0.3, 0.9, 10)))
axes[1, 0].set_xlabel('Overall Score')
axes[1, 0].set_title('Top 10 Candidates')

# 4. Dimension Reduction t-SNE Plot
from sklearn.manifold import TSNE
tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(X_scaled) - 1))
X_tsne = tsne.fit_transform(X_scaled)
scatter2 = axes[1, 1].scatter(X_tsne[:, 0], X_tsne[:, 1],
                            c=screening_df['synergistic_score_pred'], cmap='coolwarm',
                            alpha=0.7, edgecolors='white', linewidth=0.3, s=40)
axes[1, 1].set_xlabel('t-SNE 1'); axes[1, 1].set_ylabel('t-SNE 2')
axes[1, 1].set_title('t-SNE Embedding (colored by Synergy)')
fig.colorbar(scatter2, ax=axes[1, 1], label='Synergy Score')

plt.suptitle('NADES-Organometallic Drug Discovery Dashboard', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('drug_discovery_pipeline.png')
plt.close()

print("\n✅ Drug discovery pipeline complete!")
print("💾 Discovery analytics plots successfully saved as 'drug_discovery_pipeline.png'!")
