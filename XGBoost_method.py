import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ===================== 1. Data loading =====================
ROOT = Path(__file__).parent
base_path = ROOT / "python2" / "SAR_data.csv"
supply_path = ROOT / "supply_data" / "SAR_data.csv"

print("Loading the dataset...")
df_base = pd.read_csv(base_path)
df_supply = pd.read_csv(supply_path)

df = pd.concat([df_base, df_supply], axis=0, ignore_index=True)
print(f"Data merging completed! Base sample count: {len(df_base)}, Supplemental sample count: {len(df_supply)}, Total sample count: {len(df)}")

# ===================== 2. Data cleaning and outlier filtering =====================
features = ['Peak_dist', 'Peak_Intensity', 'lambda', 'theta_deg']
target = 'href'

df_clean = df[features + [target]].copy()
df_clean = df_clean.dropna()
df_clean = df_clean[np.isfinite(df_clean).all(axis=1)]

# ===================== 3. Depth-based multi-dimensional area classification labels =====================
bins = [0.0, 20.7, 41.4, 62.0, 82.7, 103.4]
labels = ['Region 1 (0.0m ~ 20.7m)', 
          'Region 2 (20.7m ~ 41.4m)', 
          'Region 3 (41.4m ~ 62.0m)', 
          'Region 4 (62.0m ~ 82.7m)', 
          'Region 5 (82.7m ~ 103.4m)']

df_clean['depth_region'] = pd.cut(df_clean[target], bins=bins, labels=labels, include_lowest=True)

# ===================== 4. Dividing the dataset and training the model =====================
X = df_clean[features]
y = df_clean[target]

X_train, X_test, y_train, y_test = train_test_split(
    df_clean[features + ['depth_region']], y, test_size=0.2, random_state=42
)

X_train_pure = X_train[features]
X_test_pure = X_test[features]
regions_test = X_test['depth_region'] 

print("Starting to train the global XGBoost bathymetry inversion model...")
model = xgb.XGBRegressor(
    n_estimators=1000,
    learning_rate=0.05,
    max_depth=7,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1 
)
model.fit(X_train_pure, y_train)
print("Model training completed!")

# ===================== 5. Global assessment and comprehensive output of regional indicators =====================
y_pred = model.predict(X_test_pure)

# calculate RE (%)
global_re = np.mean(np.abs((y_test - y_pred) / (y_test + 1e-5))) * 100

print("\n" + "="*20 + " Global Overall Evaluation Results " + "="*20)
print(f"Global Overall MAE      : {mean_absolute_error(y_test, y_pred):.2f} m")
print(f"Global Overall RMSE     : {mean_squared_error(y_test, y_pred)**0.5:.2f} m")
print(f"Global Overall RE       : {global_re:.2f}%")
print(f"Global Overall R2 Score : {r2_score(y_test, y_pred):.4f}")

print("\n" + "="*25 + " Starting Comprehensive Error Assessment of Test Set Multi-dimensional Bathymetry Segments " + "="*25)

print(f"{'Depth Region':<24} | {'Validation Samples':<8} | {'MAE':<10} | {'RMSE':<10} | {'RE':<10} | {'R2 Score'}")
print("-" * 90)

# Loop through each depth region to calculate detailed metrics
for label in labels:
    mask = (regions_test == label)
    sample_count = mask.sum()
    
    if sample_count == 0:
        print(f"{label:<22} | {0:<10} | {'N/A':<12} | {'N/A':<12} | {'N/A':<11} | N/A")
        continue
        
    y_test_sub = y_test[mask]
    y_pred_sub = y_pred[mask]
    
    # calculate region-specific metrics
    sub_mae = mean_absolute_error(y_test_sub, y_pred_sub)
    sub_rmse = mean_squared_error(y_test_sub, y_pred_sub) ** 0.5
    
    # calculate relative error RE (%)
    relative_errors = np.abs((y_test_sub - y_pred_sub) / (y_test_sub + 1e-5))
    sub_re = np.mean(relative_errors) * 100
    
    # calculate region R2 Score (with protection: cannot calculate R2 with less than 2 samples)
    if sample_count > 1:
        sub_r2 = r2_score(y_test_sub, y_pred_sub)
        r2_str = f"{sub_r2:.4f}"
    else:
        r2_str = "N/A (样本过少)"
    
    # format the output
    print(f"{label:<22} | {sample_count:<10} | {sub_mae:<10.2f} m | {sub_rmse:<10.2f} m | {sub_re:<9.2f}% | {r2_str}")

print("="*90)