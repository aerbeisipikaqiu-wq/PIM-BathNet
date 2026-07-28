import os
# Environment conflict optimization config
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
import cv2  
import numpy as np
import pandas as pd
import scipy.io as sio
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
## 🔄 [Change]: Import sklearn for R2 score computation
from sklearn.metrics import r2_score

## ==========================================
## 1. Define Channel and Spatial Dual Attention Mechanism (CBAM)
## ==========================================
class CBAM(nn.Module):
    def __init__(self, channels, reduction=16):
        super(CBAM, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels, bias=False)
        )
        self.spatial_conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)

    def forward(self, x):
        b, c, h, w = x.size()
        max_pool = F.adaptive_max_pool2d(x, 1).view(b, c)
        avg_pool = F.adaptive_avg_pool2d(x, 1).view(b, c)
        channel_att = torch.sigmoid(self.fc(max_pool) + self.fc(avg_pool)).view(b, c, 1, 1)
        x = x * channel_att
        
        max_spatial = torch.max(x, dim=1, keepdim=True)[0]
        avg_spatial = torch.mean(x, dim=1, keepdim=True)
        spatial_att = torch.sigmoid(self.spatial_conv(torch.cat([max_spatial, avg_spatial], dim=1)))
        return x * spatial_att

## ==========================================
## 2. Adaptive Gated Multimodal Fusion Module (with warmup mechanism)
## ==========================================
class WarmupGatedFusion(nn.Module):
    def __init__(self, img_dim=64, phys_dim=32, out_dim=64):
        super(WarmupGatedFusion, self).__init__()
        self.gate_network = nn.Sequential(
            nn.Linear(img_dim + phys_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
            nn.Softmax(dim=1)
        )
        self.proj_img = nn.Linear(img_dim, out_dim)
        self.proj_phys = nn.Linear(phys_dim, out_dim)
        
    def forward(self, img_feat, phys_feat, warmup_mode=False):
        img_aligned = self.proj_img(img_feat)
        phys_aligned = self.proj_phys(phys_feat)
        
        if warmup_mode:
            fused_out = 0.5 * img_aligned + 0.5 * phys_aligned
        else:
            combined = torch.cat([img_feat, phys_feat], dim=1)
            gate_weights = self.gate_network(combined)
            weight_img = gate_weights[:, 0].unsqueeze(1)
            weight_phys = gate_weights[:, 1].unsqueeze(1)
            fused_out = weight_img * img_aligned + weight_phys * phys_aligned
            
        return fused_out

## ==========================================
## 3. Ultimate Multimodal MoE-Physics Network (includes directions B, C, D)
## ==========================================
class PhysicsMoSARNet(nn.Module):
    def __init__(self):
        super(PhysicsMoSARNet, self).__init__()
        
        # Image feature extraction branch
        self.cnn_branch = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1), 
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),                                      
            CBAM(16),                                             
            
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),                                      
            CBAM(32),                                             
            
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),                                      
            CBAM(64),                                             
            
            nn.AdaptiveAvgPool2d((1, 1)),                         
            nn.Flatten(),                                         
            nn.Linear(64, 64),                                    
            nn.ReLU(),
            nn.Dropout(0.4)                                       
        )
        
        self.mlp_branch = nn.Sequential(
            nn.Linear(28, 32),
            nn.ReLU(),
            nn.Linear(32, 32),                                    
            nn.ReLU()
        )
        
        # Gating component
        self.gated_fusion = WarmupGatedFusion(img_dim=64, phys_dim=32, out_dim=64)
        
        # [Direction B]: MoE multi-expert decision system
        self.expert_shallow = nn.Sequential(
            nn.Linear(64, 32), nn.ReLU(), nn.Dropout(0.3), nn.Linear(32, 1)
        )
        self.expert_transition = nn.Sequential(
            nn.Linear(64, 32), nn.ReLU(), nn.Dropout(0.3), nn.Linear(32, 1)
        )
        self.expert_deep = nn.Sequential(
            nn.Linear(64, 32), nn.ReLU(), nn.Dropout(0.3), nn.Linear(32, 1)
        )
        
        # Expert routing router
        self.router = nn.Sequential(
            nn.Linear(28, 16),
            nn.ReLU(),
            nn.Linear(16, 3),
            nn.Softmax(dim=1)
        )

    def forward(self, img, phys_feats_28d, warmup_mode=False):
        img_feat = self.cnn_branch(img)
        phys_feat = self.mlp_branch(phys_feats_28d)
        
        fused_feat = self.gated_fusion(img_feat, phys_feat, warmup_mode=warmup_mode)
        
        out_shallow = self.expert_shallow(fused_feat)
        out_trans = self.expert_transition(fused_feat)
        out_deep = self.expert_deep(fused_feat)
        
        route_weights = self.router(phys_feats_28d)
        
        final_output = (route_weights[:, 0].unsqueeze(1) * out_shallow +
                        route_weights[:, 1].unsqueeze(1) * out_trans +
                        route_weights[:, 2].unsqueeze(1) * out_deep)
        
        return final_output

## ==========================================================================
## 4. Upgraded Multi-CSV Distributed Data Loader (calibrated feature engineering)
## ==========================================================================
class SARDataset(Dataset):
    def __init__(self, csv_dir, patch_dir, allowed_transects=None, is_train=False):
        print(f"[Dataset] Dynamically integrating distributed CSV physical data from {csv_dir}...")
        self.patch_dir = patch_dir
        all_dfs = []
        
        for t_idx in range(50):
            file_name = f"SAR_Transect_{t_idx:02d}.csv"
            full_path = os.path.join(csv_dir, file_name)
            
            if os.path.exists(full_path):
                try:
                    df_single = pd.read_csv(full_path)
                    df_single['transect_idx'] = t_idx
                    all_dfs.append(df_single)
                except Exception as e:
                    print(f"⚠️ Failed to read CSV (skipped): {full_path}. Reason: {e}")

        if len(all_dfs) == 0:
            raise FileNotFoundError(f"❌ Error: No valid SAR_Transect_XX.csv files found in target path {csv_dir}!")
            
        combined_df = pd.concat(all_dfs, ignore_index=True)
        required_cols = ['transect_idx', 's_idx', 'Peak_dist', 'Peak_Intensity', 'lambda', 'theta_deg', 'href']
        self.df = combined_df.dropna(subset=required_cols).reset_index(drop=True)
        
        if allowed_transects is not None:
            self.df = self.df[self.df['transect_idx'].isin(allowed_transects)].reset_index(drop=True)
            
        if is_train:
            mask_region4 = (self.df['href'].abs() >= 60.4) & (self.df['href'].abs() < 80.1)
            mask_region5 = self.df['href'].abs() >= 80.1
            df_r4 = self.df[mask_region4].copy()
            df_r5 = self.df[mask_region5].copy()
            self.df = pd.concat([self.df, df_r4, df_r5, df_r5], ignore_index=True).reset_index(drop=True)
            
        print(f"🎉 Dataset built successfully! Valid samples: {len(self.df)}")
        
    def __len__(self):
        return len(self.df)
        
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        t_idx = int(row['transect_idx'])
        s_idx = int(row['s_idx'])
        patch_name = f"SAR_T{t_idx:02d}_S{s_idx:03d}.png"
            
        img_path = os.path.join(self.patch_dir, patch_name)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"❌ Patch not found: {img_path}")
        
        img = img.astype(np.float32) / 255.0
        img_tensor = torch.tensor(img, dtype=torch.float32).unsqueeze(0)
        
        dist = float(row['Peak_dist'])
        intns = float(row['Peak_Intensity'])
        lam = float(row['lambda'])
        theta = float(row['theta_deg'])
        theta_rad = np.radians(theta)
        
        feats_list = [dist, intns, lam, theta]
        feats_list += [np.sin(theta_rad), np.cos(theta_rad), 1.0 / (lam + 1e-5), dist * intns]
        feats_list += [lam * np.sin(theta_rad), dist / (lam + 1e-5), intns * np.cos(theta_rad), (lam**2)]
        
        for freq in [1, 2, 4, 8]:
            feats_list.append(np.sin(freq * theta_rad))
            feats_list.append(np.cos(freq * theta_rad))
            feats_list.append(np.sin(freq * lam / 100.0))
            feats_list.append(np.cos(freq * lam / 100.0))
            
        phys_feats_28d = torch.tensor(feats_list, dtype=torch.float32)
        target = torch.tensor([abs(row['href'])], dtype=torch.float32)
        
        return img_tensor, phys_feats_28d, target

## ==========================================
## 5. Main Training Control Flow
## ==========================================
def main():
    print("[Debug 1] Successfully entered main()...")
    
    BATCH_SIZE = 64
    EPOCHS = 70  
    LEARNING_RATE = 0.001
    
    csv_dir_path = r'D:\SAR_database\python2'   
    patch_dir_path = r'D:\SAR_database\python'    
    model_save_path = r'D:\SAR_database\python2\best_advanced_cnn_model3.pth'
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[Debug 2] Compute device set to: [{device}]")
    
    all_transect_ids = list(range(50)) 
    val_transects = list(range(2, 50, 5)) 
    train_transects = [t for t in all_transect_ids if t not in val_transects]
    
    train_dataset = SARDataset(csv_dir=csv_dir_path, patch_dir=patch_dir_path, allowed_transects=train_transects, is_train=True)
    val_dataset = SARDataset(csv_dir=csv_dir_path, patch_dir=patch_dir_path, allowed_transects=val_transects, is_train=False)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=False, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=False, num_workers=0)
    
    print("[Debug 7] Initializing advanced physics-informed MoE mixture-of-experts network...")
    model = PhysicsMoSARNet().to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    
    best_val_loss = float('inf')
    early_stop_patience = 18  
    patience_counter = 0      
    
    print("\n================ Starting Formal Training (Directions B+C+D Stable Aligned - 70 Epochs) ================")
    
    for epoch in range(1, EPOCHS + 1):
        warmup_active = (epoch <= 10)
        
        model.train()
        train_loss = 0.0
        for imgs, phys_28d, targets in train_loader:
            imgs = imgs.to(device)
            phys_28d = phys_28d.to(device)
            targets = targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(imgs, phys_28d, warmup_mode=warmup_active)
            
            loss_weights = torch.ones_like(targets)
            loss_weights = torch.where((targets >= 60.4) & (targets < 80.1), 2.0, loss_weights)
            loss_weights = torch.where(targets >= 80.1, 4.5, loss_weights)
            
            raw_loss = (outputs - targets) ** 2
            weighted_loss = torch.mean(raw_loss * loss_weights)
            
            weighted_loss.backward()
            optimizer.step()
            
            train_loss += torch.mean(raw_loss).item() * imgs.size(0)
            
        model.eval()
        
        ## 🔄 [Change]: Collect all predictions and targets during validation for multi-metric and precise R2 computation
        val_preds_list = []
        val_targets_list = []
        
        with torch.no_grad(): 
            for imgs, phys_28d, targets in val_loader:
                imgs = imgs.to(device)
                phys_28d = phys_28d.to(device)
                targets = targets.to(device)
                
                outputs = model(imgs, phys_28d, warmup_mode=False)
                
                val_preds_list.append(outputs.cpu().numpy())
                val_targets_list.append(targets.cpu().numpy())
                
        epoch_train_loss = train_loss / len(train_dataset)
        
        ## 🔄 [Change]: Combine collected data into numpy arrays to compute 5 core metrics
        val_preds_all = np.vstack(val_preds_list).flatten()
        val_targets_all = np.vstack(val_targets_list).flatten()
        
        epoch_val_loss = np.mean((val_preds_all - val_targets_all) ** 2)  # MSE
        epoch_val_rmse = np.sqrt(epoch_val_loss)                          # RMSE
        epoch_val_mae = np.mean(np.abs(val_preds_all - val_targets_all))  # MAE
        epoch_val_re = np.mean(np.abs(val_preds_all - val_targets_all) / (val_targets_all + 1e-5)) * 100.0  # RE%
        epoch_val_r2 = r2_score(val_targets_all, val_preds_all)           # R2
        
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        warmup_str = "[🔥 Gate Warmup]" if warmup_active else "[🧠 Full Adaptive]"
        ## 🔄 [Change]: Updated log output to include MSE, RMSE, MAE, RE, R2
        print(f"Epoch [{epoch:02d}/{EPOCHS:02d}] {warmup_str} | "
              f"Train MSE: {epoch_train_loss:.4f} | "
              f"Val MSE: {epoch_val_loss:.4f} | "
              f"Val RMSE: {epoch_val_rmse:.4f} | "
              f"Val MAE: {epoch_val_mae:.2f} m | "
              f"Val RE: {epoch_val_re:.2f}% | "
              f"Val R2: {epoch_val_r2:.4f} | "
              f"LR: {current_lr:.6f}")
        
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            patience_counter = 0  
            torch.save(model.state_dict(), model_save_path)
            print(f" -> 🎉 Validation loss reached a new low, model weights saved.")
        else:
            patience_counter += 1
            
        if patience_counter >= early_stop_patience:
            print(f"\n🛑 [Early Stopping] Triggered to protect best weights!")
            break
            
    print("\n================ Training Phase Complete ================")
    
    # ---- Full validation set macro stratified statistics module ----
    print(f"\n🔄 Loading best model weights from disk: {model_save_path} ...")
    model.load_state_dict(torch.load(model_save_path, map_location=device))
    model.eval()
    
    all_val_records = []
    with torch.no_grad():
        for i in range(len(val_dataset)):
            img_tensor, phys_28d, target_tensor = val_dataset[i]
            
            img_in = img_tensor.unsqueeze(0).to(device)
            phys_28d_in = phys_28d.unsqueeze(0).to(device)
            
            prediction = model(img_in, phys_28d_in, warmup_mode=False).item()
            true_depth = target_tensor.item()
            
            ## 🔄 [Change]: Record dict now includes raw predictions and true values for per-bin R2 computation
            all_val_records.append({
                'true_depth': true_depth,
                'pred_depth': prediction,
                'abs_error': abs(prediction - true_depth),
                'squared_error': (prediction - true_depth) ** 2,
                'relative_error': abs(prediction - true_depth) / (true_depth + 1e-5)
            })
                
    df_res = pd.DataFrame(all_val_records)
    min_d, max_d = df_res['true_depth'].min(), df_res['true_depth'].max()
    bins = np.linspace(min_d, max_d, 6)
    df_res['depth_bin'] = pd.cut(df_res['true_depth'], bins=bins, labels=False, include_lowest=True)
    
    bin_summary = {}
    for b_id in range(5):
        df_bin = df_res[df_res['depth_bin'] == b_id]
        bin_count = len(df_bin)
        
        if bin_count > 0:
            calculated_mae = df_bin['abs_error'].mean()
            calculated_mse = df_bin['squared_error'].mean()
            calculated_rmse = np.sqrt(calculated_mse)
            calculated_re = df_bin['relative_error'].mean() * 100.0
            
            ## 🔄 [Change]: Add per-bin R2 computation logic (set to 0 if fewer than 2 samples)
            if bin_count > 1:
                calculated_r2 = r2_score(df_bin['true_depth'], df_bin['pred_depth'])
            else:
                calculated_r2 = 0.0
            
            bin_summary[b_id] = {
                'range': f"{bins[b_id]:.1f}m ~ {bins[b_id+1]:.1f}m",
                'total': bin_count,
                'mae': calculated_mae,
                'mse': calculated_mse,
                'rmse': calculated_rmse,
                're': calculated_re,
                'r2': calculated_r2
            }
        else:
            bin_summary[b_id] = {'range': f"{bins[b_id]:.1f}m ~ {bins[b_id+1]:.1f}m", 'total': 0, 'mae': 0.0, 'mse': 0.0, 'rmse': 0.0, 're': 0.0, 'r2': 0.0}
            
    ## 🔄 [Change]: Beautified table headers, added MSE and R2 columns
    print("\n" + "="*125)
    print(f" 📊 [Full validation set: Comprehensive multi-depth-bin error analysis (MSE / RMSE / MAE / RE / R2)]")
    print(f" Validation set overall depth range: {min_d:.2f}m to {max_d:.2f}m")
    print("-"*125)
    print(f"{'Depth Bin':<10}{'True Depth Range':<22}{'Samples':<12}{'MSE':<16}{'RMSE':<16}{'MAE':<18}{'RE(%)':<16}{'R2 Score'}")
    print("-"*125)
    for b_id, info in bin_summary.items():
        print(f"Zone {b_id+1}    "
              f"{info['range']:<22}"
              f"{info['total']:<12}"
              f"{info['mse']:<16.2f}"
              f"{info['rmse']:<16.2f}"
              f"{info['mae']:<18.2f}m   "
              f"{info['re']:<16.2f}%"
              f"{info['r2']:.4f}")
    print("="*125 + "\n")

if __name__ == '__main__':
    main()