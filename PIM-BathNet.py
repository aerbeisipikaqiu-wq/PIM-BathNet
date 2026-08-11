import os
# Environment conflict optimization config
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import cv2 
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path
from torch.utils.data import Dataset, DataLoader

## =================================================================
## 1. Extracted YOLOv8 core basic components (CBS block and C2f block)
## =================================================================

class CBS(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(CBS, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class Bottleneck(nn.Module):
    def __init__(self, c, shortcut=True):
        super(Bottleneck, self).__init__()
        self.cv1 = CBS(c, c, kernel_size=3, stride=1, padding=1)
        self.cv2 = CBS(c, c, kernel_size=3, stride=1, padding=1)
        self.add = shortcut

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class C2f(nn.Module):
    def __init__(self, in_channels, out_channels, n=1, shortcut=True):
        super(C2f, self).__init__()
        self.c = int(out_channels * 0.5)
        self.cv1 = CBS(in_channels, 2 * self.c, kernel_size=1, stride=1, padding=0)
        self.cv2 = CBS((2 + n) * self.c, out_channels, kernel_size=1, stride=1, padding=0)
        self.m = nn.ModuleList(Bottleneck(self.c, shortcut) for _ in range(n))

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class SPPF(nn.Module):
    def __init__(self, in_channels, out_channels, k=5):
        super(SPPF, self).__init__()
        c_ = in_channels // 2
        self.cv1 = CBS(in_channels, c_, 1, 1, 0)
        self.cv2 = CBS(c_ * 4, out_channels, 1, 1, 0)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        y3 = self.m(y2)
        return self.cv2(torch.cat([x, y1, y2, y3], 1))

## =================================================================
## 2. 🌟 Innovation: Physics-Aware Progressive Dynamic Feature Modulation (PAP-DFM) 🌟
## =================================================================

class PhysAnchorMLP(nn.Module):
    """ Multi-scale physics feature anchor: projects 4-dim physical features into control vectors for different stages """
    def __init__(self, phys_dim=4):
        super(PhysAnchorMLP, self).__init__()
        self.shared_fc = nn.Sequential(
            nn.Linear(phys_dim, 64),
            nn.LayerNorm(64),
            nn.SiLU()
        )
        self.to_stage1 = nn.Linear(64, 32)
        self.to_stage2 = nn.Linear(64, 64)
        self.to_stage3 = nn.Linear(64, 128)
        self.to_sppf   = nn.Linear(64, 128)

    def forward(self, phys_feats):
        feat = self.shared_fc(phys_feats)
        p1 = self.to_stage1(feat)
        p2 = self.to_stage2(feat)
        p3 = self.to_stage3(feat)
        p_sppf = self.to_sppf(feat)
        return p1, p2, p3, p_sppf


class SpatialChannelModulation(nn.Module):
    """ Spatial-channel co-modulator: receives physics control vectors, dynamically reshapes (Scale & Bias) image feature maps """
    def __init__(self, channels):
        super(SpatialChannelModulation, self).__init__()
        self.channels = channels
        
        self.channel_scale = nn.Sequential(
            nn.Linear(channels, channels),
            nn.Sigmoid()
        )
        self.spatial_bias = nn.Sequential(
            nn.Conv2d(channels, 1, kernel_size=1, bias=True),
            nn.Tanh()
        )

    def forward(self, x, phys_vec):
        c_weight = self.channel_scale(phys_vec).unsqueeze(-1).unsqueeze(-1) 
        x = x * c_weight
        
        spatial_input = phys_vec.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, x.size(2), x.size(3))
        s_bias = self.spatial_bias(spatial_input) 
        x = x + x * s_bias
        
        return x

## =================================================================
## 3. PAP-DFM Progressive Multimodal Collaborative Bathymetry Network
## =================================================================

class YOLOPapDfmRegNet(nn.Module):
    def __init__(self):
        super(YOLOPapDfmRegNet, self).__init__()
        
        self.phys_anchor = PhysAnchorMLP(phys_dim=4)
        
        self.stem = CBS(1, 16, kernel_size=3, stride=1, padding=1)
        self.stage1_conv = CBS(16, 32, kernel_size=3, stride=2, padding=1)
        self.stage1_c2f  = C2f(32, 32, n=1)
        self.mod1 = SpatialChannelModulation(channels=32) 
        
        self.stage2_conv = CBS(32, 64, kernel_size=3, stride=2, padding=1)
        self.stage2_c2f  = C2f(64, 64, n=2)
        self.mod2 = SpatialChannelModulation(channels=64) 
        
        self.stage3_conv = CBS(64, 128, kernel_size=3, stride=2, padding=1)
        self.stage3_c2f  = C2f(128, 128, n=2)
        self.mod3 = SpatialChannelModulation(channels=128) 
        
        self.sppf = SPPF(128, 128, k=5)
        self.mod_sppf = SpatialChannelModulation(channels=128) 
        
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.reg_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.SiLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1)
        )

    def forward(self, img, phys_feats):
        p1, p2, p3, p_sppf = self.phys_anchor(phys_feats)
        
        x = self.stem(img)
        x = self.stage1_c2f(self.stage1_conv(x))
        x = self.mod1(x, p1) 
        
        x = self.stage2_c2f(self.stage2_conv(x))
        x = self.mod2(x, p2) 
        
        x = self.stage3_c2f(self.stage3_conv(x))
        x = self.mod3(x, p3) 
        
        x = self.sppf(x)
        x = self.mod_sppf(x, p_sppf) 
        
        feat_out = self.pool(x)
        return self.reg_head(feat_out)

## =================================================================
## 4. Data Loader Supporting Multi-Source Path Addressing and Deep Water Data Merging
## =================================================================

class SARDataset(Dataset):
    def __init__(self, csv_dir, patch_dir, t_idx_list, supply_dir=None, is_train=False):
        self.patch_dir = patch_dir
        self.supply_dir = supply_dir
        self.is_train = is_train
        all_dfs = []
        
        for t_idx in t_idx_list:
            csv_name = f"SAR_Transect_{t_idx:02d}.csv"
            csv_path = os.path.join(csv_dir, csv_name)
            
            if os.path.exists(csv_path):
                df_single = pd.read_csv(csv_path)
                df_single['transect_idx'] = t_idx
                df_single['is_supply'] = False
                all_dfs.append(df_single)
                
        if len(all_dfs) == 0:
            raise ValueError(f"❌ Error: No CSV files found in path {csv_dir}!")
            
        base_df = pd.concat(all_dfs, ignore_index=True)
        
        required_cols = ['s_idx', 'Peak_dist', 'Peak_Intensity', 'lambda', 'theta_deg', 'href']
        base_df = base_df.dropna(subset=required_cols).reset_index(drop=True)
        base_df['abs_href'] = base_df['href'].abs()
        
        if self.is_train and supply_dir and os.path.exists(supply_dir):
            supply_dfs = []
            print(f"\n📥 [Data Integration] Supplementary training data source detected, scanning {supply_dir} ...")
            
            for file_name in os.listdir(supply_dir):
                if file_name.startswith("SAR_Transect_") and file_name.endswith(".csv"):
                    try:
                        t_idx = int(file_name.split("_")[2].split(".")[0])
                    except:
                        t_idx = 99 
                        
                    csv_path = os.path.join(supply_dir, file_name)
                    df_supply_single = pd.read_csv(csv_path)
                    df_supply_single['transect_idx'] = t_idx
                    df_supply_single['is_supply'] = True  
                    supply_dfs.append(df_supply_single)
            
            if len(supply_dfs) > 0:
                supply_df = pd.concat(supply_dfs, ignore_index=True)
                supply_df = supply_df.dropna(subset=required_cols).reset_index(drop=True)
                supply_df['abs_href'] = supply_df['href'].abs()
                
                print(f"    💡 Successfully loaded {len(supply_df)} deep-water (60-100m) supplementary samples")
                self.df = pd.concat([base_df, supply_df], ignore_index=True)
            else:
                print("    ⚠️ Warning: No CSV files matching the naming convention found in supply_data folder.")
                self.df = base_df
        else:
            self.df = base_df
            
        if self.is_train and len(self.df) > 0:
            min_d = self.df['abs_href'].min()
            max_d = self.df['abs_href'].max()
            dynamic_bounds = np.linspace(min_d, max_d, 6)
            area4_start_bound = dynamic_bounds[3] 
            
            deep_water_mask = self.df['abs_href'] >= area4_start_bound
            deep_samples = self.df[deep_water_mask]
            if len(deep_samples) > 0:
                print(f"    💡 [Data Augmentation] Mixed-domain dynamic deep-water threshold: {area4_start_bound:.2f}m")
                print(f"    💡 Global deep water (Zone 4 & 5) has {len(deep_samples)} samples, applying standard 2x oversampling...")
                self.df = pd.concat([self.df, deep_samples], ignore_index=True)
        
    def __len__(self):
        return len(self.df)
        
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        t_idx = int(row['transect_idx'])
        s_idx = int(row['s_idx'])
        patch_name = f"SAR_T{t_idx:02d}_S{s_idx:03d}.png"
        
        if row['is_supply']:
            img_path = os.path.join(self.supply_dir, patch_name)
        else:
            img_path = os.path.join(self.patch_dir, patch_name)
        
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"❌ Failed to read image patch! Check path: {img_path}")
        
        assert img.shape == (128, 128)
        
        img = img.astype(np.float32) / 255.0
        img_tensor = torch.tensor(img, dtype=torch.float32).unsqueeze(0)
        
        phys_feats = torch.tensor([row['Peak_dist'], row['Peak_Intensity'], row['lambda'], row['theta_deg']], dtype=torch.float32)
        target = torch.tensor([row['abs_href']], dtype=torch.float32)
        
        return img_tensor, phys_feats, target

## =================================================================
## 5. Main Control Flow and Multi-Dimensional Evaluation Engine (with R2 metric)
## =================================================================

def main():
    BATCH_SIZE = 64
    EPOCHS = 100            
    LEARNING_RATE = 0.001
    

    ROOT = Path(__file__).parent
    csv_dir_path = ROOT / "python2"
    patch_dir_path = ROOT / "python"

    supply_dir_path = ROOT / "supply_data"

    model_save_path = ROOT / "best_yolo_fixed_transect_model.pth"



    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"⚡ Compute device: [{device}]")
    
    val_transects = list(range(2, 50, 5))  
    total_transects = list(range(0, 50))
    train_transects = sorted([t for t in total_transects if t not in val_transects])
    
    print("\n================ 📐 Transect Fixed Split Report ================")
    print(f"📡 Fixed validation transect IDs ({len(val_transects)}): {val_transects}")
    print(f"🏋️ Auto-generated training transect IDs ({len(train_transects)}): {train_transects}")
    print("=============================================================")
    
    train_dataset = SARDataset(csv_dir=csv_dir_path, patch_dir=patch_dir_path, t_idx_list=train_transects, supply_dir=supply_dir_path, is_train=True)
    val_dataset = SARDataset(csv_dir=csv_dir_path, patch_dir=patch_dir_path, t_idx_list=val_transects, supply_dir=None, is_train=False)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    train_min = train_dataset.df['abs_href'].min()
    train_max = train_dataset.df['abs_href'].max()
    train_bounds = np.linspace(train_min, train_max, 6)
    
    model = YOLOPapDfmRegNet().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    
    best_val_loss = float('inf')
    early_stop_payout = 20  
    patience_counter = 0
    
    print(f"\n🔥 Algorithm config: [PAP-DFM Physics-Aware Dynamic Feature Modulation + Global Evaluation Enhancement]")
    print("\n================ Starting Fixed Transect Isolation Training (100 Epochs) ================")
    
    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0.0
        for imgs, phys, targets in train_loader:
            imgs, phys, targets = imgs.to(device), phys.to(device), targets.to(device)
            optimizer.zero_grad()
            
            outputs = model(imgs, phys)
            loss = nn.functional.mse_loss(outputs, targets)
            
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * imgs.size(0)
            
        model.eval()
        val_loss = 0.0
        val_mae = 0.0  
        val_relative_error_sum = 0.0 
        
        # Container for epoch-level global R2 computation
        epoch_true_list = []
        epoch_pred_list = []
        
        with torch.no_grad(): 
            for imgs, phys, targets in val_loader:
                imgs, phys, targets = imgs.to(device), phys.to(device), targets.to(device)
                outputs = model(imgs, phys)
                
                base_loss = nn.functional.mse_loss(outputs, targets)
                val_loss += base_loss.item() * imgs.size(0)
                val_mae += torch.sum(torch.abs(outputs - targets)).item()
                
                batch_re = torch.abs(outputs - targets) / (targets + 1e-5)
                val_relative_error_sum += torch.sum(batch_re).item()
                
                epoch_true_list.extend(targets.cpu().numpy().flatten())
                epoch_pred_list.extend(outputs.cpu().numpy().flatten())
                
        epoch_train_loss = train_loss / len(train_dataset)
        epoch_val_loss = val_loss / len(val_dataset)
        epoch_val_mae = val_mae / len(val_dataset)
        
        epoch_val_rmse = np.sqrt(epoch_val_loss)
        epoch_val_re = (val_relative_error_sum / len(val_dataset)) * 100.0
        
        # Compute global R2 metric for the current epoch validation set
        ep_true = np.array(epoch_true_list)
        ep_pred = np.array(epoch_pred_list)
        if len(ep_true) > 1 and np.var(ep_true) > 1e-6:
            epoch_val_r2 = 1.0 - (np.sum((ep_true - ep_pred) ** 2) / np.sum((ep_true - np.mean(ep_true)) ** 2))
        else:
            epoch_val_r2 = 0.0
        
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        # Print with global Val R2 metric embedded
        print(f"Epoch [{epoch:03d}/{EPOCHS:03d}] | "
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
            print(f" -> 🎉 [Fixed Transect Isolation] Validation loss hit a new record, weights saved.")
        else:
            patience_counter += 1
            
        if patience_counter >= early_stop_payout:
            print(f"\n🛑 [Early Stopping] No improvement for {early_stop_payout} consecutive epochs, terminating early.")
            break
            
    print("\n================ 🏁 Training complete, starting comprehensive multi-depth-bin error evaluation on all validation transects ================")
    
    if len(val_dataset) > 0:
        model.load_state_dict(torch.load(model_save_path))
        model.eval()
        
        all_true_depths = []
        all_pred_depths = []
        
        with torch.no_grad():
            for imgs, phys, targets in val_loader:
                imgs, phys = imgs.to(device), phys.to(device)
                preds = model(imgs, phys)
                all_true_depths.extend(targets.cpu().numpy().flatten())
                all_pred_depths.extend(preds.cpu().numpy().flatten())
        
        all_true_depths = np.array(all_true_depths)
        all_pred_depths = np.array(all_pred_depths)
        all_abs_errors = np.abs(all_pred_depths - all_true_depths)
        all_sq_errors = (all_pred_depths - all_true_depths) ** 2
        all_rel_errors = all_abs_errors / (all_true_depths + 1e-5)
        
        # Widen table to accommodate R2 field
        print("-" * 145)
        print(" 📊【All validation transects: Multi-depth-bin error analysis - Unified Physics-Aligned Evaluation Table】")
        print("-" * 145)
        print(f"{'Depth Bin':<10}\t{'Depth Range':<15}\t{'Samples':<15}\t{'MAE':<15}\t{'RMSE':<15}\t{'RE':<15}\t{'R2 Score'}")
        print("-" * 145)
        
        for i in range(5):
            low_b = train_bounds[i]
            high_b = train_bounds[i+1]
            
            if i == 4:
                mask = (all_true_depths >= low_b) & (all_true_depths <= high_b)
            else:
                mask = (all_true_depths >= low_b) & (all_true_depths < high_b)
            
            regional_count = np.sum(mask)
            if regional_count > 0:
                regional_mae = np.mean(all_abs_errors[mask])
                regional_rmse = np.sqrt(np.mean(all_sq_errors[mask]))
                regional_re = np.mean(all_rel_errors[mask]) * 100.0
                
                # Local depth-bin R2 computation logic
                y_true_sub = all_true_depths[mask]
                y_pred_sub = all_pred_depths[mask]
                if len(y_true_sub) > 1 and np.var(y_true_sub) > 1e-6:
                    regional_r2 = 1.0 - (np.sum((y_true_sub - y_pred_sub) ** 2) / np.sum((y_true_sub - np.mean(y_true_sub)) ** 2))
                    r2_str = f"{regional_r2:.4f}"
                else:
                    r2_str = "N/A (low variance)"
                
                mae_str = f"{regional_mae:.2f} m"
                rmse_str = f"{regional_rmse:.2f} m"
                re_str = f"{regional_re:.2f}%"
            else:
                mae_str = "0.00 m"
                rmse_str = "0.00 m"
                re_str = "0.00%"
                r2_str = "0.0000"
                
            print(f"Zone {i+1:<4}\t{low_b:5.1f}m ~ {high_b:5.1f}m\t\t{regional_count:<10}\t{mae_str:<15}\t{rmse_str:<15}\t{re_str:<15}\t{r2_str}")
            
        print("=" * 145)
    else:
        print("⚠️ Warning: No valid validation full-section data available.")
        
    print("\n================ 🏁 Experiment Complete ================")

if __name__ == '__main__':
    main()