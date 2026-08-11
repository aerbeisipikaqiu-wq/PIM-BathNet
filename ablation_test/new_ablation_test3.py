import os
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
## 1. Extract the core basic components of YOLOv8 (CBS blocks and C2f blocks)
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
## 2. 🌟 Ablation experiment M3 unique core component: multi-level anchoring and only channel modulator 🌟
## =================================================================

class PhysAnchorMLP(nn.Module):
    """
    Multi-level Scale Anchoringer: Maps one-dimensional 4-dimensional physical features to the number of control channels required at different levels
    """
    def __init__(self, phys_dim=4, out_dim=32):
        super(PhysAnchorMLP, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(phys_dim, 32),
            nn.LayerNorm(32),
            nn.SiLU(),
            nn.Linear(32, out_dim),
            nn.SiLU()
        )
    def forward(self, p):
        return self.mlp(p)


class ChannelOnlyModulation(nn.Module):
    """
    🌟 M3 Ablation Core: Stripped spatial adaptive bias, retaining only channel dynamic scaling
    """
    def __init__(self, feat_channels, phys_cond_dim=32):
        super(ChannelOnlyModulation, self).__init__()
        # A linear layer used for generating the channel scaling factor
        self.channel_layer = nn.Linear(phys_cond_dim, feat_channels)
        self.gate = nn.Sigmoid()

    def forward(self, x, p_cond):
        # 1. Calculate the scaling weights of the channels [B, C] -> [B, C, 1, 1]
        scale = self.gate(self.channel_layer(p_cond)).unsqueeze(-1).unsqueeze(-1)
        
        # 2. Only implement channel multiplicative scaling (intentionally discarding the complete additive spatial bias)
        return x * scale


class PhyInNetM3(nn.Module):
    """
    Ablation Group M3: Physical Infiltration Network (Channel Modulation Variant)
    """
    def __init__(self):
        super(PhyInNetM3, self).__init__()
        
        # Physical stream decoupling anchor
        self.phys_anchor1 = PhysAnchorMLP(phys_dim=4, out_dim=32)
        self.phys_anchor2 = PhysAnchorMLP(phys_dim=4, out_dim=64)
        self.phys_anchor3 = PhysAnchorMLP(phys_dim=4, out_dim=128)
        
        # 🌟 Only Channel Modulator
        self.mod1 = ChannelOnlyModulation(feat_channels=32, phys_cond_dim=32)
        self.mod2 = ChannelOnlyModulation(feat_channels=64, phys_cond_dim=64)
        self.mod3 = ChannelOnlyModulation(feat_channels=128, phys_cond_dim=128)
        
        # Spatial stream backbone
        self.stem = CBS(1, 16, kernel_size=3, stride=1, padding=1)
        self.stage1_conv = CBS(16, 32, kernel_size=3, stride=2, padding=1)
        self.stage1_c2f  = C2f(32, 32, n=1)
        
        self.stage2_conv = CBS(32, 64, kernel_size=3, stride=2, padding=1)
        self.stage2_c2f  = C2f(64, 64, n=2)
        
        self.stage3_conv = CBS(64, 128, kernel_size=3, stride=2, padding=1)
        self.stage3_c2f  = C2f(128, 128, n=2)
        
        self.sppf = SPPF(128, 128, k=5)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Regression head
        self.reg_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.SiLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1)
        )

    def forward(self, img, phys_feats):
        # Physical mapping
        p1 = self.phys_anchor1(phys_feats)
        p2 = self.phys_anchor2(phys_feats)
        p3 = self.phys_anchor3(phys_feats)
        
        # Spatial stream forward propagation + layer-wise channel modulation
        x = self.stem(img)
        
        x = self.stage1_c2f(self.stage1_conv(x))
        x = self.mod1(x, p1)  # Stage 1 only channel modulation
        
        x = self.stage2_c2f(self.stage2_conv(x))
        x = self.mod2(x, p2)  # Stage 2 only channel modulation
        
        x = self.stage3_c2f(self.stage3_conv(x))
        x = self.mod3(x, p3)  # Stage 3 only channel modulation
        
        x = self.sppf(x)
        feat_out = self.pool(x)
        return self.reg_head(feat_out)

## =================================================================
## 3. A data loader that supports multi-source path addressing and deep-water data merging
## =================================================================

class SARDataset(Dataset):
    def __init__(self, csv_dir, patch_dir, t_idx_list, supply_dir=None, is_train=False):
        self.patch_dir = patch_dir
        self.supply_dir = supply_dir
        self.is_train = is_train
        all_dfs = []
        
        # A. Load original base transect data
        for t_idx in t_idx_list:
            csv_name = f"SAR_Transect_{t_idx:02d}.csv"
            csv_path = os.path.join(csv_dir, csv_name)
            
            if os.path.exists(csv_path):
                df_single = pd.read_csv(csv_path)
                df_single['transect_idx'] = t_idx
                df_single['is_supply'] = False
                all_dfs.append(df_single)
                
        if len(all_dfs) == 0:
            raise ValueError(f"❌ Error: No specified CSV files were found under the path {csv_dir}!")
            
        base_df = pd.concat(all_dfs, ignore_index=True)
        
        # Data clean
        required_cols = ['s_idx', 'Peak_dist', 'Peak_Intensity', 'lambda', 'theta_deg', 'href']
        base_df = base_df.dropna(subset=required_cols).reset_index(drop=True)
        base_df['abs_href'] = base_df['href'].abs()
        
        # B. supply data
        if self.is_train and supply_dir and os.path.exists(supply_dir):
            supply_dfs = []
            print(f"\n📥 [Data Integration] Detected training set supplementary data source, scanning {supply_dir} ...")
            
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
                
                print(f"    💡 Successfully loaded 60-100m deep-water supplementary samples: {len(supply_df)}")
                self.df = pd.concat([base_df, supply_df], ignore_index=True)
            else:
                print("    ⚠️ Warning: No CSV files matching the naming convention were found in the supply_data folder.")
                self.df = base_df
        else:
            self.df = base_df
            
        # C. Maintain the original logic of over-sampling for the deep water lines in regions 4 and 5
        if self.is_train and len(self.df) > 0:
            min_d = self.df['abs_href'].min()
            max_d = self.df['abs_href'].max()
            dynamic_bounds = np.linspace(min_d, max_d, 6)
            area4_start_bound = dynamic_bounds[3]
            
            deep_water_mask = self.df['abs_href'] >= area4_start_bound
            deep_samples = self.df[deep_water_mask]
            if len(deep_samples) > 0:
                print(f"    💡 [Data Augmentation] Hybrid Domain Dynamic Deep Water Line: {area4_start_bound:.2f}m")
                print(f"    💡 Global Deep Water (Area 4 and Area 5) Total: {len(deep_samples)} samples, performing standard double sampling...")
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
            raise FileNotFoundError(f"❌ Image patch reading failed! Check path: {img_path}")
        
        assert img.shape == (128, 128)
        
        img = img.astype(np.float32) / 255.0
        img_tensor = torch.tensor(img, dtype=torch.float32).unsqueeze(0)
        
        phys_feats = torch.tensor([row['Peak_dist'], row['Peak_Intensity'], row['lambda'], row['theta_deg']], dtype=torch.float32)
        target = torch.tensor([row['abs_href']], dtype=torch.float32)
        
        return img_tensor, phys_feats, target

## =================================================================
## 4. Main control flow and standard validation engine
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
    print(f"⚡ Computing device locking: 【{device}】")
    
    # Select the specified survey line as the validation set
    val_transects = list(range(2, 50, 5))  
    total_transects = list(range(0, 50))
    train_transects = sorted([t for t in total_transects if t not in val_transects])
    
    print("\n================ 📐 Line Alignment and Segmentation Report for Calibration ================")
    print(f"📡 Fixed the specified [validation set] line IDs ({len(val_transects)} lines): {val_transects}")
    print(f"🏋️ Automatically generated [training set] base line IDs ({len(train_transects)} lines): {train_transects}")
    print("========================================================")
    
    train_dataset = SARDataset(csv_dir=csv_dir_path, patch_dir=patch_dir_path, t_idx_list=train_transects, supply_dir=supply_dir_path, is_train=True)
    val_dataset = SARDataset(csv_dir=csv_dir_path, patch_dir=patch_dir_path, t_idx_list=val_transects, supply_dir=None, is_train=False)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    

    train_min = train_dataset.df['abs_href'].min()
    train_max = train_dataset.df['abs_href'].max()
    train_bounds = np.linspace(train_min, train_max, 6)
    
    # Initialize M3 for only channel modulation model
    model = PhyInNetM3().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    
    best_val_loss = float('inf')
    early_stop_patience = 20  
    patience_counter = 0
    
    print(f"\n Ablation group M3 activation: 【PhyIn-Net only channel scaling variant + standard global MSE control】")
    print("\n================ Enable fixed line separation training (100 epochs) ================")
    
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
        
        epoch_true_arr = np.array(epoch_true_list)
        epoch_pred_arr = np.array(epoch_pred_list)
        ss_res_epoch = np.sum((epoch_true_arr - epoch_pred_arr) ** 2)
        ss_tot_epoch = np.sum((epoch_true_arr - np.mean(epoch_true_arr)) ** 2)
        epoch_val_r2 = 1.0 - (ss_res_epoch / (ss_tot_epoch + 1e-5))
        
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
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
            print(f" -> 🎉 【Ablation Group M3】Validation loss refreshed, weights secured.")
        else:
            patience_counter += 1
            
        if patience_counter >= early_stop_patience:
            print(f"\n🛑 [Early Stopping] Continuous {early_stop_patience} rounds without improvement, ending early.")
            break
            
    print("\n================ 🏁 Training phase completed, initiating comprehensive error analysis across all validation lines ================")
    
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
        
        print("-" * 145)
        print(" 📊【M3 Ablation Group: Comprehensive Error Analysis Across All Validation Lines】")
        print("-" * 145)
        print(f"{'Depth Level':<10}\t{'Corresponding Depth Range':<20}\t{'Regional Sample Count':<12}\t{'Average Absolute Error (MAE)':<20}\t{'Root Mean Square Error (RMSE)':<20}\t{'Average Relative Error (RE)':<20}\t{'Determination Coefficient (R2)'}")
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
                
                reg_true = all_true_depths[mask]
                reg_pred = all_pred_depths[mask]
                ss_res_reg = np.sum((reg_true - reg_pred) ** 2)
                ss_tot_reg = np.sum((reg_true - np.mean(reg_true)) ** 2)
                
                regional_r2 = 1.0 - (ss_res_reg / (ss_tot_reg + 1e-5)) if ss_tot_reg > 0 else 0.0
                
                mae_str = f"{regional_mae:.2f} m"
                rmse_str = f"{regional_rmse:.2f} m"
                re_str = f"{regional_re:.2f}%"
                r2_str = f"{regional_r2:.4f}"
            else:
                mae_str = "0.00 m"
                rmse_str = "0.00 m"
                re_str = "0.00%"
                r2_str = "0.0000"
                
            print(f"regional {i+1:<4}\t{low_b:5.1f}m ~ {high_b:5.1f}m\t\t{regional_count:<12}\t{mae_str:<20}\t{rmse_str:<20}\t{re_str:<20}\t{r2_str}")
            
        print("=" * 145)
    else:
        print("⚠️ Warning: No valid full-section data for the validation set has been obtained.")
        
    print("\n================ 🏁 Experiment completed ================")

if __name__ == '__main__':
    main()