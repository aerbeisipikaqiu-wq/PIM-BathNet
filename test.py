import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import cv2
import sys
import glob
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# =================================================================
# 1. Underlying network components
# =================================================================

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
## 2.  Innovative Component: Progressive Dynamic Feature Modulation for Physical Sensing (PAP-DFM)
## =================================================================

class PhysAnchorMLP(nn.Module):
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
## 3. PAP-DFM Progressive Multi-modal Collaborative Water Depth Inversion Network
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

# =================================================================
# 2. Single-section independent data loader
# =================================================================

class SingleTransectDataset(Dataset):
    def __init__(self, csv_dir, patch_dir, t_idx):
        self.patch_dir = patch_dir
        csv_path = os.path.join(csv_dir, f"SAR_Transect_{t_idx:02d}.csv")
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"❌ No survey line file was found: {csv_path}")
        self.df = pd.read_csv(csv_path)
        self.df['transect_idx'] = t_idx
        required_cols = ['s_idx', 'Peak_dist', 'Peak_Intensity', 'lambda', 'theta_deg', 'href']
        self.df = self.df.dropna(subset=required_cols).reset_index(drop=True)
        self.df['abs_href'] = self.df['href'].abs()
        
    def __len__(self): return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        patch_name = f"SAR_T{int(row['transect_idx']):02d}_S{int(row['s_idx']):03d}.png"
        img_path = os.path.join(self.patch_dir, patch_name)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None: raise FileNotFoundError(f"❌ No patch was found: {img_path}")
        img = img.astype(np.float32) / 255.0
        img_tensor = torch.tensor(img, dtype=torch.float32).unsqueeze(0)
        phys_feats = torch.tensor([row['Peak_dist'], row['Peak_Intensity'], row['lambda'], row['theta_deg']], dtype=torch.float32)
        target = torch.tensor([row['abs_href']], dtype=torch.float32)
        return img_tensor, phys_feats, target

## =================================================================
## 5. Test Set Multi-dimensional Evaluation Engine
## =================================================================

def evaluate_test_set():
    # ---------------- Path Configuration (Please modify your test folder path here) ----------------
    patch_dir_path = r"D:\Python3.12\Github_code\python\T32"
    csv_dir_path   = r'D:\Python3.12\Github_code\python2'
    model_weight_path = r'D:\SAR_database\python2\best_yolo_fixed_transect_model.pth'
    
    
    BATCH_SIZE = 64
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"⚡ Evaluate the reasoning device: 【{device}】")
    
    if not os.path.exists(model_weight_path):
        raise FileNotFoundError(f"❌ No pre-trained weight file was found: {model_weight_path}")
        
    print(f"\n📂 Loading test set data:")
    
    test_dataset = SingleTransectDataset(patch_dir=patch_dir_path, csv_dir=csv_dir_path, t_idx=32)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    print(f"✅ Successfully extracted total test samples: {len(test_dataset)}")
    
    print("🧠 Initializing network structure and injecting optimal weights...")
    model = YOLOPapDfmRegNet().to(device)
    model.load_state_dict(torch.load(model_weight_path, map_location=device), strict=True)
    model.eval()
    print(f"🎉 Successfully loaded weight file: {model_weight_path}")
    
    all_true_depths = []
    all_pred_depths = []

    
    print("\n The model inference is currently being performed on the test set....")
    with torch.no_grad():
        for imgs, phys, targets in test_loader:
            imgs, phys = imgs.to(device), phys.to(device)
            preds = model(imgs, phys)
            
            all_true_depths.extend(targets.cpu().numpy().flatten())
            all_pred_depths.extend(preds.cpu().numpy().flatten())
            
    all_true_depths = np.array(all_true_depths)
    all_pred_depths = np.array(all_pred_depths)
    all_abs_errors = np.abs(all_pred_depths - all_true_depths)
    all_sq_errors = (all_pred_depths - all_true_depths) ** 2
    all_rel_errors = all_abs_errors / (all_true_depths + 1e-5)
    
    # 1. Calculate global evaluation indicators
    overall_mse = np.mean(all_sq_errors)
    overall_rmse = np.sqrt(overall_mse)
    overall_mae = np.mean(all_abs_errors)
    overall_re = np.mean(all_rel_errors) * 100.0
    
    if len(all_true_depths) > 1 and np.var(all_true_depths) > 1e-6:
        overall_r2 = 1.0 - (np.sum(all_sq_errors) / np.sum((all_true_depths - np.mean(all_true_depths)) ** 2))
    else:
        overall_r2 = 0.0
        
    print("\n================ 🏆 【Overall evaluation results of the test set】 ================")
    print(f" 📊 Total test samples : {len(all_true_depths)} ")
    print(f" 📉 Mean Square Error (MSE) : {overall_mse:.4f}")
    print(f" 📐 Root Mean Square Error (RMSE): {overall_rmse:.4f} m")
    print(f" 🎯 Mean Absolute Error (MAE): {overall_mae:.4f} m")
    print(f" 🌀 Mean Relative Error (RE) : {overall_re:.2f}%")
    print(f" 📈 Coefficient of Determination (R²)    : {overall_r2:.4f}")
    print("================================================================")

    # 2. Split the assessment into 5 equal-depth intervals
    min_d, max_d = all_true_depths.min(), all_true_depths.max()
    bounds = np.linspace(min_d, max_d, 6)
    
    print("\n" + "-" * 145)
    print(" 📊【Test Set Different Water Depth Interval Inversion Error Detailed Evaluation Table】")
    print("-" * 145)
    print(f"{'Water Depth Level':<10}\t{'Corresponding Reference Water Depth Range':<15}\t{'Regional Sample Count':<15}\t{'Average Absolute Error (MAE)':<15}\t{'Root Mean Square Error (RMSE)':<15}\t{'Average Relative Error (RE)':<15}\t{'Coefficient of Determination (R²)'}")
    print("-" * 145)
    
    for i in range(5):
        low_b = bounds[i]
        high_b = bounds[i+1]
        
        if i == 4:
            mask = (all_true_depths >= low_b) & (all_true_depths <= high_b)
        else:
            mask = (all_true_depths >= low_b) & (all_true_depths < high_b)
            
        regional_count = np.sum(mask)
        if regional_count > 0:
            regional_mae = np.mean(all_abs_errors[mask])
            regional_rmse = np.sqrt(np.mean(all_sq_errors[mask]))
            regional_re = np.mean(all_rel_errors[mask]) * 100.0
            
            y_true_sub = all_true_depths[mask]
            y_pred_sub = all_pred_depths[mask]
            if len(y_true_sub) > 1 and np.var(y_true_sub) > 1e-6:
                regional_r2 = 1.0 - (np.sum((y_true_sub - y_pred_sub) ** 2) / np.sum((y_true_sub - np.mean(y_true_sub)) ** 2))
                r2_str = f"{regional_r2:.4f}"
            else:
                r2_str = "N/A (Variance too low)"
                
            mae_str = f"{regional_mae:.2f} m"
            rmse_str = f"{regional_rmse:.2f} m"
            re_str = f"{regional_re:.2f}%"
        else:
            mae_str, rmse_str, re_str, r2_str = "0.00 m", "0.00 m", "0.00%", "0.0000"
            
        print(f"Region {i+1:<4}\t{low_b:5.1f}m ~ {high_b:5.1f}m\t\t{regional_count:<10}\t{mae_str:<15}\t{rmse_str:<15}\t{re_str:<15}\t{r2_str}")
        
    print("=" * 145)

if __name__ == '__main__':
    evaluate_test_set()