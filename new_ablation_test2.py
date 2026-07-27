import os
# 环境冲突优化配置
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import cv2 
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

## =================================================================
## 1. 抽取 YOLOv8 核心基础组件 (CBS 块 与 C2f 块)
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
## 2. 消融实验 M2：末端特征拼接多模态网络 (Late Fusion Baseline)
## =================================================================

class YOLOLateFusionBaseNet(nn.Module):
    def __init__(self):
        super(YOLOLateFusionBaseNet, self).__init__()
        
        # 空间流主干网络（隔离运行）
        self.stem = CBS(1, 16, kernel_size=3, stride=1, padding=1)
        self.stage1_conv = CBS(16, 32, kernel_size=3, stride=2, padding=1)
        self.stage1_c2f  = C2f(32, 32, n=1)
        self.stage2_conv = CBS(32, 64, kernel_size=3, stride=2, padding=1)
        self.stage2_c2f  = C2f(64, 64, n=2)
        self.stage3_conv = CBS(64, 128, kernel_size=3, stride=2, padding=1)
        self.stage3_c2f  = C2f(128, 128, n=2)
        self.sppf = SPPF(128, 128, k=5)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # 输入维度变更为 128 (图像) + 4 (物理标量) = 132
        self.reg_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(132, 64), 
            nn.LayerNorm(64),
            nn.SiLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1)
        )

    def forward(self, img, phys_feats):
        # 1. 空间流独立提取特征
        x = self.stem(img)
        x = self.stage1_c2f(self.stage1_conv(x))
        x = self.stage2_c2f(self.stage2_conv(x))
        x = self.stage3_c2f(self.stage3_conv(x))
        x = self.sppf(x)
        feat_out = self.pool(x) # [B, 128, 1, 1]
        feat_flat = torch.flatten(feat_out, 1) # [B, 128]
        
        # 2. 尾部特征粗暴拼接 (Late Fusion)
        joint_feat = torch.cat([feat_flat, phys_feats], dim=1) # [B, 132]
        
        # 3. 回归输出
        return self.reg_head(joint_feat)

## =================================================================
## 3. 支持多源路径寻址与深水数据合并的数据加载器
## =================================================================

class SARDataset(Dataset):
    def __init__(self, csv_dir, patch_dir, t_idx_list, supply_dir=None, is_train=False):
        self.patch_dir = patch_dir
        self.supply_dir = supply_dir
        self.is_train = is_train
        all_dfs = []
        
        # A. 加载原始基础测线数据
        for t_idx in t_idx_list:
            csv_name = f"SAR_Transect_{t_idx:02d}.csv"
            csv_path = os.path.join(csv_dir, csv_name)
            
            if os.path.exists(csv_path):
                df_single = pd.read_csv(csv_path)
                df_single['transect_idx'] = t_idx
                df_single['is_supply'] = False
                all_dfs.append(df_single)
                
        if len(all_dfs) == 0:
            raise ValueError(f"❌ 错误：在路径 {csv_dir} 下未找到任何指定的 CSV 文件！")
            
        base_df = pd.concat(all_dfs, ignore_index=True)
        
        # 数据基础清洗
        required_cols = ['s_idx', 'Peak_dist', 'Peak_Intensity', 'lambda', 'theta_deg', 'href']
        base_df = base_df.dropna(subset=required_cols).reset_index(drop=True)
        base_df['abs_href'] = base_df['href'].abs()
        
        # B. 追加补充数据逻辑
        if self.is_train and supply_dir and os.path.exists(supply_dir):
            supply_dfs = []
            print(f"\n📥 [数据整合] 检测到训练集补充数据源，正在扫描 {supply_dir} ...")
            
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
                
                print(f"    💡 成功载入 60-100m 深水补充样本共: {len(supply_df)} 个")
                self.df = pd.concat([base_df, supply_df], ignore_index=True)
            else:
                print("    ⚠️ 警告：在 supply_data 文件夹下未解析到符合命名规则的 CSV 文件。")
                self.df = base_df
        else:
            self.df = base_df
            
        # C. 保持原始的区域4和区域5深水线过采样逻辑不变
        if self.is_train and len(self.df) > 0:
            min_d = self.df['abs_href'].min()
            max_d = self.df['abs_href'].max()
            dynamic_bounds = np.linspace(min_d, max_d, 6)
            area4_start_bound = dynamic_bounds[3] # 62.05m线
            
            deep_water_mask = self.df['abs_href'] >= area4_start_bound
            deep_samples = self.df[deep_water_mask]
            if len(deep_samples) > 0:
                print(f"    💡 [数据增强] 混合域动态深水线: {area4_start_bound:.2f}m")
                print(f"    💡 全局深水（区域4与区域5）共 {len(deep_samples)} 个，进行标准双倍过采样...")
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
            raise FileNotFoundError(f"❌ 图像切片读取失败！检查路径：{img_path}")
        
        assert img.shape == (128, 128)
        
        img = img.astype(np.float32) / 255.0
        img_tensor = torch.tensor(img, dtype=torch.float32).unsqueeze(0)
        
        phys_feats = torch.tensor([row['Peak_dist'], row['Peak_Intensity'], row['lambda'], row['theta_deg']], dtype=torch.float32)
        target = torch.tensor([row['abs_href']], dtype=torch.float32)
        
        return img_tensor, phys_feats, target

## =================================================================
## 4. 主控制流与标准验证引擎
## =================================================================

def main():
    BATCH_SIZE = 64
    EPOCHS = 100            
    LEARNING_RATE = 0.001
    
    csv_dir_path = r'D:\SAR_database\python2'   
    patch_dir_path = r'D:\SAR_database\python'  
    supply_dir_path = r'D:\SAR_database\supply_data' 
    model_save_path = r'D:\SAR_database\python2\best_yolo_fixed_transect_model.pth' 
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"⚡ 计算设备锁定: 【{device}】")
    
    # 固定指定测线作为验证集
    val_transects = list(range(2, 50, 5))  
    total_transects = list(range(0, 50))
    train_transects = sorted([t for t in total_transects if t not in val_transects])
    
    print("\n================ 📐 测线固定对齐切分报告 ================")
    print(f"📡 固定指定的【验证集】测线 ID ({len(val_transects)}条): {val_transects}")
    print(f"🏋️ 自动生成的【训练集】基础测线 ID ({len(train_transects)}条): {train_transects}")
    print("========================================================")
    
    train_dataset = SARDataset(csv_dir=csv_dir_path, patch_dir=patch_dir_path, t_idx_list=train_transects, supply_dir=supply_dir_path, is_train=True)
    val_dataset = SARDataset(csv_dir=csv_dir_path, patch_dir=patch_dir_path, t_idx_list=val_transects, supply_dir=None, is_train=False)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    # 确立基于训练集的全局基准划分线
    train_min = train_dataset.df['abs_href'].min()
    train_max = train_dataset.df['abs_href'].max()
    train_bounds = np.linspace(train_min, train_max, 6)
    
    # 初始化 Late Fusion 模型
    model = YOLOLateFusionBaseNet().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    
    best_val_loss = float('inf')
    early_stop_patience = 20  
    patience_counter = 0
    
    print(f"\n🔥 消融组 M2 激活: 【末端特征拼接多模态 Late Fusion + 标准全局 MSE 控制】")
    print("\n================ 开 启 固 定 测 线 隔 离 训 练 (100 Epochs) ================")
    
    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0.0
        for imgs, phys, targets in train_loader:
            imgs, phys, targets = imgs.to(device), phys.to(device), targets.to(device)
            optimizer.zero_grad()
            
            # 同时传入图像和物理向量进行前向传播
            outputs = model(imgs, phys)
            
            # 标准全局 MSE 损失
            loss = nn.functional.mse_loss(outputs, targets)
            
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * imgs.size(0)
            
        model.eval()
        val_loss = 0.0
        val_mae = 0.0  
        val_relative_error_sum = 0.0 
        
        # 🌟 用于收集当前 Epoch 验证集全部真值与预测值，以便准确计算 R2
        epoch_val_trues = []
        epoch_val_preds = []
        
        with torch.no_grad(): 
            for imgs, phys, targets in val_loader:
                imgs, phys, targets = imgs.to(device), phys.to(device), targets.to(device)
                outputs = model(imgs, phys)
                
                base_loss = nn.functional.mse_loss(outputs, targets)
                val_loss += base_loss.item() * imgs.size(0)
                val_mae += torch.sum(torch.abs(outputs - targets)).item()
                
                batch_re = torch.abs(outputs - targets) / (targets + 1e-5)
                val_relative_error_sum += torch.sum(batch_re).item()
                
                # 缓存当前 batch 数据
                epoch_val_trues.extend(targets.cpu().numpy().flatten())
                epoch_val_preds.extend(outputs.cpu().numpy().flatten())
                
        epoch_train_loss = train_loss / len(train_dataset)
        epoch_val_loss = val_loss / len(val_dataset)
        epoch_val_mae = val_mae / len(val_dataset)
        
        epoch_val_rmse = np.sqrt(epoch_val_loss)
        epoch_val_re = (val_relative_error_sum / len(val_dataset)) * 100.0
        
        # 🌟 计算当前 Epoch 的全局验证集 R2
        epoch_val_trues = np.array(epoch_val_trues)
        epoch_val_preds = np.array(epoch_val_preds)
        ss_res = np.sum((epoch_val_trues - epoch_val_preds) ** 2)
        ss_tot = np.sum((epoch_val_trues - np.mean(epoch_val_trues)) ** 2)
        epoch_val_r2 = 1.0 - (ss_res / (ss_tot + 1e-5))
        
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        # 🌟 输出日志中增加了 Val R2
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
            print(f" -> 🎉 【消融组 M2】验证集损失刷新纪录，权重安全锁定。")
        else:
            patience_counter += 1
            
        if patience_counter >= early_stop_patience:
            print(f"\n🛑 [Early Stopping] 连续 {early_stop_patience} 轮未见改善，提前结束。")
            break
            
    print("\n================ 🏁 训练阶段结束，启动全部验证集测线多维水深段误差综合评估 ================")
    
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
        
        # 🌟 调整打印表格列宽，增加了最后一列 '平均决定系数(R2)'
        print("-" * 150)
        print(" 📊【M2 消融组：全部验证集测线不同水深段反演误差平均分析】")
        print("-" * 150)
        print(f"{'水深层级':<10}\t{'对应基准水深区间':<20}\t{'区域样本数':<12}\t{'平均绝对误差(MAE)':<20}\t{'均方根误差(RMSE)':<20}\t{'平均相对误差(RE)':<20}\t{'决定系数(R2)'}")
        print("-" * 150)
        
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
                
                # 🌟 计算该水深分段内样本的 R2
                reg_trues = all_true_depths[mask]
                reg_preds = all_pred_depths[mask]
                reg_ss_res = np.sum((reg_trues - reg_preds) ** 2)
                reg_ss_tot = np.sum((reg_trues - np.mean(reg_trues)) ** 2)
                # 规避分段内真值完全一样导致分母为 0 的极端情况
                regional_r2 = 1.0 - (reg_ss_res / (reg_ss_tot + 1e-5)) if reg_ss_tot > 0 else 0.0
                
                mae_str = f"{regional_mae:.2f} 米"
                rmse_str = f"{regional_rmse:.2f} 米"
                re_str = f"{regional_re:.2f}%"
                r2_str = f"{regional_r2:.4f}"
            else:
                mae_str = "0.00 米"
                rmse_str = "0.00 米"
                re_str = "0.00%"
                r2_str = "0.0000"
                
            print(f"区域 {i+1:<4}\t{low_b:5.1f}m ~ {high_b:5.1f}m\t\t{regional_count:<12}\t{mae_str:<20}\t{rmse_str:<20}\t{re_str:<20}\t{r2_str}")
            
        print("=" * 150)
    else:
        print("⚠️ 提示：未获取到有效的验证集全断面数据。")
        
    print("\n================ 🏁 实验全部结束 ================")

if __name__ == '__main__':
    main()