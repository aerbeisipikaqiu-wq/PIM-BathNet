import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
import cv2  
import random 
import numpy as np
import pandas as pd
import scipy.io as sio
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import r2_score

## ==========================================
## 1. Define the multimodal components and the core Cross-Attention module
## ==========================================
class MultiModalPatchEmbedding(nn.Module):
    def __init__(self, img_size=128, patch_size=8, in_channels=1, embed_dim=64):
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2  
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)        
        x = x.flatten(2)        
        x = x.transpose(1, 2)   
        return x


class CrossAttentionFusion(nn.Module):
    """ Physical features (Q) and image features (K, V) cross-attention fusion module """
    def __init__(self, embed_dim=64, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, img_tokens, phys_tokens):
        B, N_img, C = img_tokens.shape
        _, N_phys, _ = phys_tokens.shape

        q = self.q_proj(phys_tokens).reshape(B, N_phys, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(img_tokens).reshape(B, N_img, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(img_tokens).reshape(B, N_img, self.num_heads, self.head_dim).transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(B, N_phys, C)
        output = self.norm(phys_tokens + self.out_proj(x))
        return output


class PhysicsGatedCrossAttnTransformerNet(nn.Module):
    """ Introduce the physical degradation two-channel gated network (PINN kernel) """
    def __init__(self, img_size=128, patch_size=8, embed_dim=64, depth=4, num_heads=4):
        super().__init__()
        
        # Channel 1: Multimodal Image/Physical Cross-Attention Backbone
        self.patch_embed = MultiModalPatchEmbedding(img_size, patch_size, in_channels=1, embed_dim=embed_dim)
        num_img_patches = self.patch_embed.num_patches
        self.img_pos_embed = nn.Parameter(torch.zeros(1, num_img_patches, embed_dim))
        
        self.phys_to_token = nn.Sequential(
            nn.Linear(1, 32), nn.GELU(), nn.Linear(32, embed_dim)
        )
        self.phys_pos_embed = nn.Parameter(torch.zeros(1, 4, embed_dim))
        self.cross_attention_fusion = CrossAttentionFusion(embed_dim=embed_dim, num_heads=num_heads)
        
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.total_pos_embed = nn.Parameter(torch.zeros(1, 1 + 4, embed_dim)) 
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=embed_dim * 4, 
            dropout=0.15, activation='gelu', batch_first=True, norm_first=True     
        )
        self.transformer_blocks = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        
        # 🌟 Channel 2: Pure physics fallback nonlinear backbone (avoids contact with low-SNR images to prevent contamination)
        self.pure_phys_backbone = nn.Sequential(
            nn.Linear(4, 32),
            nn.GELU(),
            nn.Linear(32, embed_dim)
        )
        
        # 🌟 Dynamic physics gating network: autonomously learns to assign trust to images based on current point's physical features
        self.gate_network = nn.Sequential(
            nn.Linear(4, 16),
            nn.GELU(),
            nn.Linear(16, 1),
            nn.Sigmoid() # Outputs smooth gating weight in [0.0, 1.0]
        )
        
        # Final mixer regression head
        self.regressor = nn.Sequential(
            nn.Linear(embed_dim, 32),
            nn.GELU(),
            nn.Linear(32, 1)    
        )
        
        nn.init.trunc_normal_(self.img_pos_embed, std=0.02)
        nn.init.trunc_normal_(self.phys_pos_embed, std=0.02)
        nn.init.trunc_normal_(self.total_pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, img, phys_feats):
        batch_size = img.shape[0]
        
        # --- [Dual Channel 1]: Multimodal Cross Feature Stream ---
        visual_tokens = self.patch_embed(img) + self.img_pos_embed
        phys_feats_expanded = phys_feats.unsqueeze(-1) 
        phys_tokens = self.phys_to_token(phys_feats_expanded) + self.phys_pos_embed
        
        fused_phys_tokens = self.cross_attention_fusion(img_tokens=visual_tokens, phys_tokens=phys_tokens) 
        cls_tokens = self.cls_token.expand(batch_size, -1, -1) 
        token_sequence = torch.cat((cls_tokens, fused_phys_tokens), dim=1) + self.total_pos_embed
        
        transformed_sequence = self.transformer_blocks(token_sequence)
        mixed_feat = transformed_sequence[:, 0, :] 
        
        # --- [Dual Channel 2]: Pure Physics Fallback Feature Stream ---
        pure_phys_feat = self.pure_phys_backbone(phys_feats)
        
        # --- [Physics Gating Decision] ---
        alpha = self.gate_network(phys_feats) 
        
        # Dynamic feature fuse-and-break fusion
        final_feat = (1.0 - alpha) * mixed_feat + alpha * pure_phys_feat
        
        output = self.regressor(final_feat)
        return output

## ==========================================
## 2. Transect-isolated Universal Data Loader
## ==========================================
class SARDataset(Dataset):
    def __init__(self, csv_dir, patch_dir, t_idx_list, global_means, global_stds):
        self.patch_dir = patch_dir
        self.phys_means = global_means
        self.phys_stds = global_stds
        
        df_list = []
        feat_cols = ['Peak_dist', 'Peak_Intensity', 'lambda', 'theta_deg']
        
        for t_idx in t_idx_list:
            csv_name = f"SAR_Transect_{t_idx:02d}.csv"
            csv_path = os.path.join(csv_dir, csv_name)
            if os.path.exists(csv_path):
                df_part = pd.read_csv(csv_path).dropna(subset=['s_idx', 'href'] + feat_cols)
                df_part['transect_idx'] = t_idx
                df_list.append(df_part)
                
        if len(df_list) == 0:
            raise ValueError(f"❌ Initialization failed: no valid transect CSV data found!")
            
        self.df = pd.concat(df_list, axis=0).reset_index(drop=True)
        
    def __len__(self):
        return len(self.df)
        
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        t_idx, s_idx = int(row['transect_idx']), int(row['s_idx'])
        
        img_path = os.path.join(self.patch_dir, f"SAR_T{t_idx:02d}_S{s_idx:03d}.png")
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"❌ Patch image not found: {img_path}")
        
        img = img.astype(np.float32) / 255.0
        img_tensor = torch.tensor(img, dtype=torch.float32).unsqueeze(0)
        
        raw_phys = np.array([row['Peak_dist'], row['Peak_Intensity'], row['lambda'], row['theta_deg']], dtype=np.float32)
        norm_phys = (raw_phys - self.phys_means) / self.phys_stds
        phys_feats = torch.tensor(norm_phys, dtype=torch.float32)
        
        target = torch.tensor([abs(row['href'])], dtype=torch.float32)
        return img_tensor, phys_feats, target, t_idx

## ==========================================
## 3. Standard Main Training Control Flow
## ==========================================
def main():
    print("【System】Scheme A: Physics-gated dual-channel multimodal network initializing...")
    
    BATCH_SIZE = 128      
    EPOCHS = 50                
    LEARNING_RATE = 2e-4  
    WEIGHT_DECAY = 3e-3   
    WARMUP_EPOCHS = 5     
    PATIENCE = 15               
    
    csv_dir_path = r'D:\SAR_database\python2'
    patch_dir_path = r'D:\SAR_database\python'
    model_save_path = r'D:\SAR_database\python2\best_transformer_model2.pth' 
    
    all_transects = list(range(50))  
    val_transects = list(range(2, 50, 5))  # [2, 7, 12, 17, 22, 27, 32, 37, 42, 47]
    train_transects = [t for t in all_transects if t not in val_transects]
    
    print(f"🎯 Fixed isolated validation transects (10 total): {val_transects}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    global_df_list = []
    feat_cols = ['Peak_dist', 'Peak_Intensity', 'lambda', 'theta_deg']
    for t in all_transects:
        c_path = os.path.join(csv_dir_path, f"SAR_Transect_{t:02d}.csv")
        if os.path.exists(c_path):
            global_df_list.append(pd.read_csv(c_path)[feat_cols])
    combined_global = pd.concat(global_df_list, axis=0).dropna()
    global_means = combined_global.mean().values.astype(np.float32)
    global_stds = combined_global.std().values.astype(np.float32)
    global_stds[global_stds == 0] = 1.0
    
    train_dataset = SARDataset(csv_dir=csv_dir_path, patch_dir=patch_dir_path, t_idx_list=train_transects, global_means=global_means, global_stds=global_stds)
    val_dataset = SARDataset(csv_dir=csv_dir_path, patch_dir=patch_dir_path, t_idx_list=val_transects, global_means=global_means, global_stds=global_stds)
    train_size, val_size = len(train_dataset), len(val_dataset)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=False, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=False, num_workers=0)
    
    model = PhysicsGatedCrossAttnTransformerNet(img_size=128, patch_size=8, embed_dim=64, depth=4, num_heads=4).to(device)
    criterion = nn.MSELoss()  
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS - WARMUP_EPOCHS, eta_min=1e-6)
    
    best_val_loss = float('inf')
    best_epoch = 0
    patience_counter = 0        
    best_predictions_cache = [] 

    print("\n================ Starting Formal Training (Physics Anti-Contamination Fuse Mode) ================")
    for epoch in range(1, EPOCHS + 1):
        if epoch <= WARMUP_EPOCHS:
            lr_factor = float(epoch) / float(max(1, WARMUP_EPOCHS))
            current_lr = LEARNING_RATE * lr_factor
            for param_group in optimizer.param_groups:
                param_group['lr'] = current_lr
        else:
            cosine_scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']
            
        model.train()
        train_loss = 0.0
        for imgs, phys, targets, _ in train_loader:
            imgs, phys, targets = imgs.to(device), phys.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(imgs, phys)
            loss = criterion(outputs, targets)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item() * imgs.size(0)
            
        model.eval()
        val_loss = 0.0
        val_absolute_errors = []
        val_relative_errors = []
        
        # 🌟 Cache full targets and predictions for the current epoch to compute R2 and comprehensive metrics
        all_epoch_true = []
        all_epoch_pred = []
        current_epoch_details = []
        
        with torch.no_grad(): 
            for imgs, phys, targets, t_ids in val_loader:
                imgs, phys, targets = imgs.to(device), phys.to(device), targets.to(device)
                outputs = model(imgs, phys)
                loss = criterion(outputs, targets)
                
                val_loss += loss.item() * imgs.size(0)
                
                out_arr = outputs.cpu().numpy().flatten()
                tgt_arr = targets.cpu().numpy().flatten()
                tid_arr = t_ids.numpy().flatten()
                
                for o, t, tid in zip(out_arr, tgt_arr, tid_arr):
                    abs_err = abs(o - t)
                    val_absolute_errors.append(abs_err)
                    rel_err = (abs_err / t) if t != 0 else 0.0
                    val_relative_errors.append(rel_err)
                    
                    all_epoch_true.append(t)
                    all_epoch_pred.append(o)
                    current_epoch_details.append({'t_idx': tid, 'true_depth': t, 'pred_depth': o})
                
        epoch_train_loss = train_loss / train_size
        epoch_val_loss = val_loss / val_size
        
        # 🌟 Compute four extended metrics for the current epoch
        epoch_val_rmse = np.sqrt(epoch_val_loss)
        epoch_val_mae = np.mean(val_absolute_errors)
        epoch_val_re_pct = np.mean(val_relative_errors) * 100.0
        epoch_val_r2 = r2_score(all_epoch_true, all_epoch_pred)
        
        print(f"Epoch [{epoch:02d}/{EPOCHS:02d}] | "
              f"Train MSE: {epoch_train_loss:.4f} | "
              f"Val RMSE: {epoch_val_rmse:.4f} m | "
              f"Val MAE: {epoch_val_mae:.2f} m | "
              f"Val RE: {epoch_val_re_pct:.2f}% | "
              f"Val R2: {epoch_val_r2:.4f} | "
              f"LR: {current_lr:.6f}")
        
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            best_epoch = epoch
            torch.save(model.state_dict(), model_save_path)
            best_predictions_cache = current_epoch_details
            patience_counter = 0  
            print(f" -> 🎉 Spatial isolation set generalization loss reached new low, physics gating weights updated.")
        else:
            patience_counter += 1 
            
        if patience_counter >= PATIENCE:
            print(f"\n🛑 [Early Stopping]: Validation set has not improved for {PATIENCE} consecutive epochs. Best model locked at epoch {best_epoch}.")
            break
            
    print("\n================ Training Complete ================")
    print(f"Scheme A best validation set overall MSE loss: {best_val_loss:.4f} (achieved at epoch {best_epoch})")
    
    # ==========================================
    # 4. 🌟 Refactored: Output error analysis report for all validation transects combined (four core metrics)
    # ==========================================
    if len(best_predictions_cache) == 0:
        print("⚠️ Could not extract valid validation point cache.")
        return

    all_true_depths = [r['true_depth'] for r in best_predictions_cache]
    min_d = min(all_true_depths)
    max_d = max(all_true_depths)
    bin_width = (max_d - min_d) / 5.0
    
    print("\n" + "-" * 105)
    print(" 📊【🌍 All 10 validation transects combined: Multi-metric error analysis by depth bin across the full section】")
    print("-" * 105)
    print(f"{'Depth Bin':<10}{'Global True Depth Range':<25}{'Valid Samples':<14}{'RMSE (m)':<14}{'MAE (m)':<14}{'RE (%)':<14}{'R2 Score'}")
    print("-" * 105)
    
    for i in range(5):
        bin_start = min_d + i * bin_width
        bin_end = min_d + (i + 1) * bin_width
        if i == 4:
            bin_data = [r for r in best_predictions_cache if bin_start <= r['true_depth'] <= bin_end]
        else:
            bin_data = [r for r in best_predictions_cache if bin_start <= r['true_depth'] < bin_end]
            
        sample_count = len(bin_data)
        if sample_count > 0:
            bin_trues = [r['true_depth'] for r in bin_data]
            bin_preds = [r['pred_depth'] for r in bin_data]
            
            # Compute per-bin metrics
            b_mse = np.mean((np.array(bin_trues) - np.array(bin_preds)) ** 2)
            b_rmse = np.sqrt(b_mse)
            b_mae = np.mean([abs(t - p) for t, p in zip(bin_trues, bin_preds)])
            b_re = np.mean([(abs(t - p) / t if t != 0 else 0.0) for t, p in zip(bin_trues, bin_preds)]) * 100.0
            
            # Guard: R2 is meaningless if too few samples or constant target values
            if len(bin_data) > 1 and np.var(bin_trues) > 1e-6:
                b_r2 = r2_score(bin_trues, bin_preds)
                r2_str = f"{b_r2:.4f}"
            else:
                r2_str = "N/A"
                
            rmse_str = f"{b_rmse:.2f}"
            mae_str = f"{b_mae:.2f}"
            re_str = f"{b_re:.2f}%"
        else:
            sample_count = 0
            rmse_str = mae_str = re_str = r2_str = "No samples"
            
        print(f"Bin {i+1:<6}{bin_start:6.1f}m ~ {bin_end:<6.1f}m{sample_count:^20}{rmse_str:<14}{mae_str:<14}{re_str:<14}{r2_str}")
    print("=========================================================================================================\n")

if __name__ == '__main__':
    main()