import os
import glob
import re
import numpy as np
import pandas as pd
import cv2
from scipy.fft import fft2, fftshift, ifft2
from sklearn.decomposition import PCA
from tqdm import tqdm

def estimate_wave_direction_acf(img, search_radius=15, threshold_ratio=0.55):
    """
    Estimate wave direction from 2D spatial autocorrelation (ACF) method (FFT-accelerated)
    """
    img_centered = img - np.mean(img)
    f_transform = fft2(img_centered)
    power_spectrum = f_transform * np.conj(f_transform)
    acf_raw = np.real(ifft2(power_spectrum))
    acf = fftshift(acf_raw)
    
    center_y, center_x = np.unravel_index(np.argmax(acf), acf.shape)
    max_val = acf[center_y, center_x]
    
    y_indices, x_indices = np.ogrid[:acf.shape[0], :acf.shape[1]]
    dist_from_center = np.sqrt((x_indices - center_x)**2 + (y_indices - center_y)**2)
    core_mask = (dist_from_center <= search_radius) & (acf >= max_val * threshold_ratio)
    
    pts_y, pts_x = np.where(core_mask)
    pts_y_rel = pts_y - center_y
    pts_x_rel = pts_x - center_x
    points = np.vstack((pts_x_rel, pts_y_rel)).T
    
    if len(points) < 5:
        return np.nan 
        
    pca = PCA(n_components=2)
    pca.fit(points)
    
    short_axis_vector = pca.components_[1] 
    wave_dir_rad = np.arctan2(short_axis_vector[1], short_axis_vector[0])
    wave_dir_deg = np.degrees(wave_dir_rad) % 180
    
    transformed_dir_deg = 180.0 - wave_dir_deg
    return transformed_dir_deg % 180


def directional_rolling_mean(angles_deg, window=25):
    """
    Scientific vector moving average for 180-degree periodic wave direction
    """
    rad_2X = np.radians(angles_deg * 2)
    sin_components = np.sin(rad_2X)
    cos_components = np.cos(rad_2X)
    
    smooth_sin = pd.Series(sin_components).rolling(window=window, min_periods=1, center=False).mean()
    smooth_cos = pd.Series(cos_components).rolling(window=window, min_periods=1, center=False).mean()
    
    recon_rad_2X = np.arctan2(smooth_sin, smooth_cos)
    recon_deg_2X = np.degrees(recon_rad_2X) % 360
    smooth_angles = recon_deg_2X / 2.0
    return smooth_angles


if __name__ == "__main__":
    print("=== Starting Multi-Transect ACF Calculation + Data Filtering + Scientific Vector Smoothing System ===")
    
    # 路径配置
    save_dir = r'D:\SAR_database\python'          # Directory for patch PNGs
    output_dir = r'D:\SAR_database\python2'        # Output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Universal matching: match any transect Txx and any sample point Sxxx
    search_pattern = os.path.join(save_dir, "SAR_T*_S*.png")
    all_img_paths = glob.glob(search_pattern)
    
    if not all_img_paths:
        print(f"❌ No SAR images matching the naming convention found in directory {save_dir}. Please check the path.")
        exit()
        
    print(f"Found {len(all_img_paths)} transect patches. Organizing transect mapping...")
    
    # Regex for precise filename parsing
    filename_regex = re.compile(r"SAR_T(\d{2})_S(\d{3})\.png")
    
    # Group image paths by transect ID
    transects_dict = {}
    for path in all_img_paths:
        filename = os.path.basename(path)
        match = filename_regex.match(filename)
        if match:
            t_idx = int(match.group(1))
            s_idx = int(match.group(2))
            if t_idx not in transects_dict:
                transects_dict[t_idx] = []
            transects_dict[t_idx].append((s_idx, path))
            
    print(f"Successfully identified {len(transects_dict)} independent transects with data.")
    
    summary_all_transects = []
    summary_deleted_data = []  # Collect all deleted data
    
    # Convert critical radians to angle filtering thresholds
    min_deg = np.degrees(0.3491)  # ~20.002°
    max_deg = np.degrees(1.3090)  # ~74.999°
    
    # Process each transect
    for t_idx in sorted(transects_dict.keys()):
        print(f"\n---> Processing transect [Transect {t_idx:02d}] ...")
        
        # Ensure points within the current transect are strictly ordered by s_idx
        pts_list = sorted(transects_dict[t_idx], key=lambda x: x[0])
        transect_dataset = []
        
        for s_idx, img_path in tqdm(pts_list, desc=f"T{t_idx:02d} ACF feature extraction"):
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                # Image read failure also captured as deleted data
                summary_deleted_data.append({
                    "transect_id": t_idx,
                    "s_idx": s_idx,
                    "Theta_ACF_Raw": np.nan,
                    "deleted_reason": "Image read failed"
                })
                continue
                
            try:
                wave_direction = estimate_wave_direction_acf(img, search_radius=15, threshold_ratio=0.55)
            except Exception:
                wave_direction = np.nan
                
            transect_dataset.append({
                "s_idx": s_idx,
                "Theta_ACF_Raw": wave_direction
            })
            
        if transect_dataset:
            df_transect = pd.DataFrame(transect_dataset)
            
            # ====== Data Cleaning and Collection Steps ======
            
            # 1. Extract and collect the last 7 data points of each transect
            if len(df_transect) > 7:
                df_last_7 = df_transect.iloc[-7:].copy()
                df_last_7.insert(0, 'transect_id', t_idx)
                df_last_7['deleted_reason'] = 'Last 7 rows of transect'
                summary_deleted_data.append(df_last_7)
                
                # Keep the preceding data
                df_transect = df_transect.iloc[:-7].reset_index(drop=True)
            else:
                # If not enough data (less than 7), mark all as deleted and skip
                df_transect.insert(0, 'transect_id', t_idx)
                df_transect['deleted_reason'] = 'Transect length <= 7'
                summary_deleted_data.append(df_transect)
                print(f"⚠️ Transect [Transect {t_idx:02d}] has fewer than 7 data points, all removed and skipped.")
                continue
            
            # 2. Extract and collect invalid angle range data (outside the specified range, or NaN)
            # Valid mask
            valid_mask = (df_transect["Theta_ACF_Raw"] >= min_deg) & (df_transect["Theta_ACF_Raw"] <= max_deg)
            
            # Extract invalid data
            df_invalid = df_transect[~valid_mask].copy()
            if not df_invalid.empty:
                df_invalid.insert(0, 'transect_id', t_idx)
                # Distinguish between computation errors (NaN) and out-of-range
                df_invalid['deleted_reason'] = np.where(
                    df_invalid['Theta_ACF_Raw'].isna(), 
                    'ACF calculation NaN/Error', 
                    f'Angle out of range ({min_deg:.2f} to {max_deg:.2f} deg)'
                )
                summary_deleted_data.append(df_invalid)
            
            # Filter DataFrame to keep only valid rows
            df_transect = df_transect[valid_mask].reset_index(drop=True)
            
            # If no remaining data after filtering, skip
            if df_transect.empty:
                print(f"⚠️ Transect [Transect {t_idx:02d}] has no valid rows after filtering.")
                continue
            
            # ==================================
            
            # Apply 25-window vector smoothing to the cleaned continuous angles
            df_transect["Theta_ACF_Smoothed"] = directional_rolling_mean(df_transect["Theta_ACF_Raw"], window=25)
            
            df_transect["Theta_ACF_Raw"] = df_transect["Theta_ACF_Raw"].round(4)
            df_transect["Theta_ACF_Smoothed"] = df_transect["Theta_ACF_Smoothed"].round(4)
            
            # 1. Export individual transect CSV
            transect_csv_path = os.path.join(output_dir, f'SAR_theta_ACF_Transect_{t_idx:02d}.csv')
            df_transect.to_csv(transect_csv_path, index=False)
            
            # 2. Copy for master table, inject transect ID
            df_copy = df_transect.copy()
            df_copy.insert(0, 'transect_id', t_idx)
            summary_all_transects.append(df_copy)
            
    # 3. Vertically merge all valid transects, export global master table
    if summary_all_transects:
        master_df = pd.concat(summary_all_transects, ignore_index=True)
        master_csv_path = os.path.join(output_dir, 'SAR_theta_ACF_All_Master.csv')
        master_df.to_csv(master_csv_path, index=False)
        print(f"\n📁 Master table for all transects and sample points saved to: {master_csv_path} (total rows: {len(master_df)})")

    # 4.Merge all the deleted data vertically and export the summary table of the independent cleaning log.
    if summary_deleted_data:
        deleted_df = pd.concat([pd.DataFrame(x) if isinstance(x, list) else x for x in summary_deleted_data], ignore_index=True)
        cols = ['transect_id', 's_idx', 'Theta_ACF_Raw', 'deleted_reason']
        deleted_df = deleted_df[cols]
        if 'Theta_ACF_Raw' in deleted_df.columns:
            deleted_df['Theta_ACF_Raw'] = deleted_df['Theta_ACF_Raw'].round(4)
            
        deleted_csv_path = os.path.join(output_dir, 'SAR_theta_ACF_Deleted_Rows.csv')
        deleted_df.to_csv(deleted_csv_path, index=False)
        print(f" All the abnormal sample data that were removed have been synchronized to: {deleted_csv_path} (Total number of lines: {len(deleted_df)})")
        
    print("\n====================================================")
    print(f"🎉 🎉 🎉 complete！")