import pandas as pd
import numpy as np
from scipy.stats import ks_2samp
import warnings
import os

warnings.filterwarnings('ignore')

DATA_PATH = "data/gold_reviews.parquet"

def run_drift_detection(df, feature_name, confidence=0.05):

    df['text_length'] = df['text'].apply(lambda x: len(str(x)))
    
    mid_point = len(df) // 2
    df_old = df.iloc[:mid_point]
    df_new = df.iloc[mid_point:]
    

    ks_stat, p_value = ks_2samp(df_old[feature_name], df_new[feature_name])
    
    print(f"\n--- Drift Detection Results for '{feature_name}' (Text Length) ---")
    print(f"KS Statistic: {ks_stat:.4f}")
    print(f"P-Value: {p_value:.4f}")
    
    if p_value < confidence:
        print(f"ALERT: Data Drift Detected (p < {confidence}). The distribution of '{feature_name}' has changed significantly.")
    else:
        print(f"OK: No significant Data Drift detected (p >= {confidence}).")
        
    return {
        "drift_detected": p_value < confidence,
        "p_value": p_value
    }

if __name__ == "__main__":
    if not os.path.exists(DATA_PATH):
        print(f"ERROR: Data file not found at {DATA_PATH}. Check your DVC setup.")
    else:
        print("Loading full dataset for drift simulation...")
        full_df = pd.read_parquet(DATA_PATH)
        run_drift_detection(full_df, 'text_length')

# docker-compose exec backend python drift_detector.py