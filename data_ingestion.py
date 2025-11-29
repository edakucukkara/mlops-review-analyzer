import pandas as pd
from datasets import load_dataset
import os

# CONFIGURATION
# Direct links to the full JSONL files
REVIEW_URL = "https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/resolve/main/raw/review_categories/All_Beauty.jsonl"
META_URL = "https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/resolve/main/raw/meta_categories/meta_All_Beauty.jsonl"

OUTPUT_DIR = "data"
OUTPUT_FILE = "gold_reviews.parquet"

def extract_image_url(image_data):
    """Helper to get a clean URL from the complex image list."""
    try:
        if isinstance(image_data, list) and len(image_data) > 0:
            first_image = image_data[0]
            # Priority: Large > Hi_Res > Thumb
            if 'large' in first_image and first_image['large']: return first_image['large']
            if 'hi_res' in first_image and first_image['hi_res']: return first_image['hi_res']
            if 'thumb' in first_image and first_image['thumb']: return first_image['thumb']
    except:
        return None
    return None

def ingest_and_merge():
    print("Downloading FULL Reviews Dataset...")
    ds_reviews = load_dataset("json", data_files=REVIEW_URL, split="train")
    
    print("Downloading Metadata (Using Pandas to bypass schema error)...")
    df_meta = pd.read_json(META_URL, lines=True)

    print("Converting Reviews to Table...")
    df_reviews = ds_reviews.to_pandas()

    print(f"   -> Raw Reviews Count: {len(df_reviews)}")
    print(f"   -> Raw Products Count: {len(df_meta)}")

    # --- CLEANING REVIEWS ---
    print("Cleaning Reviews...")
    cols_reviews = ['rating', 'text', 'title', 'parent_asin', 'asin', 'timestamp', 'helpful_vote']
    df_reviews = df_reviews[cols_reviews]
    df_reviews.rename(columns={'title': 'review_title'}, inplace=True)

    # --- CLEANING METADATA ---
    print("Cleaning Metadata...")
    df_meta['product_image'] = df_meta['images'].apply(extract_image_url)
    
    cols_meta = ['parent_asin', 'title', 'main_category', 'categories', 'average_rating', 'rating_number', 'store', 'product_image']
    df_meta = df_meta[cols_meta]
    df_meta.rename(columns={'title': 'product_title'}, inplace=True)
    
    # Drop duplicates (One row per product in metadata)
    df_meta.drop_duplicates(subset=['parent_asin'], inplace=True)

    # --- MERGING ---
    print("Merging Reviews with Product Info...")
    df_final = pd.merge(df_reviews, df_meta, on='parent_asin', how='inner')

    # Remove rows where critical text is missing
    df_final.dropna(subset=['text', 'product_title'], inplace=True)

    # --- STORAGE ---
    print("Saving to Parquet...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    save_path = os.path.join(OUTPUT_DIR, OUTPUT_FILE)
    
    df_final.to_parquet(save_path, index=False)
    
    print("-" * 30)
    print(f"PIPELINE SUCCESS!")
    print(f"Data Saved to: {save_path}")
    print(f"Final Dataset Size: {len(df_final)} rows")
    print(f"Columns: {df_final.columns.tolist()}")

if __name__ == "__main__":
    ingest_and_merge()