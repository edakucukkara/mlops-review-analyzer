from fastapi import FastAPI, HTTPException
import pandas as pd
import torch
from transformers import pipeline
from pydantic import BaseModel
from typing import List, Dict
from functools import lru_cache
import re

# --- CONFIGURATION ---
DATA_PATH = "data/gold_reviews.parquet"
CLASSIFIER_MODEL = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"
MAX_REVIEWS_TO_ANALYZE = 50 

LABELS = [
    "Quality & Effectiveness", "Scent & Texture", "Price & Value", 
    "Packaging & Shipping", "Safety & Authenticity", "Service"
]

app = FastAPI(title="Product Review Analyzer")

# --- LOAD DATA ---
print("Loading Data...")
df = pd.read_parquet(DATA_PATH)

# FILTER MENU (Min 5 Reviews)
counts = df['parent_asin'].value_counts()
valid_asins = counts[counts >= 5].index
df_valid = df[df['parent_asin'].isin(valid_asins)]
top_products = df_valid.groupby('parent_asin').first().sort_values('rating_number', ascending=False).head(100)
product_menu = top_products[['product_title', 'product_image']].to_dict(orient='index')
print(f"Data Loaded. Menu contains {len(product_menu)} valid products.")

# --- LOAD CLASSIFIER ONLY ---
device = 0 if torch.cuda.is_available() else -1
print("Loading Classifier...")
classifier = pipeline("zero-shot-classification", model=CLASSIFIER_MODEL, device=device)
print("Classifier Ready")

# --- DATA MODELS ---
class ReviewDetail(BaseModel):
    text: str
    rating: int
    topic: str
    topic_score: float

class ReviewAnalysis(BaseModel):
    total_reviews_local: int
    analyzed_count: int
    product_title: str
    product_image: str | None
    average_rating: float 
    rating_number: int    
    top_topics: Dict[str, float]
    sentiment_breakdown: Dict[str, int]
    reviews: List[ReviewDetail]
    ai_summary: str 

# --- HELPERS ---
def clean_text(text):
    text = re.sub(r'<br\s*/?>', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def generate_smart_summary(topic_counts, sentiment_counts, total_analyzed):
    """
    Constructs a professional summary based on the HARD DATA from DeBERTa.
    This guarantees no hallucinations or slang.
    """
    # 1. Identify Top Topic
    top_topic = max(topic_counts, key=topic_counts.get)
    top_topic_pct = (topic_counts[top_topic] / total_analyzed) * 100
    
    # 2. Identify Top Sentiment
    top_sentiment = max(sentiment_counts, key=sentiment_counts.get)
    
    # 3. Construct Sentence
    summary = f"Based on an analysis of {total_analyzed} reviews, the customer sentiment is predominantly **{top_sentiment}**. "
    summary += f"The primary driver of conversation is **{top_topic}**, which appears in **{top_topic_pct:.0f}%** of the feedback. "
    
    # 4. Add Context
    if top_sentiment == "Negative":
        summary += "This suggests users are facing critical issues in this specific area."
    elif top_sentiment == "Positive":
        summary += "This indicates high user satisfaction regarding this feature."
    else:
        summary += "Opinions on this product appear to be mixed."
        
    return summary

@lru_cache(maxsize=50) 
def cached_inference(asin: str):
    product_reviews = df[df['parent_asin'] == asin].copy()
    if product_reviews.empty: return None

    meta = product_reviews.iloc[0]
    
    # Sort by helpfulness
    target_reviews = product_reviews.sort_values(by=['helpful_vote', 'timestamp'], ascending=[False, False]).head(MAX_REVIEWS_TO_ANALYZE)
    
    # Inference
    processed_reviews = []
    topic_counts = {}
    sentiment_counts = {"Positive": 0, "Negative": 0, "Neutral": 0}

    raw_texts = [clean_text(t) for t in target_reviews['text'].tolist()]
    results = classifier(raw_texts, candidate_labels=LABELS, multi_label=True)

    for i, row in enumerate(target_reviews.itertuples()):
        res = results[i]
        best_topic, best_score = max(zip(res['labels'], res['scores']), key=lambda x: x[1])
        
        processed_reviews.append(ReviewDetail(
            text=raw_texts[i],
            rating=row.rating,
            topic=best_topic,
            topic_score=best_score
        ))

        topic_counts[best_topic] = topic_counts.get(best_topic, 0) + 1
        
        if row.rating >= 4: sentiment_counts["Positive"] += 1
        elif row.rating <= 2: sentiment_counts["Negative"] += 1
        else: sentiment_counts["Neutral"] += 1

    # Generate Professional Summary (Logic-Based)
    total = len(processed_reviews)
    ai_generated_text = generate_smart_summary(topic_counts, sentiment_counts, total)

    topic_percentages = {k: v / total for k, v in topic_counts.items()}

    return {
        "total_reviews_local": len(product_reviews),
        "analyzed_count": total,
        "product_title": meta['product_title'],
        "product_image": meta['product_image'],
        "average_rating": float(meta['average_rating']) if meta['average_rating'] else 0.0,
        "rating_number": int(meta['rating_number']) if meta['rating_number'] else 0,
        "top_topics": topic_percentages,
        "sentiment_breakdown": sentiment_counts,
        "reviews": processed_reviews,
        "ai_summary": ai_generated_text
    }

# --- ENDPOINTS ---
@app.get("/products")
def get_products():
    return [{"asin": k, "title": v['product_title']} for k, v in product_menu.items()]

@app.get("/analyze/{asin}", response_model=ReviewAnalysis)
def analyze_product(asin: str):
    result = cached_inference(asin)
    if not result:
        raise HTTPException(status_code=404, detail="Product not found")
    return result