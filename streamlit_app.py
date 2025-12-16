import streamlit as st
import pandas as pd
import plotly.express as px
import datetime
import sys
import torch
from transformers import pipeline
import re
import time
import os

# --- CONFIGURATION ---
# Note: Data is tracked with Git LFS for Streamlit Cloud deployment
# For local development with DVC, run: dvc pull
DATA_PATH = "data/gold_reviews.parquet"
CLASSIFIER_MODEL = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"
MAX_REVIEWS_TO_ANALYZE = 50

LABELS = [
    "Quality & Effectiveness", "Scent & Texture", "Price & Value",
    "Packaging & Shipping", "Safety & Authenticity", "Service"
]

st.set_page_config(page_title="Product Review Analyzer", page_icon="🛍️", layout="wide")

def log_feedback(asin, feedback_type):
    """Simple HITL logging function for user feedback tracking"""
    timestamp = datetime.datetime.now().isoformat()
    print(f"HITL_LOG: {timestamp}, ASIN: {asin}, Feedback: {feedback_type}")
    sys.stdout.flush()

# --- LOAD DATA ---
@st.cache_data
def load_data():
    """Load and prepare product data"""
    print("Loading Data...")
    df = pd.read_parquet(DATA_PATH)

    # FILTER MENU (Min 5 Reviews)
    counts = df['parent_asin'].value_counts()
    valid_asins = counts[counts >= 5].index
    df_valid = df[df['parent_asin'].isin(valid_asins)]
    top_products = df_valid.groupby('parent_asin').first().sort_values('rating_number', ascending=False).head(100)
    product_menu = top_products[['product_title', 'product_image']].to_dict(orient='index')
    print(f"Data Loaded. Menu contains {len(product_menu)} valid products.")

    return df, product_menu

# --- LOAD CLASSIFIER ---
@st.cache_resource
def load_classifier():
    """Load and warm up the classifier model"""
    device = 0 if torch.cuda.is_available() else -1
    print("Loading Classifier...")
    classifier = pipeline("zero-shot-classification", model=CLASSIFIER_MODEL, device=device)
    print("Classifier Loaded. Warming up model...")

    # Warm-up: Run dummy inference to load model weights into memory
    with torch.inference_mode():
        _ = classifier(["warm up test"], candidate_labels=LABELS[:2])
    print("Model Ready and Warmed Up")

    return classifier

# --- HELPER FUNCTIONS ---
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

@st.cache_data
def analyze_product(asin: str, _classifier, _df):
    """
    Analyze product reviews using the classifier.
    Note: _classifier and _df are prefixed with _ to prevent Streamlit from hashing them
    """
    start_time = time.time()

    product_reviews = _df[_df['parent_asin'] == asin].copy()
    if product_reviews.empty:
        return None

    meta = product_reviews.iloc[0]

    # Sort by helpfulness
    target_reviews = product_reviews.sort_values(
        by=['helpful_vote', 'timestamp'],
        ascending=[False, False]
    ).head(MAX_REVIEWS_TO_ANALYZE)

    # Inference
    processed_reviews = []
    topic_counts = {}
    sentiment_counts = {"Positive": 0, "Negative": 0, "Neutral": 0}

    raw_texts = [clean_text(t) for t in target_reviews['text'].tolist()]

    # Use torch.inference_mode() for faster inference and batch processing
    with torch.inference_mode():
        results = _classifier(raw_texts, candidate_labels=LABELS, multi_label=True, batch_size=16)

    for i, row in enumerate(target_reviews.itertuples()):
        res = results[i]
        best_topic, best_score = max(zip(res['labels'], res['scores']), key=lambda x: x[1])

        processed_reviews.append({
            "text": raw_texts[i],
            "rating": row.rating,
            "topic": best_topic,
            "topic_score": best_score
        })

        topic_counts[best_topic] = topic_counts.get(best_topic, 0) + 1

        if row.rating >= 4:
            sentiment_counts["Positive"] += 1
        elif row.rating <= 2:
            sentiment_counts["Negative"] += 1
        else:
            sentiment_counts["Neutral"] += 1

    # Generate Professional Summary (Logic-Based)
    total = len(processed_reviews)
    ai_generated_text = generate_smart_summary(topic_counts, sentiment_counts, total)

    topic_percentages = {k: v / total for k, v in topic_counts.items()}

    end_time = time.time()
    latency_ms = (end_time - start_time) * 1000
    print(f"INFO: [MONITORING] Analysis Latency for {asin}: {latency_ms:.2f} ms")

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

# --- LOAD RESOURCES ---
df, product_menu = load_data()
classifier = load_classifier()

# --- SESSION STATE ---
if 'analysis_data' not in st.session_state:
    st.session_state.analysis_data = None
if 'selected_asin' not in st.session_state:
    st.session_state.selected_asin = None

# --- SIDEBAR ---
with st.sidebar:
    st.header("🔎 Select Product")

    # Create a dictionary for the dropdown
    menu_options = {p['product_title'][:60] + "...": asin for asin, p in product_menu.items()}

    # Add a "None" option to show the landing page
    options_list = ["-- Select a Product --"] + list(menu_options.keys())
    selected_name = st.selectbox("Popular Products:", options_list)

    if selected_name != "-- Select a Product --":
        final_asin = menu_options[selected_name]
    else:
        final_asin = None

    st.divider()
    custom_asin = st.text_input("Or enter ASIN ID:", placeholder="e.g. B00YQ6X8EO")
    if custom_asin:
        final_asin = custom_asin

    if final_asin:
        if st.button("🚀 Analyze Reviews", type="primary", use_container_width=True):
            st.session_state.selected_asin = final_asin
            # Reset data to trigger reload
            st.session_state.analysis_data = None
            st.rerun()

# --- MAIN PAGE LOGIC ---

if not st.session_state.selected_asin:
    # 🏠 LANDING PAGE
    st.title("🛍️ Product Review Analyzer")
    st.markdown("### Powered by DeBERTa Zero-Shot Classification")

    col1, col2 = st.columns(2)
    with col1:
        st.info("""
        **What is this?**
        This is an MLOps-engineered system that analyzes customer feedback in real-time.
        Instead of reading thousands of reviews, it uses AI to aggregate topics and sentiment instantly.
        """)
        st.markdown("""
        **Features:**
        - 🧠 **Zero-Shot Learning:** Detects topics (Quality, Price, Shipping) without specific training.
        - ⚡ **Caching Strategy:** Delivers instant results for popular items.
        - 📊 **Interactive Dashboard:** Filter and explore review data.
        """)
    with col2:
        # Example chart
        st.markdown("#### Real-Time Insights")
        dummy_df = pd.DataFrame({"Topic": ["Quality", "Price", "Shipping"], "Value": [60, 30, 10]})
        fig = px.pie(dummy_df, values='Value', names='Topic', hole=0.6, color_discrete_sequence=px.colors.sequential.Teal)
        st.plotly_chart(fig, use_container_width=True)

else:
    # 📊 ANALYSIS PAGE
    # Fetch Data if needed
    if st.session_state.analysis_data is None:
        with st.spinner("🧠 AI is processing reviews..."):
            data = analyze_product(st.session_state.selected_asin, classifier, df)
            if data:
                st.session_state.analysis_data = data
            else:
                st.error("Product not found or not enough reviews.")

    data = st.session_state.analysis_data

    if data:
        # 1. HEADER SECTION
        st.subheader(data['product_title'])
        col_img, col_stats = st.columns([1, 3])

        with col_img:
            if data['product_image']:
                st.image(data['product_image'], width=300)

        with col_stats:
            c1, c2, c3 = st.columns(3)
            c1.metric("Global Rating", f"⭐ {data['average_rating']}")
            c2.metric("Total Reviews", f"{data['rating_number']:,}")
            c3.metric("Sample Analyzed", data['analyzed_count'], help="Top helpful & recent reviews")

            # EXTRACTIVE SUMMARY
            st.success(f"""
            **AI Executive Summary:**
            {data['ai_summary']}
            """)

        st.divider()

        # 2. CHARTS
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### 🗣️ Topic Distribution")
            df_topics = pd.DataFrame(list(data['top_topics'].items()), columns=['Topic', 'Score'])
            fig = px.pie(df_topics, values='Score', names='Topic', hole=0.5, color_discrete_sequence=px.colors.sequential.RdBu)
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            st.markdown("#### ❤️ Sentiment Breakdown")
            df_sent = pd.DataFrame(list(data['sentiment_breakdown'].items()), columns=['Sentiment', 'Count'])
            fig2 = px.bar(df_sent, x='Sentiment', y='Count', color='Sentiment',
                         color_discrete_map={"Positive": "#2ecc71", "Negative": "#e74c3c", "Neutral": "#95a5a6"})
            st.plotly_chart(fig2, use_container_width=True)

        # 3. REVIEW EXPLORER
        st.subheader("🔍 Review Explorer")

        reviews_df = pd.DataFrame(data['reviews'])

        # Filter Logic
        filter_topic = st.selectbox("Filter by Topic:", ["All"] + list(data['top_topics'].keys()))
        if filter_topic != "All":
            reviews_df = reviews_df[reviews_df['topic'] == filter_topic]

        st.dataframe(
            reviews_df[['rating', 'topic', 'topic_score', 'text']],
            column_config={
                "rating": st.column_config.NumberColumn("⭐", format="%d", width="small"),
                "topic_score": st.column_config.ProgressColumn("Conf.", format="%.2f", min_value=0, max_value=1, width="small"),
                "text": st.column_config.TextColumn("Review", width="large")
            },
            hide_index=True
        )

        # 4. FEEDBACK
        st.divider()
        c1, c2, _ = st.columns([1, 1, 6])
        selected_asin = st.session_state.selected_asin
        if c1.button("👍 Verified"):
            st.toast("Feedback saved!", icon="💾")
            log_feedback(selected_asin, "POSITIVE")

        if c2.button("👎 Inaccurate"):
            st.toast("Negative feedback logged for review.", icon="🚩")
            log_feedback(selected_asin, "NEGATIVE")
