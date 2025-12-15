import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import datetime
import sys

def log_feedback(asin, feedback_type):
    """Simple HITL logging function for user feedback tracking"""
    timestamp = datetime.datetime.now().isoformat()
    # This log will be written to the Streamlit container logs
    print(f"HITL_LOG: {timestamp}, ASIN: {asin}, Feedback: {feedback_type}")

    # Flush stdout to ensure immediate logging
    sys.stdout.flush()

# --- CONFIGURATION ---
#API_URL = "http://127.0.0.1:8000"
#API_URL = "http://localhost:8000"
API_URL = "http://backend:8000"
st.set_page_config(page_title="Product Review Analyzer", page_icon="🛍️", layout="wide")

# --- SESSION STATE ---
if 'analysis_data' not in st.session_state: st.session_state.analysis_data = None
if 'selected_asin' not in st.session_state: st.session_state.selected_asin = None

# --- SIDEBAR ---
with st.sidebar:
    st.header("🔎 Select Product")
    
    try:
        if 'product_menu' not in st.session_state:
            with st.spinner("Connecting to MLOps Backend..."):
                st.session_state.product_menu = requests.get(f"{API_URL}/products").json()
        
        # Create a dictionary for the dropdown
        menu_options = {p['title'][:60] + "...": p['asin'] for p in st.session_state.product_menu}
        
        # Add a "None" option to show the landing page
        options_list = ["-- Select a Product --"] + list(menu_options.keys())
        selected_name = st.selectbox("Popular Products:", options_list)
        
        if selected_name != "-- Select a Product --":
            final_asin = menu_options[selected_name]
        else:
            final_asin = None

        st.divider()
        custom_asin = st.text_input("Or enter ASIN ID:", placeholder="e.g. B00YQ6X8EO")
        if custom_asin: final_asin = custom_asin

        if final_asin:
            if st.button("🚀 Analyze Reviews", type="primary", use_container_width=True):
                st.session_state.selected_asin = final_asin
                # Reset data to trigger reload
                st.session_state.analysis_data = None 
                
    except Exception as e:
        st.error("⚠️ Backend Offline. Run: python -m uvicorn src.app:app --reload")

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
        try:
            with st.spinner("🧠 AI is processing reviews..."):
                resp = requests.get(f"{API_URL}/analyze/{st.session_state.selected_asin}")
                if resp.status_code == 200:
                    st.session_state.analysis_data = resp.json()
                else:
                    st.error("Product not found or not enough reviews.")
        except:
            st.error("Connection Error.")

    data = st.session_state.analysis_data
    
    if data:
        # 1. HEADER SECTION
        st.subheader(data['product_title'])
        col_img, col_stats = st.columns([1, 3])
        
        with col_img:
            if data['product_image']:
                # FIX: Removed use_container_width to fix warning, replaced with explicit width if needed or just let it be
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

        # FIX: Removed use_container_width=True from st.dataframe to fix warning.
        st.dataframe(
            reviews_df[['rating', 'topic', 'topic_score', 'text']],
            column_config={
                "rating": st.column_config.NumberColumn("⭐", format="%d", width="small"),
                "topic_score": st.column_config.ProgressColumn("Conf.", format="%.2f", min_value=0, max_value=1, width="small"),
                "text": st.column_config.TextColumn("Review", width="large")
            },
            hide_index=True
        )

        # 4. FEEDBACK (Restored BOTH Buttons)
        st.divider()
        c1, c2, _ = st.columns([1, 1, 6])
        selected_asin = st.session_state.selected_asin
        if c1.button("👍 Verified"):
            st.toast("Feedback saved!", icon="💾")
            # Log positive HITL feedback
            log_feedback(selected_asin, "POSITIVE")

        if c2.button("👎 Inaccurate"):
            st.toast("Negative feedback logged for review.", icon="🚩")
            # Log negative HITL feedback
            log_feedback(selected_asin, "NEGATIVE")

# python -m streamlit run ui.py