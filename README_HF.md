---
title: Product Review Analyzer
emoji: 📊
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
license: mit
app_port: 7860
---

# Product Review Analyzer - MLOps Project

An on-demand system for e-commerce review analysis and topic classification using Zero-Shot Learning.

## About This Application

This MLOps project demonstrates a complete pipeline from data ingestion to cloud deployment. The system uses **DeBERTa Zero-Shot Classification** to analyze Amazon product reviews and extract actionable insights across 6 predefined topics.

### Key Features

- **Zero-Shot Topic Classification**: Analyzes reviews for Quality, Price, Scent, Packaging, Safety, and Service
- **Real-time Inference**: Optimized for fast response times with batch processing
- **Interactive Dashboard**: Streamlit UI with data visualization
- **Production Ready**: Dockerized deployment with monitoring

### How to Use

1. Select a product from the dropdown menu or enter an ASIN ID
2. Click "Analyze Reviews" to process customer feedback
3. View topic distribution, sentiment breakdown, and individual reviews
4. Provide feedback using the Verified/Inaccurate buttons

### Technical Stack

- **Model**: DeBERTa-v3-base-mnli-fever-anli
- **Backend**: FastAPI with inference optimization
- **Frontend**: Streamlit with Plotly visualizations
- **Data**: 701K+ Amazon Beauty Reviews (DVC versioned)
- **Deployment**: Docker with Nginx reverse proxy

### Performance

- Average latency: ~1000ms per review (50 reviews analyzed)
- Cache-enabled for repeated requests
- Batch processing with `batch_size=16`
- Torch inference mode for 20-30% speedup

### Architecture

```
Frontend (Streamlit:8501) <-> Backend (FastAPI:8000) <-> DeBERTa Model
                     |
                 Nginx:7860 (Public)
```

## Project Team

- Eda Kucukkara (150210325)
- Revna Altinoz (150220756)

**Course**: YZV448E - MLOps (Istanbul Technical University)

---

For more details, visit the [GitHub repository](https://github.com/edakucukkara/mlops-review-analyzer)
