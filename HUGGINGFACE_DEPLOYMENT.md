# Hugging Face Spaces Deployment Guide

## Prerequisites

1. **Hugging Face Account**
   - Go to https://huggingface.co/join
   - Sign up with email or GitHub

2. **Git LFS Installed**
   ```bash
   # Windows (with Git for Windows)
   git lfs install

   # Or download from: https://git-lfs.github.com/
   ```

---

## Step-by-Step Deployment

### Step 1: Create a New Space

1. Go to https://huggingface.co/new-space
2. Fill in the details:
   - **Space name**: `mlops-review-analyzer` (or your choice)
   - **License**: MIT
   - **SDK**: Docker
   - **Visibility**: Public (or Private if you prefer)
3. Click "Create Space"

### Step 2: Clone the Space Repository

After creating the Space, you'll get a repository URL. Clone it:

```bash
# Clone your new Space
git clone https://huggingface.co/spaces/YOUR_USERNAME/mlops-review-analyzer
cd mlops-review-analyzer
```

### Step 3: Copy Project Files

Copy the following files from your local project to the Space directory:

**Required Files:**
```bash
# Core application files
cp app.py ../mlops-review-analyzer/
cp ui.py ../mlops-review-analyzer/
cp requirements.txt ../mlops-review-analyzer/

# Docker configuration for HF Spaces
cp Dockerfile.huggingface ../mlops-review-analyzer/Dockerfile

# HF Spaces README with metadata
cp README_HF.md ../mlops-review-analyzer/README.md

# Data and config
cp -r data/ ../mlops-review-analyzer/
cp -r .streamlit/ ../mlops-review-analyzer/

# Optional helper files
cp drift_detector.py ../mlops-review-analyzer/
cp .dockerignore ../mlops-review-analyzer/
```

### Step 4: Commit and Push to Hugging Face

```bash
cd mlops-review-analyzer

# Add all files
git add .

# Commit
git commit -m "Initial deployment: MLOps Review Analyzer

- FastAPI backend with DeBERTa model
- Streamlit frontend with interactive dashboard
- Docker deployment with Nginx reverse proxy
- Performance optimizations included"

# Push to Hugging Face (will trigger automatic build)
git push
```

### Step 5: Wait for Build

1. Go to your Space URL: `https://huggingface.co/spaces/YOUR_USERNAME/mlops-review-analyzer`
2. The Space will automatically build (takes 5-15 minutes first time)
3. You'll see build logs in the "Logs" tab
4. Status will change from "Building" to "Running"

---

## Alternative: Quick Push from Current Directory

If you want to push directly without cloning:

```bash
# Add HF Space as remote
cd /path/to/mlops_project
git remote add hf https://huggingface.co/spaces/YOUR_USERNAME/mlops-review-analyzer

# Create deployment branch
git checkout -b hf-deployment

# Copy HF-specific files
cp Dockerfile.huggingface Dockerfile
cp README_HF.md README.md

# Stage HF deployment files
git add Dockerfile README.md app.py ui.py requirements.txt data/ .streamlit/ drift_detector.py .dockerignore

# Commit
git commit -m "Hugging Face Spaces deployment"

# Push to HF
git push hf hf-deployment:main --force
```

---

## Troubleshooting

### Build Fails

**Issue**: "Out of memory" or "Build timeout"
**Solution**: Reduce model size or use smaller data sample

**Issue**: "Port 7860 not responding"
**Solution**: Check Dockerfile.huggingface - nginx must listen on 7860

### Application Not Loading

**Issue**: "Application Error" after build
**Solution**: Check logs for Python errors. Common issues:
- Missing dependencies in requirements.txt
- Data files too large (>50MB requires Git LFS)
- Model download failures

### Git LFS Required

If `data/gold_reviews.parquet` is >50MB:

```bash
# Track large files with Git LFS
git lfs install
git lfs track "data/*.parquet"
git add .gitattributes
git commit -m "Track parquet files with Git LFS"
git push
```

---

## Post-Deployment

### Testing Your Deployed App

1. Visit: `https://huggingface.co/spaces/YOUR_USERNAME/mlops-review-analyzer`
2. Wait for "Running" status
3. Select a product and test analysis
4. Check response time (first request will be slower)

### Monitoring

- **Logs**: Click "Logs" tab to see application output
- **Metrics**: HF Spaces provides basic usage metrics
- **Updates**: Push new commits to update the Space automatically

### Sharing

Your Space is now public at:
```
https://huggingface.co/spaces/YOUR_USERNAME/mlops-review-analyzer
```

Share this link in your project report!

---

## Configuration Options

### Upgrade to GPU (If Needed)

If CPU inference is too slow:

1. Go to Space Settings
2. Select "GPU" hardware
3. Restart Space
4. Note: Requires HF Pro subscription ($9/month)

### Private Space

1. Go to Space Settings
2. Change visibility to "Private"
3. Share access with specific users

---

## Expected Performance on HF Spaces

- **Cold Start**: 30-60 seconds (model loading)
- **First Request**: ~60-90 seconds (50 reviews)
- **Cached Requests**: ~10-20 seconds
- **Concurrent Users**: Handles 1-2 users well
- **RAM Usage**: ~4-6 GB

---

## Summary

Once deployed, your MLOps project is:
- Publicly accessible
- Fully functional with all features
- Demonstrable for your course project
- Shareable via simple URL

**Next Steps**: Include HF Space URL in your final report and demo video!
