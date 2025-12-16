# Streamlit Cloud Deployment with DVC

This guide explains how to deploy the MLOps Review Analyzer to Streamlit Cloud with DVC data versioning.

## Prerequisites

1. Google Cloud Service Account with access to your Google Drive DVC remote
2. Streamlit Cloud account connected to your GitHub

## Deployment Steps

### 1. Deploy to Streamlit Cloud

1. Go to https://share.streamlit.io/
2. Click "New app"
3. Configure:
   - **Repository**: `edakucukkara/mlops-review-analyzer`
   - **Branch**: `main`
   - **Main file path**: `streamlit_app.py`
4. Click "Deploy"

### 2. Configure Secrets

1. Go to your app settings: `https://share.streamlit.io/` > Your app > Settings > Secrets
2. Copy the contents from `.streamlit/secrets.toml.example`
3. Fill in your Google Drive credentials:
   - `folder_id`: Your Google Drive folder ID where DVC stores data
   - `credentials`: Your Google Service Account JSON

### 3. Get Google Drive Folder ID

Your DVC remote is configured to use Google Drive. To get the folder ID:

1. Open your DVC remote folder in Google Drive
2. The URL will look like: `https://drive.google.com/drive/folders/FOLDER_ID_HERE`
3. Copy the `FOLDER_ID_HERE` part

### 4. Create Google Service Account (if needed)

If you don't have a service account:

1. Go to https://console.cloud.google.com/
2. Create a new project or select existing
3. Enable Google Drive API
4. Go to IAM & Admin > Service Accounts
5. Create Service Account
6. Create and download JSON key
7. Share your Google Drive folder with the service account email

### 5. Restart App

After configuring secrets:
1. Go to app settings
2. Click "Reboot app"
3. The app will pull data from DVC on startup

## How It Works

1. `streamlit_app.py` checks if data exists locally
2. If not, it calls `setup_dvc.py`
3. `setup_dvc.py` reads secrets and configures DVC
4. DVC pulls `gold_reviews.parquet` from Google Drive
5. App loads and runs normally

## Troubleshooting

### "No such file or directory: data/gold_reviews.parquet"

- Check that secrets are configured correctly
- Verify Google Service Account has access to Drive folder
- Check app logs for DVC errors

### DVC pull fails

- Verify `folder_id` matches your Google Drive DVC remote
- Ensure service account JSON is valid
- Check Drive API is enabled in Google Cloud

## Local Development

For local development, the app uses the data pulled by DVC:

```bash
dvc pull
python -m streamlit run streamlit_app.py
```

## Notes

- First startup will be slow (pulling 107 MB data)
- Subsequent starts use cached data
- Data is stored in Streamlit Cloud's persistent storage
