"""
DVC Setup Script for Streamlit Cloud
This script configures DVC with Google Drive credentials from Streamlit secrets
and pulls the data before the app starts.
"""
import os
import subprocess
import streamlit as st

def setup_dvc():
    """Configure DVC and pull data from Google Drive"""

    # Check if data already exists
    if os.path.exists("data/gold_reviews.parquet"):
        print("Data already exists, skipping DVC pull")
        return True

    try:
        # Get Google Drive credentials from Streamlit secrets
        if hasattr(st, 'secrets') and 'gdrive' in st.secrets:
            # Configure DVC remote with Google Drive
            gdrive_folder_id = st.secrets['gdrive']['folder_id']

            # Set up DVC remote
            subprocess.run([
                'dvc', 'remote', 'modify', 'myremote',
                'gdrive_use_service_account', 'true'
            ], check=True)

            subprocess.run([
                'dvc', 'remote', 'modify', 'myremote',
                'gdrive_service_account_json_file_path',
                '/tmp/gdrive_credentials.json'
            ], check=True)

            # Write credentials to temp file
            import json
            with open('/tmp/gdrive_credentials.json', 'w') as f:
                json.dump(st.secrets['gdrive']['credentials'], f)

            # Pull data from DVC
            print("Pulling data from DVC...")
            subprocess.run(['dvc', 'pull'], check=True)
            print("DVC pull completed successfully!")
            return True
        else:
            print("WARNING: No DVC credentials found in secrets. Using local data if available.")
            return False

    except Exception as e:
        print(f"Error setting up DVC: {e}")
        return False

if __name__ == "__main__":
    setup_dvc()
