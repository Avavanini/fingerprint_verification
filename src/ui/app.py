"""
app.py — Streamlit Frontend Demo UI for Fingerprint Verification.
"""

import streamlit as st
import requests
import os
from PIL import Image

# Configuration
API_URL = "http://localhost:8000/api/v1"
API_KEY = os.environ.get("FINGERPRINT_API_KEY", "secret-demo-key-123")
HEADERS = {"X-API-Key": API_KEY}

st.set_page_config(
    page_title="Fingerprint Biometrics Demo",
    page_icon="🖐️",
    layout="centered"
)

st.title("🖐️ Fingerprint Verification System")
st.markdown("""
Welcome to the Fingerprint Biometrics Demo! 
This UI connects to our high-performance FastAPI backend, which uses a Siamese CNN to extract and compare 128-D Deep Embeddings.
""")

# Create Tabs
tab_enroll, tab_verify = st.tabs(["📝 Enroll Fingerprint", "🔐 Verify Identity"])

# --- TAB 1: ENROLLMENT ---
with tab_enroll:
    st.header("Enroll a New User")
    st.write("Upload a fingerprint image to extract its deep embedding and save it to the database.")
    
    enroll_user_id = st.text_input("User ID (e.g., user_123)", key="enroll_id")
    enroll_file = st.file_uploader("Choose a Fingerprint Image", type=["png", "jpg", "jpeg", "bmp"], key="enroll_file")
    
    if st.button("Enroll User", type="primary"):
        if not enroll_user_id:
            st.error("Please enter a User ID.")
        elif not enroll_file:
            st.error("Please upload a fingerprint image.")
        else:
            with st.spinner("Processing image and extracting embedding..."):
                try:
                    # Show image
                    img = Image.open(enroll_file)
                    st.image(img, caption="Uploaded Fingerprint", width=250)
                    
                    # Prepare request
                    enroll_file.seek(0)
                    files = {"file": (enroll_file.name, enroll_file, enroll_file.type)}
                    data = {"user_id": enroll_user_id}
                    
                    response = requests.post(f"{API_URL}/enroll", headers=HEADERS, data=data, files=files)
                    
                    if response.status_code == 200:
                        res_data = response.json()
                        st.success(f"✅ Success! {res_data['message']}")
                    else:
                        st.error(f"❌ Failed: {response.json().get('detail', 'Unknown error')}")
                except requests.exceptions.ConnectionError:
                    st.error("❌ Could not connect to the API. Is the FastAPI server running?")
                except Exception as e:
                    st.error(f"❌ An error occurred: {e}")

# --- TAB 2: VERIFICATION ---
with tab_verify:
    st.header("Verify an Identity")
    st.write("Upload a probe fingerprint and enter a Claimed User ID to verify a match against the database.")
    
    verify_user_id = st.text_input("Claimed User ID", key="verify_id")
    verify_file = st.file_uploader("Choose a Probe Fingerprint", type=["png", "jpg", "jpeg", "bmp"], key="verify_file")
    
    if st.button("Verify Identity", type="primary"):
        if not verify_user_id:
            st.error("Please enter a Claimed User ID.")
        elif not verify_file:
            st.error("Please upload a probe fingerprint image.")
        else:
            with st.spinner("Extracting live embedding and comparing..."):
                try:
                    # Show image
                    img = Image.open(verify_file)
                    st.image(img, caption="Probe Fingerprint", width=250)
                    
                    # Prepare request
                    verify_file.seek(0)
                    files = {"file": (verify_file.name, verify_file, verify_file.type)}
                    data = {"user_id": verify_user_id}
                    
                    response = requests.post(f"{API_URL}/verify", headers=HEADERS, data=data, files=files)
                    
                    if response.status_code == 200:
                        res_data = response.json()
                        is_match = res_data["match"]
                        score = res_data["score"]
                        threshold = res_data["threshold"]
                        
                        if is_match:
                            st.success(f"🎉 **MATCH SUCCESSFUL!** Identity Confirmed.")
                        else:
                            st.error(f"🚫 **MATCH FAILED.** Access Denied.")
                            
                        # Show metrics
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric(label="Similarity Score", value=f"{score:.4f}")
                        with col2:
                            st.metric(label="Required Threshold", value=f"{threshold:.4f}")
                            
                    elif response.status_code == 404:
                        st.warning(f"⚠️ {response.json().get('detail')}")
                    else:
                        st.error(f"❌ Failed: {response.json().get('detail', 'Unknown error')}")
                        
                except requests.exceptions.ConnectionError:
                    st.error("❌ Could not connect to the API. Is the FastAPI server running?")
                except Exception as e:
                    st.error(f"❌ An error occurred: {e}")

st.markdown("---")
st.caption("Powered by FastAPI & PyTorch • Developed for the Fingerprint Verification System")
