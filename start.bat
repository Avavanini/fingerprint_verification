@echo off
echo Starting Fingerprint Verification Demo
echo ---------------------------------------

echo [1/2] Starting FastAPI Server in a new window...
start cmd /c "env\Scripts\uvicorn src.api.main:app --reload"

echo [2/2] Starting Streamlit App...
env\Scripts\streamlit run src\ui\app.py
