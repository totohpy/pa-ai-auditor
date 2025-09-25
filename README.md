# PA.AI – Planning Studio (Streamlit Cloud)

## Deploy (Streamlit Community Cloud)
1. Push this folder to a **public GitHub repo**.
2. Visit https://share.streamlit.io → **New app** → choose your repo/branch.
3. **Main file path**: `pa_ai_app_with_llm_fixed_v5_paassist.py`
4. (Optional) **Secrets**: add `TYPHOON_API_KEY = your_key_here` (or enter in the app UI).
5. Deploy.

## Data (FindingsLibrary)
- Upload inside the app or host it as a Google Sheets CSV URL.
- If you want a bundled static file, add `FindingsLibrary.csv` at repo root.

## Local run
```bash
pip install -r requirements.txt
streamlit run pa_ai_app_with_llm_fixed_v5_paassist.py --server.port 8501
```
