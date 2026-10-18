QQQ -> NQ Streamlit dashboard

Files:
- streamlit_app.py: Streamlit UI/dashboard
- qqq_nq_model.py: shared model logic (keep in the same folder)
- requirements_streamlit.txt: Python dependencies

Run locally:
1. Install Python 3.10+.
2. Put streamlit_app.py and qqq_nq_model.py in the same folder.
3. In a terminal in that folder:
   python -m venv .venv
   # macOS/Linux:
   source .venv/bin/activate
   # Windows PowerShell:
   .venv\Scripts\Activate.ps1
   pip install -r requirements_streamlit.txt
   streamlit run streamlit_app.py
4. A browser tab opens. Choose Yahoo data or upload CSVs, choose a forecast time, and click Run model and walk-forward backtest.

For Streamlit Community Cloud:
- Upload streamlit_app.py, qqq_nq_model.py, and requirements_streamlit.txt to one GitHub repository.
- Create a Streamlit app and select streamlit_app.py as the entry point.

Price CSV schema:
timestamp,open,high,low,close,volume

Options CSV schema:
session_date,forecast_time,qqq_atm_iv,qqq_iv_change,net_gex_norm,net_vanna_norm,net_dex_norm,gex_near_spot,call_wall_distance_pct,put_wall_distance_pct,gamma_flip_distance_pct

Options values must be from a snapshot available at or before the selected forecast time. Do not use future or end-of-day options data for earlier forecasts.
