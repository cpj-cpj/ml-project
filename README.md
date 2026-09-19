# Sydney Housing Price Prediction — Part 5

## Files
- `app.py` — Streamlit deployment prototype.
- `master_dataset.csv` — training dataset.
- `requirements.txt` — Python dependencies.

## Run locally

1. Put `app.py` and `master_dataset.csv` in the same folder.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Start Streamlit:

```bash
streamlit run app.py
```

4. Open the local URL shown by Streamlit, normally `http://localhost:8501`.

## Deployment

The app can be deployed using Streamlit Community Cloud by placing these files in a GitHub repository and selecting `app.py` as the application entry point.

The application retrains the same Random Forest approach used in Part 3 when it starts. This keeps the prototype reproducible without requiring a separately saved binary model.

## Important limitation

The application provides an ML estimate rather than a formal valuation. It should be interpreted cautiously because the dataset is small and manually collected, and important property-level information such as condition, renovation quality, views and exact micro-location is not consistently available.
