# Fraud Detection Web App (Flask + SHAP)

This is a minimal starter to:
- Log in
- Enter a single claim (claim-level model)
- Upload a CSV for provider-level scoring
- Show prediction results + SHAP explanations
- Export a CSV report
- Provide a simple JSON API endpoint (for Power BI or a GenAI bot)

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Set creds (optional)
export APP_USERNAME=admin
export APP_PASSWORD=admin123
export SECRET_KEY=change-me

# Place your models here:
#   models/claim_model.pkl
#   models/provider_model.pkl
# They should be sklearn-compatible and expose `predict` (optionally `predict_proba`).
# Include `feature_names_in_` on the estimator for best alignment.

python app.py
```

Open http://localhost:5000

## Notes
- If models are missing, the app auto-builds a tiny dummy RandomForest so pages work.
- SHAP images are saved into `static/shap/` and linked in the Results page.
- CSV uploads are stored in `uploads/`.
- Reports (scored CSVs) are saved to `reports/`.
- For production: replace the simple auth with Flask-Login + HTTPS + a real DB.
