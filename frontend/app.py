import os
import uuid
import joblib
import shap
import requests
import os
import json
import time
import pypdf
from io import BytesIO
from flask_cors import CORS
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from flask import send_from_directory
from flask import request, jsonify, make_response
import openai # Not used in the provided context, but kept
import pymysql
import lightgbm as lgb # Not explicitly used for model, but kept
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
import pickle # Ensure pickle is imported

from flask import Flask, render_template, request, redirect, url_for, flash, session
from werkzeug.utils import secure_filename
from pathlib import Path

# =========================
# CONFIG
# =========================
BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
REPORT_DIR = BASE_DIR / "reports"
MODEL_DIR = BASE_DIR / "models"
STATIC_DIR = BASE_DIR / "static/shap" # This path is for SHAP images, not general static files

# Ensure static/images directory exists for new UI elements (logo, avatar)
(BASE_DIR / "static" / "images").mkdir(exist_ok=True)

for d in [UPLOAD_DIR, REPORT_DIR, MODEL_DIR, STATIC_DIR]:
    d.mkdir(exist_ok=True)

ALLOWED_EXTENSIONS = {"csv"}

app = Flask(__name__)
app.secret_key = "supersecret"
CORS(app)

# =========================
# LOAD MODEL + FEATURES
# =========================
# Ensure these model files exist in the 'models' directory
try:
    rf_bal = joblib.load(MODEL_DIR / "rf_bal.pkl")
    feature_columns = joblib.load(MODEL_DIR / "provider_features.pkl")
    CLAIM_MODEL_PATH = MODEL_DIR / "fraud_model_pipeline.pkl"
    fraud_model = joblib.load(CLAIM_MODEL_PATH) if CLAIM_MODEL_PATH.exists() else None
except FileNotFoundError as e:
    print(f"Error loading model files: {e}. Please ensure 'rf_bal.pkl', 'provider_features.pkl', and 'fraud_model_pipeline.pkl' are in the 'models' directory.")
    rf_bal = None
    feature_columns = []
    fraud_model = None


# =========================
# HELPER FUNCTIONS
# =========================
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def logged_in():
    return "user" in session




def store_fraud_to_db(df, host, user, password, database, table="fraud_providers"):
    """
    df: DataFrame with columns 'Provider', 'FraudPrediction', 'FraudProbability'
    Only rows with FraudPrediction == 1 will be stored.
    """
    # Filter only fraud providers
    fraud_df = df[df["FraudPrediction"] == 1].copy() # Only store fraud predictions
    if fraud_df.empty:
        print("No fraud providers to insert.")
        return

    # Connect to MySQL
    # IMPORTANT: Replace with your actual MySQL credentials and host
    try:
        conn = pymysql.connect(
            host=host, # e.g., 'localhost'
            user=user, # e.g., 'root'
            password=password, # e.g., 'chudar27@'
            database=database, # e.g., 'FraudProviders'
            charset="utf8mb4",
            cursorclass=pymysql.cursors.DictCursor
        )
    except pymysql.Error as e:
        print(f"Error connecting to MySQL: {e}")
        return

    try:
        with conn.cursor() as cursor:
            # Create table if not exists
            cursor.execute(f"""
                CREATE TABLE IF NOT EXISTS {table} (
                    Provider VARCHAR(255) PRIMARY KEY,
                    FraudPrediction INT,
                    FraudProbability FLOAT
                )
            """)

            # Insert each row
            for _, row in fraud_df.iterrows():
                cursor.execute(f"""
                    INSERT INTO {table} (Provider, FraudPrediction, FraudProbability)
                    VALUES (%s, %s, %s)
                    ON DUPLICATE KEY UPDATE
                        FraudPrediction = VALUES(FraudPrediction),
                        FraudProbability = VALUES(FraudProbability)
                """, (row["Provider"], int(row["FraudPrediction"]), float(row["FraudProbability"])))

        conn.commit()
        print(f"{len(fraud_df)} fraud providers inserted/updated.")
    finally:
        conn.close()


def generate_claim_pdf(filepath, result, new_claim_data=None, shap_values=None, feature_names=None):
    """
    result = {
        "prediction": int,
        "score": float,
        "shap_image": str (path to shap plot),
        "features": dict
    }
    new_claim_data: DataFrame with the input features for the claim.
    shap_values: SHAP values for the prediction.
    feature_names: List of feature names corresponding to shap_values.
    """
    doc = SimpleDocTemplate(filepath, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []

    # Title
    story.append(Paragraph("<b>Fraud Detection - Claim Level Report</b>", styles["h1"]))
    story.append(Spacer(1, 12))

    # Prediction
    pred_text = "Fraudulent ⚠" if result["prediction"] == 1 else "Legitimate ✅"
    story.append(Paragraph(f"<b>Prediction:</b> {pred_text}", styles["Normal"]))
    story.append(Spacer(1, 6))

    # Confidence
    if "score" in result and result["score"] is not None:
        story.append(Paragraph(f"<b>Confidence:</b> {result['score']:.2f}", styles["Normal"]))
        story.append(Spacer(1, 12))

    # SHAP image (if available from previous logic)
    # Ensure the path is correct for ReportLab
    if result.get("shap_image") and Path(result["shap_image"]).exists():
        story.append(Paragraph("<b>SHAP Feature Importance (Overall)</b>", styles["h3"]))
        story.append(Spacer(1, 6))
        # ReportLab Image expects a string path
        story.append(Image(str(result["shap_image"]), width=400, height=200))
        story.append(Spacer(1, 12))

    # Features table (from original logic)
    if result.get("features"):
        story.append(Paragraph("<b>Input Features</b>", styles["h3"]))
        data = [["Feature", "Value"]]
        for k, v in result["features"].items():
            data.append([k, str(v)])

        table = Table(data, colWidths=[200, 200])
        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.grey),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
            ("ALIGN", (0, 0), (-1, -1), "LEFT"),
            ("GRID", (0, 0), (-1, -1), 1, colors.black),
        ]))
        story.append(table)
        story.append(Spacer(1, 12))


    # --- New SHAP Explanation Sections ---
    if shap_values is not None and feature_names is not None and new_claim_data is not None:
        story.append(Paragraph("<b>Detailed Prediction Explanation (SHAP)</b>", styles['h2']))
        story.append(Spacer(1, 12))

        # --- Prediction Summary Section (from new code) ---
        story.append(Paragraph("<b>1. Prediction Summary</b>", styles['h2']))
        story.append(Spacer(1, 6))

        prediction_text = "Not Fraudulent" if result["prediction"] == 0 else "Potentially Fraudulent"
        prediction_color = "green" if result["prediction"] == 0 else "red"

        # Define a custom style for colored text if needed, or use HTML font tag
        # styles.add(ParagraphStyle(name='ColoredNormal', parent=styles['Normal'], textColor=colors.green))
        story.append(Paragraph(f"<b>Predicted Fraud Class:</b> <font color='{prediction_color}'>{prediction_text}</font>", styles['Normal']))
        story.append(Paragraph(f"<b>Predicted Fraud Probability:</b> {result['score']:.4f}", styles['Normal']))
        story.append(Spacer(1, 12))

        # --- New Section: Explanation Paragraph ---
        story.append(Paragraph("<b>2. Reason for Prediction</b>", styles['h2']))
        story.append(Spacer(1, 6))

        shap_df = pd.DataFrame({'Feature': feature_names, 'SHAP Value': shap_values})
        positive_contributors = shap_df[shap_df['SHAP Value'] > 0].sort_values(by='SHAP Value', ascending=False)
        negative_contributors = shap_df[shap_df['SHAP Value'] < 0].sort_values(by='SHAP Value', ascending=True)

        if result["prediction"] == 1: # Fraudulent
            if not positive_contributors.empty:
                top_positive_feature = positive_contributors.iloc[0]['Feature']
                # Ensure feature exists in new_claim_data before accessing
                top_positive_value = new_claim_data[top_positive_feature].iloc[0] if top_positive_feature in new_claim_data.columns else "N/A"
                explanation = f"This claim is flagged as potentially fraudulent primarily due to the <b>{top_positive_feature}</b> feature, which has a value of <b>{top_positive_value}</b>. This value is unusual for a non-fraudulent claim and significantly contributed to the model's prediction. The model's confidence in this assessment is {result['score']:.2%}."
            else:
                explanation = "This claim is flagged as potentially fraudulent, but the model did not identify a clear top feature contributing to the decision. This may occur in complex cases where multiple features have a small, combined effect."
        else: # Not Fraudulent
            if not negative_contributors.empty:
                top_negative_feature = negative_contributors.iloc[0]['Feature']
                # Ensure feature exists in new_claim_data before accessing
                top_negative_value = new_claim_data[top_negative_feature].iloc[0] if top_negative_feature in new_claim_data.columns else "N/A"
                explanation = f"The claim is not flagged as fraudulent, largely because of the <b>{top_negative_feature}</b> feature, which has a value of <b>{top_negative_value}</b>. This feature's value aligns with what the model has learned from past non-fraudulent claims, pushing the prediction away from the fraud class. The model's confidence in this assessment is {1 - result['score']:.2%}."
            else:
                explanation = "This claim is not flagged as fraudulent, but the model did not identify a clear top feature preventing a fraudulent prediction. This may occur in complex cases where multiple features have a small, combined effect."

        story.append(Paragraph(explanation, styles['Normal']))
        story.append(Spacer(1, 12))

        # --- Top 5 Contributing Features (Positive) ---
        story.append(Paragraph("<b>3. Features Contributing to the Fraud Prediction</b>", styles['h2']))
        story.append(Spacer(1, 6))

        if not positive_contributors.empty:
            data = [['Feature', 'Contribution to Fraud']]
            for index, row in positive_contributors.head(5).iterrows():
                data.append([row['Feature'], f"{row['SHAP Value']:.4f}"])

            table = Table(data, colWidths=[200, 150])
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ]))
            story.append(table)
        else:
            story.append(Paragraph("No features contributed positively to the fraud prediction for this claim.", styles['Normal']))
        story.append(Spacer(1, 12))

        # --- Top 5 Contributing Features (Negative) ---
        story.append(Paragraph("<b>4. Features Preventing a Fraud Prediction</b>", styles['h2']))
        story.append(Spacer(1, 6))

        if not negative_contributors.empty:
            data = [['Feature', 'Contribution to Not Fraud']]
            for index, row in negative_contributors.head(5).iterrows():
                data.append([row['Feature'], f"{row['SHAP Value']:.4f}"])

            table = Table(data, colWidths=[200, 150])
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.lightblue),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ]))
            story.append(table)
        else:
            story.append(Paragraph("No features contributed negatively to the fraud prediction for this claim.", styles['Normal']))
        story.append(Spacer(1, 12))

    doc.build(story)

# =========================
# PIPELINE FUNCTIONS (No changes needed here)
# =========================
def parse_dates(df, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")
    return df

date_cols_in  = ["ClaimStartDt", "ClaimEndDt", "AdmissionDt", "DischargeDt"]
date_cols_out = ["ClaimStartDt", "ClaimEndDt"]

def build_claim_features(claims_df, bene_df, high_value_q=0.95):
    df = claims_df.copy()
    df["TotalPaid"] = df["InscClaimAmtReimbursed"].fillna(0) + df["DeductibleAmtPaid"].fillna(0)
    df["ClaimDuration"] = ((df["ClaimEndDt"] - df["ClaimStartDt"]).dt.days).fillna(0).clip(lower=0) + 1
    df["ClaimYear"]  = df["ClaimStartDt"].dt.year
    df["ClaimMonth"] = df["ClaimStartDt"].dt.month
    df["ClaimYM"]    = df["ClaimStartDt"].dt.to_period("M").astype(str)
    df["IsWeekend"]  = df["ClaimStartDt"].dt.weekday >= 5
    proc_cols = [c for c in df.columns if c.startswith("ClmProcedureCode_")]
    diag_cols = [c for c in df.columns if c.startswith("ClmDiagnosisCode_")]
    df["NumProcedures"] = df[proc_cols].notna().sum(axis=1) if proc_cols else 0
    df["NumDiagnoses"]  = df[diag_cols].notna().sum(axis=1) if diag_cols else 0

    join_cols = ["BeneID", "Gender", "Race", "State", "County", "RenalDiseaseIndicator"]
    join_cols += [c for c in bene_df.columns if c.startswith("ChronicCond_")]
    join_cols = [c for c in join_cols if c in bene_df.columns]
    df = df.merge(bene_df[join_cols + ["DOB", "DOD"]], on="BeneID", how="left")
    df["AgeAtClaim"] = ((df["ClaimStartDt"] - df["DOB"]).dt.days / 365.25).astype(float)
    df["IsDeceasedAtClaim"] = (df["DOD"].notna()) & (df["DOD"] <= df["ClaimEndDt"])

    hv_thr = df["TotalPaid"].quantile(high_value_q)
    df["IsHighValueClaim"] = df["TotalPaid"] > hv_thr
    df.attrs["high_value_threshold"] = float(hv_thr)

    df["PaidPerDay"] = df["TotalPaid"] / df["ClaimDuration"]
    df["ProcPerDay"] = df["NumProcedures"] / df["ClaimDuration"]

    df = df.sort_values(["Provider", "ClaimStartDt"])
    df["PrevClaimDate"] = df.groupby("Provider")["ClaimStartDt"].shift(1)
    df["InterClaimGapDays"] = (df["ClaimStartDt"] - df["PrevClaimDate"]).dt.days
    df["InterClaimGapDays"] = df["InterClaimGapDays"].fillna(df["InterClaimGapDays"].median())

    if "ClmDiagnosisCode_1" in df.columns:
        diag_mean = df.groupby("ClmDiagnosisCode_1")["TotalPaid"].mean().rename("Diag1_MeanPaid")
        df = df.merge(diag_mean, left_on="ClmDiagnosisCode_1", right_index=True, how="left")
        df["DiagPaidDiff"] = df["TotalPaid"] - df["Diag1_MeanPaid"].fillna(df["TotalPaid"].median())
    else:
        df["DiagPaidDiff"] = 0.0

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    for c in numeric_cols:
        if df[c].isna().any():
            df[f"{c}_missing"] = df[c].isna().astype(int)
            df[c] = df[c].fillna(df[c].median())

    cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
    for c in cat_cols:
        df[c] = df[c].fillna("Missing")

    return df

def month_index(s):
    df = s.str.split("-", expand=True).astype(int)
    y = df[0]; m = df[1]
    return (y - y.min()) * 12 + (m - m.min())

def aggregate_provider_features(df_claims_fe, df_bene):
    df = df_claims_fe.copy()
    g = df.groupby("Provider")
    agg = pd.DataFrame(index=g.size().index)
    agg["Claims_Count"] = g.size()
    agg["TotalPaid_Sum"] = g["TotalPaid"].sum()
    agg["TotalPaid_Mean"] = g["TotalPaid"].mean()
    agg["TotalPaid_Median"] = g["TotalPaid"].median()
    agg["TotalPaid_Std"] = g["TotalPaid"].std().fillna(0)
    agg["TotalPaid_Max"] = g["TotalPaid"].max()
    agg["ClaimDuration_Mean"] = g["ClaimDuration"].mean()
    agg["ClaimDuration_Max"]  = g["ClaimDuration"].max()
    agg["PaidPerDay_Mean"] = g["PaidPerDay"].mean()
    agg["PaidPerDay_Std"]  = g["PaidPerDay"].std().fillna(0)
    agg["NumProcedures_Mean"] = g["NumProcedures"].mean()
    agg["NumDiagnoses_Mean"]  = g["NumDiagnoses"].mean()
    agg["UniquePatients"] = g["BeneID"].nunique()
    agg["IsWeekend_Share"] = g["IsWeekend"].mean()
    agg["HighValue_Share"] = g["IsHighValueClaim"].mean()
    agg["InterClaimGap_Mean"] = g["InterClaimGapDays"].mean()
    agg["InterClaimGap_Std"]  = g["InterClaimGapDays"].std().fillna(0)

    ct = df.pivot_table(index="Provider", columns="ClaimType", values="ClaimID", aggfunc="count", fill_value=0)
    for col in ["Inpatient", "Outpatient"]:
        if col not in ct.columns: ct[col] = 0
    agg["Inpatient_Share"]  = ct["Inpatient"] / (ct["Inpatient"] + ct["Outpatient"] + 1e-6)
    agg["Outpatient_Share"] = 1 - agg["Inpatient_Share"]

    monthly = df.groupby(["Provider", "ClaimYM"]).size().rename("Cnt").reset_index()
    mm = monthly.groupby("Provider")["Cnt"].agg(["mean","std","max"])
    agg["MonthlyCnt_Std"] = mm["std"].fillna(0)
    by_p = []
    for p, sub in monthly.groupby("Provider"):
        x = month_index(sub["ClaimYM"])
        slope = np.polyfit(x, sub["Cnt"].values, 1)[0] if len(x)>=2 else 0.0
        by_p.append((p, slope))
    growth_df = pd.DataFrame(by_p, columns=["Provider", "MonthlyCnt_Slope"]).set_index("Provider")
    agg = agg.join(growth_df, how="left").fillna({"MonthlyCnt_Slope":0.0})

    diag_cols = [c for c in df.columns if c.startswith("ClmDiagnosisCode_")]
    proc_cols = [c for c in df.columns if c.startswith("ClmProcedureCode_")]
    def nunique_flat(group, cols):
        vals = pd.unique(pd.concat([group[c].astype(str) for c in cols if c in group], axis=0))
        vals = vals[vals != "Missing"]
        return len(vals)
    diag_div = df.groupby("Provider").apply(lambda g: nunique_flat(g, diag_cols)).rename("DiagCodes_Unique")
    proc_div = df.groupby("Provider").apply(lambda g: nunique_flat(g, proc_cols)).rename("ProcCodes_Unique")
    agg = agg.join(diag_div).join(proc_div)

    chronic_cols = [c for c in df.columns if c.startswith("ChronicCond_")]
    for c in chronic_cols:
        df[c] = df[c].map({1: 1, 2: 0})
        agg[f"{c}_Rate"] = g[c].mean()

    if "Gender" in df.columns:
        agg["Share_Male"] = g["Gender"].apply(lambda s: (s==2).mean() if hasattr(s,"mean") else 0)
    if "Race" in df.columns:
        for r in sorted(df["Race"].dropna().unique()):
            agg[f"Race_{r}_Share"] = g["Race"].apply(lambda s, r= r: (s==r).mean())
    if "RenalDiseaseIndicator" in df.columns:
        agg["Renal_Y_Share"] = g["RenalDiseaseIndicator"].apply(lambda s:(s.astype(str)=="Y").mean())
    if "State" in df.columns:
        state_mean = df.groupby("State")["TotalPaid"].mean().rename("StateMeanPaid")
        tmp = df[["Provider","State","TotalPaid"]].merge(state_mean, on="State", how="left")
        tmp["Paid_vs_State"] = tmp["TotalPaid"] / (tmp["StateMeanPaid"] + 1e-6)
        agg["Paid_vs_State_Mean"] = tmp.groupby("Provider")["Paid_vs_State"].mean()

    agg.reset_index(inplace=True)
    return agg

def apply_pipeline_to_test(path_in, path_out, path_bene, rf_model, feature_columns):
    if rf_model is None or not feature_columns:
        raise ValueError("Random Forest model or feature columns not loaded.")

    df_bene = pd.read_csv(path_bene)
    for c in ["DOB","DOD"]:
        if c in df_bene.columns:
            df_bene[c] = pd.to_datetime(df_bene[c], errors="coerce")
    df_in = pd.read_csv(path_in)
    df_in = parse_dates(df_in, date_cols_in)
    df_out = pd.read_csv(path_out)
    df_out = parse_dates(df_out, date_cols_out)
    df_out["ClaimType"] = "Outpatient"
    df_claims = pd.concat([df_in, df_out], ignore_index=True, sort=False)
    claims_fe = build_claim_features(df_claims, df_bene)
    provider_features = aggregate_provider_features(claims_fe, df_bene)
    provider_features = provider_features.replace([np.inf, -np.inf, np.nan], 0)

    X_test = provider_features.copy()
    prov_ids = X_test["Provider"].values
    X_test = X_test.drop(columns=["Provider"], errors="ignore")
    X_test = X_test.replace([np.inf, -np.inf], np.nan).fillna(0)
    for col in feature_columns:
        if col not in X_test.columns:
            X_test[col] = 0
    X_test = X_test[feature_columns]
    pred_test = rf_model.predict(X_test)
    proba_test = rf_model.predict_proba(X_test)[:,1]

    out = pd.DataFrame({
        "Provider": prov_ids,
        "FraudPrediction": pred_test,
        "FraudProbability": proba_test
    })
    return out, claims_fe, provider_features

# =========================
# FLASK ROUTES
# =========================
@app.route("/home")
def home():
    if not logged_in():
        return redirect(url_for("login"))
    return render_template("home.html")

@app.route("/", methods=["GET","POST"])
def login():
    if request.method=="POST":
        user=request.form.get("username")
        pw=request.form.get("password")
        if user=="admin" and pw=="password":
            session["user"]=user
            flash("Logged in successfully!", "success") # Added flash message
            return redirect(url_for("home"))
        flash("Invalid username or password.", "error") # More specific error
    return render_template("login.html")

@app.route("/dashboard")
def dashboard():
    if not logged_in():
        flash("Please log in to access the dashboard.", "info") # Added flash message
        return redirect(url_for("login"))

    # Pass a variable to the template to control which tab is active on load
    # This allows direct links like /dashboard#claim to work better
    active_tab = request.args.get('tab', 'claim') # Default to 'claim' tab
    return render_template("dashboard.html", active_tab=active_tab)


@app.route("/single-claim", methods=["GET", "POST"])
def single_claim():
    if not logged_in():
        flash("Please log in to analyze claims.", "info")
        return redirect(url_for("login"))

    if fraud_model is None:
        flash("Claim-level model not found. Please ensure 'fraud_model_pipeline.pkl' is in the 'models' directory.", "error")
        # Redirect back to dashboard, potentially to the claim tab
        return redirect(url_for("dashboard", tab='claim'))

    if request.method == "POST":
        try:
            data = {
                'Claim_Duration': [int(request.form['claim_duration'])],
                'LOS': [int(request.form['los'])],
                'Claim_Amount': [float(request.form['claim_amount'])],
                'DeductiblePaid': [float(request.form['deductible_paid'])],
                'Pay_Ratio': [float(request.form['pay_ratio'])],
                'Num_Diags': [int(request.form['num_diags'])],
                'Num_Procs': [int(request.form['num_procs'])],
                'Patient_Age': [int(request.form['patient_age'])],
                'Death_During_Claim': [int(request.form['death_during_claim'])],
                'Is_Inpatient': [int(request.form['is_inpatient'])],
                'Is_Outpatient': [int(request.form['is_outpatient'])],
                'Gender': [int(request.form['gender'])],
                'State': [int(request.form['state'])],
                'Chronic_Count': [int(request.form['chronic_count'])],
                'Renal_Disease': [int(request.form['renal_disease'])],
                'Num_Physicians': [int(request.form['num_physicians'])],
                'Attending_Operating_Same': [int(request.form['attending_operating_same'])],
                'Claim_Month': [int(request.form['claim_month'])],
                'Claim_Year': [int(request.form['claim_year'])],
                'Weekend_Admission': [int(request.form['weekend_admission'])]
            }
            df = pd.DataFrame(data)
            pred = int(fraud_model.predict(df)[0])
            # label = 'Fraudulent ⚠' if pred == 1 else 'Legitimate ✅' # Label is handled in template

            # Prepare result for template
            result = {
                "prediction": pred,
                "score": fraud_model.predict_proba(df)[0][1],
                "features": data
            }

            # --- SHAP Explanation Logic ---
            shap_values = None
            feature_names = None
            shap_image_path = None # Initialize SHAP image path

            # Check if the model is a pipeline and has the necessary steps for SHAP
            if hasattr(fraud_model, 'named_steps') and 'scaler' in fraud_model.named_steps and 'classifier' in fraud_model.named_steps:
                scaler = fraud_model.named_steps['scaler']
                model = fraud_model.named_steps['classifier']

                # Transform the new data using the scaler from the pipeline
                transformed_data = scaler.transform(df)

                # Get the feature names (they remain the same after scaling)
                feature_names = df.columns.tolist()

                try:
                    explainer = shap.TreeExplainer(model)
                    # For binary classification, shap_values can be a list of two arrays.
                    # We usually take the SHAP values for the positive class (index 1).
                    shap_values = explainer.shap_values(transformed_data)[1] if isinstance(explainer.shap_values(transformed_data), list) else explainer.shap_values(transformed_data)

                    # Generate SHAP force plot image
                    # Ensure matplotlib is used in a non-interactive backend for web apps
                    plt.switch_backend('Agg')
                    shap.initjs() # Initialize JS for SHAP plots (though not directly used for static image)

                    # Create a unique filename for the SHAP image
                    shap_image_filename = f"shap_force_plot_{uuid.uuid4().hex}.png"
                    shap_image_full_path = STATIC_DIR / shap_image_filename

                    # Generate the force plot and save it
                    # For single prediction, use shap.force_plot
                    # Ensure base_values is correctly extracted from explainer
                    base_value = explainer.expected_value[1] if isinstance(explainer.expected_value, list) else explainer.expected_value
                    shap.force_plot(base_value, shap_values[0], df.iloc[0], matplotlib=True, show=False).savefig(shap_image_full_path, bbox_inches='tight')
                    plt.close() # Close the plot to free memory

                    result["shap_image"] = str(shap_image_full_path) # Store full path for PDF generation
                    shap_image_path = str(shap_image_full_path) # Store for template rendering

                except Exception as shap_e:
                    flash(f"Error generating SHAP explanation: {shap_e}", "warning")
                    app.logger.error(f"SHAP error: {shap_e}", exc_info=True)
            else:
                flash("Model pipeline does not contain 'scaler' and 'classifier' steps for SHAP explanation. SHAP plot will not be generated.", "warning")


            # Generate PDF report with SHAP explanation
            pdf_path = REPORT_DIR / f"claim_report_{uuid.uuid4().hex}.pdf"
            generate_claim_pdf(str(pdf_path), result, new_claim_data=df, shap_values=shap_values[0] if shap_values is not None else None, feature_names=feature_names)

            # Pass only file name to template for download
            return render_template(
                "results.html",
                mode="claim",
                result=result,
                report_file=pdf_path.name,
                # Pass the relative path for the HTML template to display the image
                # The template expects a path relative to 'static/'
                shap_image=f"static/shap/{shap_image_filename}" if shap_image_path else None
            )

        except Exception as e:
            flash(f"Error analyzing single claim: {e}", "error")
            # Log the full traceback for debugging
            import traceback
            app.logger.error(f"Error in single_claim: {e}\n{traceback.format_exc()}")
            return redirect(url_for("dashboard", tab='claim')) # Redirect back to the form with error

    # If GET request, just render the form
    return render_template("single_claim.html") # This template is no longer directly used, dashboard.html handles it


@app.route("/predict-provider", methods=["POST"])
def predict_provider():
    if not logged_in():
        flash("Please log in to predict provider fraud.", "info")
        return redirect(url_for("login"))

    if rf_bal is None or not feature_columns:
        flash("Provider-level model or features not found. Please ensure 'rf_bal.pkl' and 'provider_features.pkl' are in the 'models' directory.", "error")
        return redirect(url_for("dashboard", tab='provider'))

    files = {k: request.files.get(k) for k in ["inpatient","outpatient","beneficiary"]}
    missing = [k for k,v in files.items() if v is None or v.filename==""]
    if missing:
        flash(f"Missing files: {', '.join(missing)}. Please upload all required CSVs.", "error")
        return redirect(url_for("dashboard", tab='provider'))

    saved_paths={}
    for key,f in files.items():
        if not allowed_file(f.filename):
            flash(f"File for {key} must be a .csv file. Please check the file type.", "error")
            return redirect(url_for("dashboard", tab='provider'))
        fname=secure_filename(f.filename)
        path=UPLOAD_DIR/f"{key}_{uuid.uuid4().hex}_{fname}"
        f.save(path)
        saved_paths[key]=path

    try:
        out, claims_fe, provider_features = apply_pipeline_to_test(
            path_in=saved_paths["inpatient"],
            path_out=saved_paths["outpatient"],
            path_bene=saved_paths["beneficiary"],
            rf_model=rf_bal,
            feature_columns=feature_columns
        )
        # Store fraud providers to DB
        store_fraud_to_db(
            df=out,
            host="localhost", # Replace with your MySQL host
            user="root",      # Replace with your MySQL user
            password="chudar27@", # Replace with your MySQL password
            database="FraudProviders" # Replace with your MySQL database name
        )
        flash("Provider analysis complete! Results are ready.", "success")

    except Exception as e:
        flash(f"Error during provider analysis: {e}. Please check your input files and try again.", "error")
        import traceback
        app.logger.error(f"Provider pipeline error: {e}\n{traceback.format_exc()}")
        return redirect(url_for("dashboard", tab='provider'))

    report_path = REPORT_DIR/f"provider_predictions_{uuid.uuid4().hex}.csv"
    out.to_csv(report_path, index=False)

    # =========================
    # SHAP Visualization for Provider (Global Plot)
    # =========================
    # The custom_plot.png is assumed to be pre-generated and placed in static/images.
    # If you want to generate a global SHAP plot dynamically here, you would need
    # to fit an explainer on the provider_features and generate the plot.
    # For now, we assume it's a static image.
    custom_image_path = BASE_DIR / "static/images/custom_plot.png"
    if custom_image_path.exists():
        shap_path_for_template = "images/custom_plot.png" # Relative path for HTML
    else:
        shap_path_for_template = None
        flash("Global feature importance plot (custom_plot.png) not found in static/images.", "warning")


    preview = out.head(50).to_dict(orient="records")
    return render_template(
        "results.html",
        mode="provider",
        table_preview=preview,
        shap_image=shap_path_for_template, # Pass the relative path
        report_file=report_path.name
    )

@app.route("/logout")
def logout():
    session.pop("user", None)
    flash("You have been logged out.", "info") # Added flash message
    return redirect(url_for("login"))

@app.route("/download/<fname>")
def download_report(fname):
    # Ensure the file is in the REPORT_DIR for security
    return send_from_directory(REPORT_DIR, fname, as_attachment=True)

extracted_report_text = ""
uploaded_file_name = "" 

@app.route("/chatbot")
def chatbot():
    if not logged_in():
        flash("Please log in to use the chatbot.", "info")
        return redirect(url_for("login"))
    return render_template("chatbot.html")
@app.route('/upload', methods=['POST'])
def upload_pdf():
    """
    Endpoint to handle PDF file uploads, extract text, and store it.
    This function is called by the frontend's drag-and-drop mechanism.
    """
    global extracted_report_text
    global uploaded_file_name
    
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
        
    if file and file.filename.endswith('.pdf'):
        try:
            # Read the PDF file in-memory
            pdf_reader = pypdf.PdfReader(BytesIO(file.read()))
            full_text = ""
            for page in pdf_reader.pages:
                full_text += page.extract_text() or ""
            
            if not full_text:
                return jsonify({"error": "Could not extract text from PDF. The file may be an image."}), 400

            extracted_report_text = full_text
            uploaded_file_name = file.filename
            
            return jsonify({"message": f"Successfully uploaded and processed {file.filename}"})

        except Exception as e:
            print(f"Error processing PDF: {e}")
            return jsonify({"error": "Failed to process PDF file. Please check file integrity."}), 500

    return jsonify({"error": "Invalid file type. Only PDF files are accepted."}), 400

@app.route('/chat', methods=['POST'])
def chat():
    """
    Endpoint to handle the chatbot's requests.
    This function is called when a user sends a message.
    """
    global extracted_report_text

    try:
        data = request.get_json()
        user_message = data.get('userMessage', '').strip()

        if not user_message:
            return jsonify({"error": "Missing user message"}), 400
        
        # Conditionally build the prompt based on if a report is uploaded
        if extracted_report_text:
            prompt = f"""
            Answer the user's question. Use the provided context to answer the question, but if the context is not relevant, use your general knowledge. Do not mention the context or your sources of information. Just provide a direct, concise answer.
            
            --- Context ---
            {extracted_report_text}
            --- End of Context ---

            User's question: {user_message}
            """
        else:
            # If no report is uploaded, use a more general prompt
            prompt = f"""
            Answer the user's question using your general knowledge. Be helpful and forward-looking, especially for questions about future or speculative events. **Do not ever respond with "I cannot provide information on..." or similar phrases.** Instead, provide a thoughtful, reasoned answer based on general principles or trends. Do not mention the provided context or your sources of information. Just provide a direct, concise answer.
            
            User's question: {user_message}
            """

        # Ollama API details
        ollama_api_url = "http://localhost:11434/api/generate"
        ollama_payload = {
            "model": "llama3", # You can change this to a different model you have pulled
            "prompt": prompt,
            "stream": False # Set to True for streaming responses
        }

        # Make the API call with exponential backoff
        for i in range(3): # Retry up to 3 times
            try:
                response = requests.post(ollama_api_url, headers={'Content-Type': 'application/json'}, json=ollama_payload)
                response.raise_for_status() # Raises an HTTPError for bad responses (4xx or 5xx)

                # Assuming stream is False, parse the single JSON object
                result = response.json()
                bot_response = result.get('response', "An error occurred with the Ollama response.")
                
                return jsonify({"response": bot_response})
            except requests.exceptions.RequestException as e:
                print(f"Request failed: {e}. Retrying...")
                time.sleep(2 ** i) # Exponential backoff
        
        return jsonify({"error": "Failed to connect to Ollama after multiple retries"}), 500

    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return jsonify({"error": "An unexpected server error occurred."}), 500

@app.route('/clear-report', methods=['POST'])
def clear_report():
    """
    Endpoint to clear the extracted report text from the global variable.
    This is called when the user clicks the 'remove' button.
    """
    global extracted_report_text
    global uploaded_file_name
    extracted_report_text = ""
    uploaded_file_name = ""
    return jsonify({"message": "Report cleared successfully."})

# =========================
# MAIN
# =========================
if __name__=="__main__":
    # For development, debug=True is fine. For production, set debug=False
    # and use a production-ready WSGI server like Gunicorn or uWSGI.
    app.run(debug=True)

