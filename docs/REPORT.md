# Telco Churn — Technical & Business Report

## 1. Business Goal
Predict the probability that a customer will churn so marketing can target retention offers. The tool must be simple to use and run locally with **open-source** components.

## 2. Data
- Source: WA_Fn-UseC_-Telco-Customer-Churn.csv (7,043 rows, 21 columns).
- Target: `Churn` (Yes/No).
- Business rule for missing `Total_Charges`: if `tenure == 0` and `Total_Charges` is missing → set `Total_Charges = 0`.

## 3. Preprocessing
- Dropped `customerID`.
- Numeric: `tenure`, `Monthly_Charges`, `Total_Charges`, `Senior_Citizen`.
- Categorical: one-hot encoding (`drop_first=True`), columns aligned with `features.json`.
- Scaling: `StandardScaler` on the encoded matrix.
- Imbalance: ~26.6% churn (handled via metrics; optional threshold tuning).

## 4. Models & Selection
Trained classic models (LogReg, RandomForest, GradientBoosting, SVC, KNN).  
Selected by **ROC-AUC** (tie-break **PR-AUC**).  
Saved `model.pkl`, `scaler.pkl`, `features.json` for reproducible serving.

## 5. Serving Architecture
- FastAPI with two endpoints:
  - `/predict`: accepts structured JSON (direct integration).
  - `/chat_llm`: uses **Hugging Face FLAN-T5 Small** (open-source) to parse natural language → strict JSON → validate/coerce → predict.
- The serving pipeline guarantees the request is aligned to training columns before scaling and prediction.

## 6. Why this approach
- **Open-source LLM** satisfies the client’s no closed-source / third-party requirement.
- **Local inference**: runs offline, no data leaves the machine.
- **Simplicity**: easy to test via Swagger/Postman; low friction for marketing.

## 7. Threshold & Interpretation
- Default decision threshold = **0.5** (can be changed via query parameter).
- Lower threshold (e.g., 0.3) increases recall (catch more churners) but may reduce precision.
- Provide probability alongside label so business teams can decide action bands (e.g., >0.6 = retention offer A).

## 8. Validation & Monitoring (next steps)
- Add calibration check (reliability curve) to ensure probabilities are well-scaled.
- Track post-deployment metrics: acceptance rate, precision/recall, lift in retention campaigns.
- Consider cost-sensitive threshold setting based on campaign ROI.

## 9. Security & Privacy
- All artifacts and inference are local; no external API calls required.
- Keep the Postman workspace public only if it contains no secret URLs or credentials.

## 10. Limitations & Future Work
- LLM slot filling is small (FLAN-T5 Small) for CPU; accuracy can improve with `flan-t5-base`.
- Could add guided forms for marketing to reduce typos.
- Explore SHAP for top feature drivers and add explanations to responses.
