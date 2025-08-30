# Telco Churn — API + Open-Source Chatbot

This project provides:
- A trained churn classifier (sklearn)
- A FastAPI service with:
  - `/predict` for structured JSON
  - `/chat_llm` using an **open-source** HF model (FLAN-T5 Small) to parse natural language → structured JSON → predict
- Artifacts: `model/churn_model.pkl`, `model/scaler.pkl`, `model/features.json`

## Project Layout
    telco_churn_project/
        ├─ app/
        │  ├─ main.py
        │  └─ model/
        │     ├─ churn_model.pkl
        │     ├─ scaler.pkl
        │     └─ features.json
        ├─ data/
        │  └─ WA_Fn-UseC_-Telco-Customer-Churn.csv
        ├─ docs/
        │  ├─ README.md
        │  ├─ MARKETING_GUIDE.md
        │  ├─ ARCHITECTURE.md
        │  └─ REPORT.md
        ├─ notebooks/
        │  └─ churn_model.ipynb
        ├─ venv/
        ├─ requirements.txt


## Quickstart
```bash
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

## Endpoints
- /predict 
  - example: {
  "tenure": 12,
  "MonthlyCharges": 70.5,
  "TotalCharges": 845.0,
  "Gender": "Male",
  "Senior_Citizen": 0,
  "Is_Married": "No",
  "Dependents": "No",
  "Phone_Service": "Yes",
  "Dual": "No",
  "Internet_Service": "Fiber optic",
  "Online_Security": "No",
  "Online_Backup": "No",
  "Device_Protection": "No",
  "Tech_Support": "No",
  "Streaming_TV": "Yes",
  "Streaming_Movies": "Yes",
  "Contract": "Month-to-month",
  "Paperless_Billing": "Yes",
  "Payment_Method": "Electronic check"
}

- /chat_llm
    - example: {
  "text": "male, fiber optic, month-to-month, electronic check, tenure 12, pays 70, total 845, no online security",
  "threshold": 0.5
}

## How To Test
- Postman

Public workspace: https://www.postman.com/medobaker60-2384421/workspace/churn-predictor-chatbot/request/48042902-db9beb92-feed-422f-83f0-f12409eef05a?action=share&creator=48042902&ctx=documentation

- import OpenAPI directly: http://127.0.0.1:8000/openapi.json → Postman → “Import”.

### - Business rule

If tenure == 0 and Total_Charges is missing, we set Total_Charges = 0 during preprocessing (and in the chatbot pipeline).

Troubleshooting

“Attribute app not found”: run from repo root → uvicorn app.main:app ...

LLM JSON errors: endpoint is hardened; if model outputs non-JSON, we fallback to defaults instead of crashing.

Mismatched columns: make sure you’re using the features.json saved by the notebook that trained the deployed model.
