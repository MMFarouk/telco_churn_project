from fastapi import FastAPI, Query
from pydantic import BaseModel
import joblib, json, pandas as pd, re
from pathlib import Path

APP_DIR = Path(r"C:\Users\mfarouk\Desktop\telco_churn_project\app")
MODEL_DIR = APP_DIR / "model"
MODEL_PATH   = MODEL_DIR / "churn_model.pkl"
SCALER_PATH  = MODEL_DIR / "scaler.pkl"
FEATS_PATH   = MODEL_DIR / "features.json"

# 1) Load ML artifacts
model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
with open(FEATS_PATH, "r") as f:
    feature_list = json.load(f)

NUMERIC_COLS = ["tenure", "Monthly_Charges", "Total_Charges", "Senior_Citizen"]


# 2) FastAPI
app = FastAPI(
    title="Telco Churn API",
    description="Open-source LLM pipeline: NL → JSON → Predict",
    version="2.0"
)

@app.get("/")
def root():
    return {"message": "Churn Prediction API is running!"}

@app.get("/healthz")
def healthz():
    return {
        "status": "ok",
        "model_loaded": MODEL_PATH.exists(),
        "scaler_loaded": SCALER_PATH.exists(),
        "features_loaded": FEATS_PATH.exists(),
        "n_features": len(feature_list),
    }


# 3) /predict
class CustomerData(BaseModel):
    tenure: float
    Monthly_Charges: float | None = None
    Total_Charges: float | None = None
    MonthlyCharges: float | None = None
    TotalCharges: float | None = None

    Gender: str
    Senior_Citizen: int
    Is_Married: str
    Dependents: str
    Phone_Service: str
    Dual: str
    Internet_Service: str
    Online_Security: str
    Online_Backup: str
    Device_Protection: str
    Tech_Support: str
    Streaming_TV: str
    Streaming_Movies: str
    Contract: str
    Paperless_Billing: str
    Payment_Method: str

def _predict_from_record(record: dict, threshold: float = 0.5):
    df = pd.DataFrame([record])
    # one-hot categoricals (drop_first=True) as in training
    df_cat = df.drop(columns=NUMERIC_COLS, errors="ignore")
    df_cat_enc = pd.get_dummies(df_cat, drop_first=True)
    df_enc = pd.concat([df[NUMERIC_COLS], df_cat_enc], axis=1)
    # align columns
    df_enc = df_enc.reindex(columns=feature_list, fill_value=0)
    # scale + predict
    X = scaler.transform(df_enc)
    proba = model.predict_proba(X)[0, 1]
    label = "Churn" if proba >= threshold else "No Churn"
    print(f"\n[PREDICT] prob={round(float(proba),4)} label={label} thr={threshold}\n")
    return {"churn_probability": round(float(proba), 4), "prediction": label}

@app.post("/predict")
def predict(data: CustomerData, threshold: float = Query(0.5, ge=0.0, le=1.0)):
    try:
        rec = data.dict()
        # normalize numeric key variants
        if rec.get("Monthly_Charges") is None and rec.get("MonthlyCharges") is not None:
            rec["Monthly_Charges"] = rec.pop("MonthlyCharges")
        if rec.get("Total_Charges") is None and rec.get("TotalCharges") is not None:
            rec["Total_Charges"] = rec.pop("TotalCharges")
        out = _predict_from_record(rec, threshold=threshold)
        out["threshold"] = threshold
        return out
    except Exception as e:
        print("ERROR /predict:", e)
        return {"error": str(e)}

# 4) LLM: Hugging Face will use: google/flan-t5-small
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

LLM_NAME = "google/flan-t5-small"
_tokenizer = AutoTokenizer.from_pretrained(LLM_NAME)
_llm = AutoModelForSeq2SeqLM.from_pretrained(LLM_NAME)

SCHEMA_KEYS = [
    "Gender","Senior_Citizen","Is_Married","Dependents","Phone_Service","Dual",
    "Internet_Service","Online_Security","Online_Backup","Device_Protection",
    "Tech_Support","Streaming_TV","Streaming_Movies","Contract",
    "Paperless_Billing","Payment_Method",
    "tenure","Monthly_Charges","Total_Charges"
]

CATS_ALLOWED = {
    "Gender": ["Male","Female"],
    "Is_Married": ["Yes","No"],
    "Dependents": ["Yes","No"],
    "Phone_Service": ["Yes","No"],
    "Dual": ["Yes","No","No phone service"],
    "Internet_Service": ["DSL","Fiber optic","No"],
    "Online_Security": ["Yes","No","No internet service"],
    "Online_Backup": ["Yes","No","No internet service"],
    "Device_Protection": ["Yes","No","No internet service"],
    "Tech_Support": ["Yes","No","No internet service"],
    "Streaming_TV": ["Yes","No","No internet service"],
    "Streaming_Movies": ["Yes","No","No internet service"],
    "Contract": ["Month-to-month","One year","Two year"],
    "Paperless_Billing": ["Yes","No"],
    "Payment_Method": ["Electronic check","Mailed check","Bank transfer (automatic)","Credit card (automatic)"],
}

SYSTEM_PROMPT = """You extract structured data for a churn model.
Return ONLY a valid JSON with these keys:

- Gender: "Male" | "Female"
- Senior_Citizen: 0 | 1
- Is_Married: "Yes" | "No"
- Dependents: "Yes" | "No"
- Phone_Service: "Yes" | "No"
- Dual: "Yes" | "No" | "No phone service"
- Internet_Service: "DSL" | "Fiber optic" | "No"
- Online_Security: "Yes" | "No" | "No internet service"
- Online_Backup: "Yes" | "No" | "No internet service"
- Device_Protection: "Yes" | "No" | "No internet service"
- Tech_Support: "Yes" | "No" | "No internet service"
- Streaming_TV: "Yes" | "No" | "No internet service"
- Streaming_Movies: "Yes" | "No" | "No internet service"
- Contract: "Month-to-month" | "One year" | "Two year"
- Paperless_Billing: "Yes" | "No"
- Payment_Method: "Electronic check" | "Mailed check" | "Bank transfer (automatic)" | "Credit card (automatic)"
- tenure: number
- Monthly_Charges: number
- Total_Charges: number

Rules:
- If a field is missing, infer a sensible default.
- If tenure == 0 and Total_Charges is missing, set Total_Charges = 0.
- Output JSON only, no extra text.
"""

def llm_extract(message: str) -> dict:
    """Run FLAN-T5, try to extract strict JSON, validate type, coerce & default."""
    prompt = SYSTEM_PROMPT + "\n\nUser message:\n" + message.strip()
    inputs = _tokenizer(prompt, return_tensors="pt", truncation=True)
    outputs = _llm.generate(**inputs, max_new_tokens=256)
    text = _tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

    # ---- try to isolate JSON ----
    # remove code fences
    text = text.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    # if there's a {...} block, keep the first one
    m = re.search(r"\{[\s\S]*\}", text)
    if m:
        text = m.group(0)

    # ---- parse safely ----
    data = {}
    try:
        parsed = json.loads(text)
        # accept only dicts; anything else (list/int/str) → fallback to {}
        if isinstance(parsed, dict):
            data = parsed
        else:
            print(f"[LLM PARSE] Non-dict JSON returned ({type(parsed).__name__}); using empty dict fallback.")
    except Exception as ex:
        print(f"[LLM PARSE] JSON decode failed: {ex}; using empty dict fallback.")

    # ---- build record with coercion/defaults ----
    record = {}
    for k in SCHEMA_KEYS:
        v = data.get(k, None)

        if k in ["tenure", "Monthly_Charges", "Total_Charges"]:
            try:
                v = float(v)
            except Exception:
                v = None
        elif k == "Senior_Citizen":
            try:
                v = 1 if int(v) == 1 else 0
            except Exception:
                v = 0
        else:
            if isinstance(v, str):
                v = v.strip()
            if k in CATS_ALLOWED and isinstance(v, str):
                low = v.lower()
                synonyms = {
                    "month to month": "Month-to-month",
                    "one-year": "One year",
                    "two-year": "Two year",
                    "bank transfer": "Bank transfer (automatic)",
                    "credit card": "Credit card (automatic)",
                    "no internet": "No",
                    "no internet service": "No internet service",
                    "no phone": "No phone service",
                }
                if low in synonyms:
                    v = synonyms[low]
                for allowed in CATS_ALLOWED[k]:
                    if low == allowed.lower():
                        v = allowed
        record[k] = v

    # business rule
    if (record.get("tenure") == 0) and (record.get("Total_Charges") is None):
        record["Total_Charges"] = 0.0

    # defaults
    defaults = {
        "Gender":"Male","Is_Married":"No","Dependents":"No","Phone_Service":"Yes","Dual":"No",
        "Internet_Service":"Fiber optic","Online_Security":"No","Online_Backup":"No","Device_Protection":"No",
        "Tech_Support":"No","Streaming_TV":"No","Streaming_Movies":"No","Contract":"Month-to-month",
        "Paperless_Billing":"Yes","Payment_Method":"Electronic check",
        "tenure":12.0,"Monthly_Charges":70.0,"Total_Charges":800.0,"Senior_Citizen":0
    }
    for k,v in defaults.items():
        if record.get(k) in [None, ""]:
            record[k] = v

    # final numeric coercion (defensive)
    for c in NUMERIC_COLS:
        try:
            record[c] = float(record[c])
        except Exception:
            record[c] = defaults[c]

    print("\n[LLM PARSE] record built:")
    for k in SCHEMA_KEYS:
        print(f"  - {k}: {record[k]}")
    print()
    return record


# 5) LLM Chat endpoint
class ChatLLMRequest(BaseModel):
    text: str
    threshold: float | None = 0.5

@app.post("/chat_llm")
def chat_llm(req: ChatLLMRequest):
    try:
        thr = float(req.threshold if req.threshold is not None else 0.5)
        record = llm_extract(req.text)
        result = _predict_from_record(record, threshold=thr)
        msg = f"Estimated churn probability is {result['churn_probability']*100:.1f}% → {result['prediction']} (threshold={thr})."
        return {
            "parsed_record": record,
            "churn_probability": result["churn_probability"],
            "prediction": result["prediction"],
            "threshold": thr,
            "message": msg
        }
    except Exception as e:
        print("ERROR /chat_llm:", e)
        return {"error": str(e)}
