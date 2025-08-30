# Churn Chatbot — Quick Guide (Marketing)

You can ask questions in plain English (no tech terms needed).  
Open the API docs at: http://127.0.0.1:8000/docs → **POST /chat_llm → Try it out**.

### Examples you can paste
1. male customer, 12 months tenure, pays 70 monthly, total charges 845, fiber optic internet, month-to-month contract, electronic check, no online security

2. female, DSL, one year, bank transfer, tenure 6 months, monthly 55, has tech support
3. High risk demo: senior citizen yes, fiber optic, month-to-month, electronic check, tenure 1, pays 95, no online security, no backup, no tech support


### What you get back
- **Churn probability** (e.g., 0.31 → 31%)
- **Prediction**: “Churn” or “No Churn”
- A friendly **message** summarizing the result

### Tips
- If you don’t mention something, the bot fills sensible defaults (Yearly contract → lower churn, etc.)
- You can change sensitivity using `threshold` (e.g., 0.3 catches more potential churners).

