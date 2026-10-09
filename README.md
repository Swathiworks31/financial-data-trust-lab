# Financial Data Trust Lab

**Independent financial data science portfolio project · synthetic data only**

A compact demonstration of a real financial analytics workflow: detect unreliable records, reconcile against verified records, compare a statistical forecast to a naive baseline, and retrieve evidence for a cautious AI explanation.

### Live-demo features

1. **Data reliability:** intentionally corrupted incoming ledger, transparent validation rules (missing fields, duplicate periods, implausible expenses, debt > assets), and a correction audit trail. Reconciliation uses a second **synthetic reference ledger**, not speculative imputation.
2. **Predictive experiment:** one-month-ahead operating expense forecast with a time-ordered train/test split. Ridge regression is compared with a persistence baseline using MAE. Actual results are displayed, even if the baseline wins.
3. **Evidence assistant:** TF-IDF retrieves verified synthetic financial notes with inspectable source IDs. A real LLM explanation is available if the owner securely configures `OPENAI_API_KEY`; otherwise retrieval still works. The model is instructed to disclose unknown causes and to avoid investment advice.

This is a **working portfolio prototype**, not production-ready financial software. It does not use data from Morgan Stanley, Vanguard, or any financial client. It is not a compliance product or a forecasting system suitable for actual financial decisions.

### Run locally

```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

### Publish for recruiters

1. Create a **public** repository on [GitHub](https://github.com/new) named `financial-data-trust-lab`.
2. Upload **`app.py`, `core.py`, `requirements.txt`, `README.md`, and `.streamlit/config.toml`**. If uploading by browser, extract the ZIP first; GitHub does not automatically unpack it.
3. Open [Streamlit Community Cloud](https://share.streamlit.io), select **Create app**, choose your repository/branch and entry file `app.py`, and deploy.
4. Share the generated `*.streamlit.app` URL together with the GitHub repository link.

**Do not commit your API key.** If you want to activate the LLM, configure `OPENAI_API_KEY` in the app's Streamlit Secrets, not inside code. A public LLM feature without server-side quotas and authentication can create API bills; keep it disabled until you have those protections or are comfortable managing restricted demo access.

### Testing

```bash
python -m pip install pytest
pytest -q
```

### Tech stack

Python, Pandas, NumPy, Scikit-learn, Streamlit, Plotly, TF-IDF retrieval, optional OpenAI Responses API. The synthetic ledger uses a canonical monthly financial schema and keeps validation/model logic separate from the UI.

### Recommended follow-up improvements

- Replace synthetic figures with properly licensed public financial records and a documented metric dictionary.
- Add SQL queries (DuckDB or Postgres) and reproducible ETL jobs.
- Add statistically meaningful model backtests, missing data handling, concept drift checks, and more robust validation.
- Evaluate generated AI responses with a fixed question set, groundedness scoring, access controls, and human review workflow.
- Add CI, security controls, logging, API budgets, and monitoring before calling the app production-ready.
