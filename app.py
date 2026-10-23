"""Flask web app: Dual-Engine Fake News & Fact-Checking System.

Features:
1. AI Fact-Checking (Priority 1: Anthropic Claude, Priority 2: Groq).
2. Machine Learning Classifier (TF-IDF + Passive-Aggressive / Logistic Regression).
3. Live Web Cross-Referencing (DuckDuckGo search).
4. SQLite Persistence (checks.db) with historical auditing.
"""

import json
import math
import os
import sqlite3
import uuid
from datetime import datetime
from pathlib import Path

try:
    import joblib
except ImportError:
    joblib = None

try:
    import numpy as np
except ImportError:
    np = None

from flask import Flask, abort, g, redirect, render_template, request, url_for  # type: ignore[import-untyped, import-not-found]

from utils.ai_service import fact_check_with_ai
from utils.clean_text import preprocess
from utils.web_search import build_query, search_related

ROOT_DIR = Path(__file__).resolve().parent
MODEL_DIR = ROOT_DIR / "model"

# On Vercel / AWS Lambda serverless runtimes, only /tmp is writable
IS_VERCEL = bool(os.environ.get("VERCEL"))
DB_PATH = Path("/tmp/checks.db") if IS_VERCEL else ROOT_DIR / "checks.db"

SHORT_TEXT_WORD_THRESHOLD = 20
SNIPPET_LENGTH = 300
RELATED_RESULTS_POOL_SIZE = 15
RELATED_RESULTS_PAGE_SIZE = 5

app = Flask(__name__)
app.secret_key = "dev-secret-key-kz-ml-02"
handler = app


def init_db(conn=None):
    close_after = False
    if conn is None:
        try:
            conn = sqlite3.connect(DB_PATH)
            close_after = True
        except Exception as exc:
            print(f"Warning: Cannot connect to DB at {DB_PATH}: {exc}")
            return

    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS checks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                text_snippet TEXT NOT NULL,
                prediction TEXT NOT NULL,
                confidence REAL NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        existing_cols = {row[1] for row in conn.execute("PRAGMA table_info(checks)")}
        for column, col_type in (
            ("search_query", "TEXT"),
            ("word_count", "INTEGER"),
            ("related_json", "TEXT"),
            ("ai_verdict", "TEXT"),
            ("ai_confidence", "REAL"),
            ("ai_summary", "TEXT"),
            ("ai_provider", "TEXT"),
            ("ai_details_json", "TEXT"),
            ("ml_prediction", "TEXT"),
            ("ml_confidence", "REAL"),
            ("device_id", "TEXT"),
        ):
            if column not in existing_cols:
                conn.execute(f"ALTER TABLE checks ADD COLUMN {column} {col_type}")
        conn.commit()
    except Exception as exc:
        print(f"Warning: Failed to initialize SQLite database: {exc}")
    finally:
        if close_after:
            try:
                conn.close()
            except Exception:
                pass


def get_db():
    if "db" not in g:
        try:
            g.db = sqlite3.connect(DB_PATH)
            g.db.row_factory = sqlite3.Row
        except Exception:
            try:
                g.db = sqlite3.connect("/tmp/checks.db")
                g.db.row_factory = sqlite3.Row
            except Exception:
                g.db = sqlite3.connect(":memory:")
                g.db.row_factory = sqlite3.Row
                init_db(g.db)
    return g.db


def load_model():
    if joblib is None:
        return None, None
    model_path = MODEL_DIR / "model.pkl"
    vectorizer_path = MODEL_DIR / "tfidf.pkl"
    if not model_path.exists() or not vectorizer_path.exists():
        return None, None
    try:
        return joblib.load(model_path), joblib.load(vectorizer_path)
    except Exception as exc:
        print(f"Warning: Could not load ML model: {exc}")
        return None, None


model, vectorizer = load_model()
model_name = (
    (MODEL_DIR / "model_name.txt").read_text(encoding="utf-8").strip()
    if (MODEL_DIR / "model_name.txt").exists()
    else "Passive Aggressive"
)


def predict_ml(raw_text: str):
    """Clean, vectorise and classify raw_text with local TF-IDF model."""
    if model is None or vectorizer is None or np is None:
        return "UNKNOWN", 0.0

    try:
        cleaned = preprocess(raw_text)
        vec = vectorizer.transform([cleaned])
        pred = model.predict(vec)[0]

        if hasattr(model, "decision_function"):
            score = model.decision_function(vec)[0]
            confidence = 1 / (1 + np.exp(-abs(score)))
        else:
            confidence = model.predict_proba(vec)[0].max()

        label = "REAL" if pred == 1 else "FAKE"
        return label, round(confidence * 100, 2)
    except Exception as exc:
        print(f"Warning: ML prediction failed: {exc}")
        return "UNKNOWN", 0.0


@app.route("/", methods=["GET"])
def index():
    model_missing = model is None or vectorizer is None
    return render_template(
        "index.html",
        model_missing=model_missing,
        model_name=model_name,
    )


@app.route("/check", methods=["POST"])
def check():
    headline = request.form.get("headline", "").strip()
    article = request.form.get("article", "").strip()
    full_text = f"{headline} {article}".strip()

    if not full_text:
        return render_template(
            "index.html",
            model_missing=model is None,
            model_name=model_name,
            error="Please enter a headline or article text to verify.",
        )

    word_count = len(full_text.split())
    snippet = full_text[:SNIPPET_LENGTH]
    created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # 1. Traditional ML Model Prediction
    ml_label, ml_confidence = predict_ml(full_text)

    # 2. AI Fact-Checking (Priority 1: Anthropic, Priority 2: Groq)
    ai_result = fact_check_with_ai(full_text)
    ai_verdict = ai_result.get("verdict")
    ai_confidence = ai_result.get("confidence", 0.0)
    ai_summary = ai_result.get("summary", "")
    ai_provider = ai_result.get("provider", "AI Engine")
    ai_details = ai_result.get("details", [])

    # Final verdict defaults to AI if available (factual truth), otherwise ML stylistic prediction
    if ai_verdict in ("REAL", "FAKE"):
        final_prediction = ai_verdict
        final_confidence = ai_confidence
    else:
        final_prediction = ml_label
        final_confidence = ml_confidence

    # 3. Web Search Citations
    search_query = build_query(headline, article)
    related_results = search_related(search_query, max_results=RELATED_RESULTS_POOL_SIZE)

    # 4. Save to Database with device_id
    db = get_db()
    cursor = db.execute(
        """
        INSERT INTO checks (
            text_snippet, prediction, confidence, created_at,
            search_query, word_count, related_json,
            ai_verdict, ai_confidence, ai_summary, ai_provider, ai_details_json,
            ml_prediction, ml_confidence, device_id
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            snippet,
            final_prediction,
            final_confidence,
            created_at,
            search_query,
            word_count,
            json.dumps(related_results),
            ai_verdict,
            ai_confidence,
            ai_summary,
            ai_provider,
            json.dumps(ai_details),
            ml_label,
            ml_confidence,
            g.device_id,
        ),
    )
    db.commit()

    return redirect(url_for("result", check_id=cursor.lastrowid))


@app.route("/result/<int:check_id>", methods=["GET"])
def result(check_id):
    db = get_db()
    check_row = db.execute("SELECT * FROM checks WHERE id = ?", (check_id,)).fetchone()
    if check_row is None:
        abort(404)

    related_results = json.loads(check_row["related_json"] or "[]")
    total_pages = max(1, math.ceil(len(related_results) / RELATED_RESULTS_PAGE_SIZE))
    page = max(1, min(request.args.get("page", 1, type=int), total_pages))
    start = (page - 1) * RELATED_RESULTS_PAGE_SIZE
    page_results = related_results[start : start + RELATED_RESULTS_PAGE_SIZE]

    word_count = check_row["word_count"]
    short_warning = (
        f"Input is only {word_count} words — brief claims benefit strongly from the AI fact-check below."
        if word_count is not None and word_count < SHORT_TEXT_WORD_THRESHOLD
        else None
    )

    ai_details = json.loads(check_row["ai_details_json"] or "[]")

    return render_template(
        "result.html",
        label=check_row["prediction"],
        confidence=check_row["confidence"],
        snippet=check_row["text_snippet"],
        short_warning=short_warning,
        model_name=model_name,
        search_query=check_row["search_query"],
        related_results=page_results,
        page=page,
        total_pages=total_pages,
        check_id=check_id,
        ai_verdict=check_row["ai_verdict"],
        ai_confidence=check_row["ai_confidence"],
        ai_summary=check_row["ai_summary"],
        ai_provider=check_row["ai_provider"],
        ai_details=ai_details,
        ml_prediction=check_row["ml_prediction"],
        ml_confidence=check_row["ml_confidence"],
        created_at=check_row["created_at"],
    )


@app.route("/history", methods=["GET"])
def history():
    db = get_db()
    device_id = g.device_id

    rows = db.execute(
        "SELECT * FROM checks WHERE device_id = ? ORDER BY id DESC LIMIT 100",
        (device_id,),
    ).fetchall()

    # If this is the developer's first time with device tracking, associate any previous unassigned checks
    if not rows:
        unassigned_count = db.execute("SELECT COUNT(*) FROM checks WHERE device_id IS NULL").fetchone()[0]
        if unassigned_count > 0:
            db.execute("UPDATE checks SET device_id = ? WHERE device_id IS NULL", (device_id,))
            db.commit()
            rows = db.execute(
                "SELECT * FROM checks WHERE device_id = ? ORDER BY id DESC LIMIT 100",
                (device_id,),
            ).fetchall()

    return render_template("history.html", checks=rows, device_id=device_id)


@app.route("/history/clear", methods=["POST"])
def clear_history():
    db = get_db()
    device_id = g.device_id
    db.execute("DELETE FROM checks WHERE device_id = ?", (device_id,))
    db.commit()
    return redirect(url_for("history"))


if __name__ == "__main__":
    init_db()
    app.run(debug=True, port=5000)
else:
    init_db()
