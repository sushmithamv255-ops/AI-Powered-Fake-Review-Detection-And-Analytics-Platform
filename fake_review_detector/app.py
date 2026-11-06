import os
import io
from datetime import datetime
from functools import wraps
import pandas as pd
from flask import (Flask, render_template, request, redirect, url_for,session, flash, send_file, jsonify)
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
import database as db
from ml.detector import predict_review, predict_batch, get_model_accuracy
from utils.sentiment import analyze_sentiment, analyze_batch as sentiment_batch
from utils.duplicates import find_duplicates
from utils.spam_keywords import analyze_spam, analyze_batch as spam_batch
from utils.reviewer_analysis import analyze_reviewers
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
ALLOWED_EXTENSIONS = {"csv", "xlsx", "xls"}
MAX_UPLOAD_SIZE_MB = 100 * 1024
app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "dev-secret-key-change-in-production")
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_SIZE_MB * 1024 * 1024
os.makedirs(UPLOAD_DIR, exist_ok=True)
db.init_db()


@app.errorhandler(413)
def handle_413(error):
    flash(
        f"The uploaded file exceeds the {MAX_UPLOAD_SIZE_MB / 1024:g} GB limit. Please choose a smaller CSV or Excel file.",
        "danger",
    )
    return redirect(url_for("upload"))

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def login_required(view):
    @wraps(view)
    def wrapped(*args, **kwargs):
        if not session.get("user_id"):
            flash("Please log in to continue.", "warning")
            return redirect(url_for("login"))
        return view(*args, **kwargs)
    return wrapped

def admin_required(view):
    @wraps(view)
    def wrapped(*args, **kwargs):
        if not session.get("is_admin"):
            flash("Admin access required.", "danger")
            return redirect(url_for("dashboard"))
        return view(*args, **kwargs)
    return wrapped

def current_user_id():
    return session.get("user_id")

def read_dataset_file(filepath, filename):
    if filename.lower().endswith(".csv"):
        return pd.read_csv(filepath)
    return pd.read_excel(filepath)

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Map common alternate column names onto our expected schema."""
    rename_map = {}
    for col in df.columns:
        lc = col.strip().lower()
        if lc in ("review", "text", "comment", "review_text", "content"):
            rename_map[col] = "review_text"
        elif lc in ("reviewer", "reviewer_name", "user", "username", "author"):
            rename_map[col] = "reviewer_name"
        elif lc in ("rating", "stars", "score"):
            rename_map[col] = "rating"
    df = df.rename(columns=rename_map)
    return df

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/get-started")
def get_started():
    if session.get("user_id"):
        return redirect(url_for("dashboard"))
    return render_template("get_started.html")

@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        confirm = request.form.get("confirm_password", "")

        if not username or not email or not password:
            flash("All fields are required.", "danger")
            return redirect(url_for("register"))
        if password != confirm:
            flash("Passwords do not match.", "danger")
            return redirect(url_for("register"))
        if len(password) < 6:
            flash("Password must be at least 6 characters.", "danger")
            return redirect(url_for("register"))

        conn = db.get_db()
        existing = conn.execute(
            "SELECT id FROM users WHERE username = ? OR email = ?", (username, email)
        ).fetchone()
        if existing:
            conn.close()
            flash("Username or email already registered.", "danger")
            return redirect(url_for("register"))

        conn.execute(
            "INSERT INTO users (username, email, password_hash) VALUES (?, ?, ?)",
            (username, email, generate_password_hash(password)),
        )
        conn.commit()
        conn.close()
        flash("Registration successful. Please log in.", "success")
        return redirect(url_for("login"))

    return render_template("register.html")

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        identifier = request.form.get("username", "").strip()
        password = request.form.get("password", "")

        conn = db.get_db()
        user = conn.execute(
            "SELECT * FROM users WHERE username = ? OR email = ?",
            (identifier, identifier.lower()),
        ).fetchone()
        conn.close()
        if user and check_password_hash(user["password_hash"], password):
            session["user_id"] = user["id"]
            session["username"] = user["username"]
            session["is_admin"] = bool(user["is_admin"])
            db.log_activity(user["id"], "Login", f"{user['username']} logged in")
            flash(f"Welcome back, {user['username']}!", "success")
            return redirect(url_for("dashboard"))
        flash("Invalid username/email or password.", "danger")
        return redirect(url_for("login"))
    return render_template("login.html")

@app.route("/forgot-password", methods=["GET", "POST"])
def forgot_password():
    # Simplified reset flow (no email server): verify username + email match,
    # then allow setting a new password directly.
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        email = request.form.get("email", "").strip().lower()
        new_password = request.form.get("new_password", "")

        if len(new_password) < 6:
            flash("New password must be at least 6 characters.", "danger")
            return redirect(url_for("forgot_password"))

        conn = db.get_db()
        user = conn.execute(
            "SELECT * FROM users WHERE username = ? AND email = ?", (username, email)
        ).fetchone()
        if not user:
            conn.close()
            flash("No matching account found.", "danger")
            return redirect(url_for("forgot_password"))
        conn.execute(
            "UPDATE users SET password_hash = ? WHERE id = ?",
            (generate_password_hash(new_password), user["id"]),
        )
        conn.commit()
        conn.close()
        flash("Password updated. Please log in.", "success")
        return redirect(url_for("login"))

    return render_template("forgot_password.html")

@app.route("/logout")
def logout():
    session.clear()
    flash("You have been logged out.", "info")
    return redirect(url_for("index"))

# ---dashboard---
@app.route("/dashboard")
@login_required
def dashboard():
    conn = db.get_db()
    uid = current_user_id()
    total = conn.execute(
        "SELECT COUNT(*) c FROM reviews WHERE user_id = ?", (uid,)
    ).fetchone()["c"]
    fake = conn.execute(
        "SELECT COUNT(*) c FROM reviews WHERE user_id = ? AND prediction = 'Fake'", (uid,)
    ).fetchone()["c"]
    genuine = conn.execute(
        "SELECT COUNT(*) c FROM reviews WHERE user_id = ? AND prediction = 'Genuine'", (uid,)
    ).fetchone()["c"]
    recent = conn.execute(
        "SELECT action, details, created_at FROM activity_log WHERE user_id = ? "
        "ORDER BY created_at DESC LIMIT 8", (uid,)
    ).fetchall()
    conn.close()
    accuracy = get_model_accuracy() or 0.0
    return render_template(
        "dashboard.html",
        total=total, fake=fake, genuine=genuine, accuracy=accuracy, recent=recent,
    )
# --- upload---
@app.route("/upload", methods=["GET", "POST"])
@login_required
def upload():
    if request.method == "POST":
        file = request.files.get("dataset_file")
        if not file or file.filename == "":
            flash("Please choose a CSV or Excel file.", "danger")
            return redirect(url_for("upload"))
        if not allowed_file(file.filename):
            flash("Only .csv, .xlsx, .xls files are supported.", "danger")
            return redirect(url_for("upload"))
        filename = secure_filename(file.filename)
        timestamped = f"{datetime.now().strftime('%Y%m%d%H%M%S')}_{filename}"
        filepath = os.path.join(UPLOAD_DIR, timestamped)
        file.save(filepath)
        try:
            df = read_dataset_file(filepath, filename)
            df = normalize_columns(df)
        except Exception as e:
            flash(f"Could not read file: {e}", "danger")
            return redirect(url_for("upload"))
        if "review_text" not in df.columns:
            flash(
                "Dataset must include a review text column "
                "(e.g. 'review_text', 'review', 'text', or 'comment').", "danger"
            )
            return redirect(url_for("upload"))
        df = df.dropna(subset=["review_text"]).reset_index(drop=True)
        if len(df) == 0:
            flash("No valid review rows found in the file.", "danger")
            return redirect(url_for("upload"))
        # very large uploads
        if len(df) > 6000:
            df = df.head(6000)
        uid = current_user_id()
        conn = db.get_db()
        cur = conn.execute(
            "INSERT INTO datasets (user_id, filename, row_count) VALUES (?, ?, ?)",
            (uid, filename, len(df)),
        )
        dataset_id = cur.lastrowid
        conn.commit()
        texts = df["review_text"].astype(str).tolist()
        predictions = predict_batch(texts)
        sentiments = sentiment_batch(texts)
        spam_results = spam_batch(texts)
        dup_flags, dup_pairs, dup_pct = find_duplicates(texts)
        reviewer_names = df["reviewer_name"].tolist() if "reviewer_name" in df.columns else [None] * len(df)
        ratings = df["rating"].tolist() if "rating" in df.columns else [None] * len(df)
        for i, text in enumerate(texts):
            label, confidence = predictions[i]
            sentiment_label, _ = sentiments[i]
            spam_info = spam_results[i]
            conn.execute(
                """INSERT INTO reviews
                   (user_id, dataset_id, reviewer_name, review_text, rating,
                    prediction, confidence, sentiment, spam_score, is_duplicate)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (uid, dataset_id, reviewer_names[i], text, ratings[i],
                 label, confidence, sentiment_label, spam_info["spam_score"],
                 int(dup_flags[i])),
            )
        conn.commit()
        conn.close()
        session["active_dataset_id"] = dataset_id
        db.log_activity(uid, "Dataset uploaded", f"{filename} ({len(df)} reviews analyzed)")

        flash(f"Analyzed {len(df)} reviews from {filename}.", "success")
        return redirect(url_for("upload_preview", dataset_id=dataset_id))
    conn = db.get_db()
    datasets = conn.execute(
        "SELECT * FROM datasets WHERE user_id = ? ORDER BY uploaded_at DESC LIMIT 10",
        (current_user_id(),),
    ).fetchall()
    conn.close()
    return render_template("upload.html", datasets=datasets)
@app.route("/upload/preview/<int:dataset_id>")
@login_required
def upload_preview(dataset_id):
    conn = db.get_db()
    dataset = conn.execute(
        "SELECT * FROM datasets WHERE id = ? AND user_id = ?",
        (dataset_id, current_user_id()),
    ).fetchone()
    rows = conn.execute(
        "SELECT * FROM reviews WHERE dataset_id = ? LIMIT 25", (dataset_id,)
    ).fetchall()
    conn.close()
    if not dataset:
        flash("Dataset not found.", "danger")
        return redirect(url_for("upload"))
    return render_template("upload_preview.html", dataset=dataset, rows=rows)

# ---- fake review detection (single)---
@app.route("/detect", methods=["GET", "POST"])
@login_required
def detect():
    result = None
    if request.method == "POST":
        text = request.form.get("review_text", "").strip()
        if not text:
            flash("Please enter a review to analyze.", "danger")
            return redirect(url_for("detect"))
        label, confidence = predict_review(text)
        sentiment_label, _ = analyze_sentiment(text)
        spam_info = analyze_spam(text)
        uid = current_user_id()
        conn = db.get_db()
        conn.execute(
            """INSERT INTO reviews
               (user_id, dataset_id, reviewer_name, review_text, rating,
                prediction, confidence, sentiment, spam_score, is_duplicate)
               VALUES (?, NULL, NULL, ?, NULL, ?, ?, ?, ?, 0)""",
            (uid, text, label, confidence, sentiment_label, spam_info["spam_score"]),
        )
        conn.commit()
        conn.close()
        db.log_activity(uid, "Single review analyzed", f"Result: {label} ({confidence}%)")
        result = {
            "label": label,
            "confidence": confidence,
            "sentiment": sentiment_label,
            "spam": spam_info,
        }
    return render_template("detect.html", result=result)

# --- sentiment---
@app.route("/sentiment")
@login_required
def sentiment_page():
    dataset_id = request.args.get("dataset_id", session.get("active_dataset_id"))
    review_type = request.args.get("review_type", "")
    conn = db.get_db()
    query = "SELECT * FROM reviews WHERE user_id = ?"
    params = [current_user_id()]
    if dataset_id:
        query = "SELECT * FROM reviews WHERE dataset_id = ? AND user_id = ?"
        params = [dataset_id, current_user_id()]
    if review_type == "Genuine":
        query += " AND prediction = 'Genuine'"
    elif review_type == "Fake":
        query += " AND prediction = 'Fake'"
    query += " ORDER BY created_at DESC"
    rows = conn.execute(query, params).fetchall()
    datasets = conn.execute(
        "SELECT * FROM datasets WHERE user_id = ? ORDER BY uploaded_at DESC",
        (current_user_id(),),
    ).fetchall()
    conn.close()
    counts = {"Positive": 0, "Negative": 0, "Neutral": 0}
    for r in rows:
        if r["sentiment"] in counts:
            counts[r["sentiment"]] += 1
    return render_template(
        "sentiment.html", rows=rows, counts=counts, datasets=datasets,
        active_dataset_id=str(dataset_id) if dataset_id else "",
        review_type=review_type,
    )
@app.route("/sentiment/download")
@login_required
def download_sentiment_report():
    dataset_id = request.args.get("dataset_id", session.get("active_dataset_id"))
    review_type = request.args.get("review_type", "")
    conn = db.get_db()
    query = "SELECT reviewer_name, review_text, rating, prediction, confidence, sentiment, spam_score, is_duplicate FROM reviews WHERE user_id = ?"
    params = [current_user_id()]
    if dataset_id:
        query = "SELECT reviewer_name, review_text, rating, prediction, confidence, sentiment, spam_score, is_duplicate FROM reviews WHERE dataset_id = ? AND user_id = ?"
        params = [dataset_id, current_user_id()]
    if review_type == "Genuine":
        query += " AND prediction = 'Genuine'"
    elif review_type == "Fake":
        query += " AND prediction = 'Fake'"
    rows = conn.execute(query, params).fetchall()
    conn.close()
    df = pd.DataFrame([dict(r) for r in rows])
    buf = io.BytesIO()
    filename = "all_reviews"
    if dataset_id:
        dataset = db.get_db().execute(
            "SELECT filename FROM datasets WHERE id = ? AND user_id = ?",
            (dataset_id, current_user_id()),
        ).fetchone()
        filename = dataset["filename"] if dataset else filename
    if review_type:
        filename = f"{review_type.lower()}_{filename}"
    df.to_csv(buf, index=False, encoding="utf-8")
    buf.seek(0)
    return send_file(
        buf, as_attachment=True,
        download_name=f"sentiment_report_{filename}.csv",
        mimetype="text/csv",
    )
# --- duplicates---
@app.route("/duplicates")
@login_required
def duplicates_page():
    dataset_id = request.args.get("dataset_id", session.get("active_dataset_id"))
    conn = db.get_db()
    datasets = conn.execute(
        "SELECT * FROM datasets WHERE user_id = ? ORDER BY uploaded_at DESC",
        (current_user_id(),),
    ).fetchall()
    dup_pct = 0.0
    rows = []
    if dataset_id:
        rows = conn.execute(
            "SELECT * FROM reviews WHERE dataset_id = ? AND user_id = ?",
            (dataset_id, current_user_id()),
        ).fetchall()
        if rows:
            dup_pct = round(100 * sum(r["is_duplicate"] for r in rows) / len(rows), 2)
    conn.close()
    return render_template(
        "duplicates.html", rows=rows, dup_pct=dup_pct, datasets=datasets,
        active_dataset_id=str(dataset_id) if dataset_id else "",
    )
# ---- spam---
@app.route("/spam")
@login_required
def spam_page():
    dataset_id = request.args.get("dataset_id", session.get("active_dataset_id"))
    conn = db.get_db()
    datasets = conn.execute(
        "SELECT * FROM datasets WHERE user_id = ? ORDER BY uploaded_at DESC",
        (current_user_id(),),
    ).fetchall()
    rows = []
    if dataset_id:
        rows = conn.execute(
            "SELECT * FROM reviews WHERE dataset_id = ? AND user_id = ? "
            "ORDER BY spam_score DESC", (dataset_id, current_user_id()),
        ).fetchall()
    conn.close()
    return render_template(
        "spam.html", rows=rows, datasets=datasets,
        active_dataset_id=str(dataset_id) if dataset_id else "",
    )
# --- reviewer analysis---
@app.route("/reviewer")
@login_required
def reviewer_page():
    dataset_id = request.args.get("dataset_id", session.get("active_dataset_id"))
    conn = db.get_db()
    datasets = conn.execute(
        "SELECT * FROM datasets WHERE user_id = ? ORDER BY uploaded_at DESC",
        (current_user_id(),),
    ).fetchall()

    table = []
    if dataset_id:
        rows = conn.execute(
            "SELECT reviewer_name, review_text, rating FROM reviews "
            "WHERE dataset_id = ? AND user_id = ?", (dataset_id, current_user_id()),
        ).fetchall()
        if rows:
            df = pd.DataFrame([dict(r) for r in rows])
            result_df = analyze_reviewers(df)
            table = result_df.to_dict(orient="records")
    conn.close()
    return render_template(
        "reviewer.html", table=table, datasets=datasets,
        active_dataset_id=str(dataset_id) if dataset_id else "",
    )
# --- reports---
@app.route("/reports")
@login_required
def reports_page():
    conn = db.get_db()
    datasets = conn.execute(
        "SELECT * FROM datasets WHERE user_id = ? ORDER BY uploaded_at DESC",
        (current_user_id(),),
    ).fetchall()
    conn.close()
    return render_template("reports.html", datasets=datasets)
@app.route("/reports/pdf/<int:dataset_id>")
@login_required
def report_pdf(dataset_id):
    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors
    from reportlab.lib.units import cm
    from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
    from reportlab.lib.styles import getSampleStyleSheet
    conn = db.get_db()
    dataset = conn.execute(
        "SELECT * FROM datasets WHERE id = ? AND user_id = ?",
        (dataset_id, current_user_id()),
    ).fetchone()
    rows = conn.execute(
        "SELECT * FROM reviews WHERE dataset_id = ?", (dataset_id,)
    ).fetchall()
    conn.close()
    if not dataset:
        flash("Dataset not found.", "danger")
        return redirect(url_for("reports_page"))
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, topMargin=2 * cm)
    styles = getSampleStyleSheet()
    elements = [
        Paragraph("Fake Review Detection - Analysis Report", styles["Title"]),
        Spacer(1, 12),
        Paragraph(f"Dataset: {dataset['filename']}", styles["Normal"]),
        Paragraph(f"Total reviews: {len(rows)}", styles["Normal"]),
        Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}", styles["Normal"]),
        Spacer(1, 16),
    ]
    fake_count = sum(1 for r in rows if r["prediction"] == "Fake")
    genuine_count = sum(1 for r in rows if r["prediction"] == "Genuine")
    dup_count = sum(1 for r in rows if r["is_duplicate"])
    summary_data = [
        ["Metric", "Value"],
        ["Genuine reviews", genuine_count],
        ["Fake reviews", fake_count],
        ["Duplicate reviews", dup_count],
    ]
    summary_table = Table(summary_data, colWidths=[8 * cm, 6 * cm])
    summary_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f4e8c")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
    ]))
    elements.append(summary_table)
    elements.append(Spacer(1, 20))
    elements.append(Paragraph("Sample Review Results", styles["Heading2"]))
    detail_data = [["Review (truncated)", "Prediction", "Confidence", "Sentiment"]]
    for r in rows[:40]:
        text = (r["review_text"] or "")[:60] + ("..." if len(r["review_text"] or "") > 60 else "")
        detail_data.append([text, r["prediction"], f"{r['confidence']}%", r["sentiment"]])
    detail_table = Table(detail_data, colWidths=[9 * cm, 2.5 * cm, 2.5 * cm, 2.5 * cm])
    detail_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f4e8c")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
    ]))
    elements.append(detail_table)

    doc.build(elements)
    buf.seek(0)

    db.log_activity(current_user_id(), "PDF report downloaded", dataset["filename"])
    return send_file(
        buf, as_attachment=True,
        download_name=f"report_{dataset['filename']}.pdf",
        mimetype="application/pdf",
    )
@app.route("/reports/excel/<int:dataset_id>")
@login_required
def report_excel(dataset_id):
    conn = db.get_db()
    dataset = conn.execute(
        "SELECT * FROM datasets WHERE id = ? AND user_id = ?",
        (dataset_id, current_user_id()),
    ).fetchone()
    rows = conn.execute(
        "SELECT reviewer_name, review_text, rating, prediction, confidence, "
        "sentiment, spam_score, is_duplicate FROM reviews WHERE dataset_id = ?",
        (dataset_id,),
    ).fetchall()
    conn.close()
    if not dataset:
        flash("Dataset not found.", "danger")
        return redirect(url_for("reports_page"))
    df = pd.DataFrame([dict(r) for r in rows])
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Results")
    buf.seek(0)

    db.log_activity(current_user_id(), "Excel report exported", dataset["filename"])
    return send_file(
        buf, as_attachment=True,
        download_name=f"report_{dataset['filename']}.xlsx",
        mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
@app.route("/reports/qr/<int:dataset_id>")
@login_required
def report_qr(dataset_id):
    try:
        import qrcode
    except ImportError:
        flash(
            "QR code generation requires the 'qrcode' package. "
            "Install it with: pip install qrcode[pil]", "warning"
        )
        return redirect(url_for("reports_page"))
    conn = db.get_db()
    dataset = conn.execute(
        "SELECT * FROM datasets WHERE id = ? AND user_id = ?",
        (dataset_id, current_user_id()),
    ).fetchone()
    conn.close()
    if not dataset:
        flash("Dataset not found.", "danger")
        return redirect(url_for("reports_page"))
    share_url = url_for("upload_preview", dataset_id=dataset_id, _external=True)
    img = qrcode.make(share_url)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return send_file(buf, mimetype="image/png")
# ---admin---
@app.route("/admin")
@admin_required
def admin_panel():
    conn = db.get_db()
    users = conn.execute("SELECT * FROM users ORDER BY created_at DESC").fetchall()
    datasets = conn.execute(
        "SELECT datasets.*, users.username FROM datasets "
        "JOIN users ON users.id = datasets.user_id ORDER BY uploaded_at DESC"
    ).fetchall()
    stats = {
        "total_users": conn.execute("SELECT COUNT(*) c FROM users").fetchone()["c"],
        "total_datasets": conn.execute("SELECT COUNT(*) c FROM datasets").fetchone()["c"],
        "total_reviews": conn.execute("SELECT COUNT(*) c FROM reviews").fetchone()["c"],
        "total_fake": conn.execute(
            "SELECT COUNT(*) c FROM reviews WHERE prediction = 'Fake'"
        ).fetchone()["c"],
    }
    conn.close()
    return render_template("admin.html", users=users, datasets=datasets, stats=stats)
@app.route("/admin/delete-user/<int:user_id>", methods=["POST"])
@admin_required
def admin_delete_user(user_id):
    if user_id == session.get("user_id"):
        flash("You cannot delete your own account while logged in.", "danger")
        return redirect(url_for("admin_panel"))
    conn = db.get_db()
    conn.execute("DELETE FROM users WHERE id = ?", (user_id,))
    conn.commit()
    conn.close()
    flash("User deleted.", "success")
    return redirect(url_for("admin_panel"))
@app.route("/admin/delete-dataset/<int:dataset_id>", methods=["POST"])
@admin_required
def admin_delete_dataset(dataset_id):
    conn = db.get_db()
    conn.execute("DELETE FROM reviews WHERE dataset_id = ?", (dataset_id,))
    conn.execute("DELETE FROM datasets WHERE id = ?", (dataset_id,))
    conn.commit()
    conn.close()
    flash("Dataset and its reviews deleted.", "success")
    return redirect(url_for("admin_panel"))
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
