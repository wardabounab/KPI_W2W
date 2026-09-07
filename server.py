"""
server.py — Flask REST API
Exposes calculation endpoints for Tripping Speed and Weight-to-Weight (W2W).
"""

import os
from flask import Flask, request, render_template, jsonify
import pandas as pd
from typing import Optional

from calculations.tripping_speed import calculate_tripping_speed
from calculations.w2w import calculate_w2w

app = Flask(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

ALLOWED_EXTENSIONS = {"xlsx", "xls"}


def _allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def _read_excel(request_files) -> Optional[pd.DataFrame]:
    """
    Validates and reads the uploaded Excel file from the multipart request.
    Returns a DataFrame or raises a ValueError with a descriptive message.
    """
    if "file" not in request_files:
        raise ValueError("No file part found in the request. Use key 'file'.")

    file = request_files["file"]

    if file.filename == "":
        raise ValueError("No file selected.")

    if not _allowed_file(file.filename):
        raise ValueError(f"Unsupported file type. Allowed: {ALLOWED_EXTENSIONS}")

    # Try reading with pandas. If a legacy .xls file is uploaded and the
    # optional `xlrd` engine is not installed, provide a helpful error.
    ext = file.filename.rsplit(".", 1)[1].lower()
    try:
        if ext == "xls":
            # Older Excel BIFF format may require the 'xlrd' package.
            try:
                df = pd.read_excel(file, engine="xlrd")
            except ImportError as ie:
                raise ValueError(
                    "Reading .xls files requires the 'xlrd' package. "
                    "Install it with: pip install xlrd"
                ) from ie
        else:
            df = pd.read_excel(file)
    except ValueError as ve:
        # Re-raise as ValueError with friendly message
        raise ValueError(f"Could not read Excel file: {ve}") from ve

    return df


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.route("/")
def index():
    """
    Renders the local ad-hoc web testing dashboard.
    Flask expects 'index.html' to live inside a adjacent '/templates' directory.
    """
    return render_template("index.html")


@app.post("/api/calculate/tripping")
def tripping():
    """
    Calculate tripping speed from an uploaded Excel file.

    Request  : multipart/form-data  — field name: 'file'  (.xlsx / .xls)
    Response : application/json     — calculation results or error detail
    """
    try:
        df = _read_excel(request.files)
        result = calculate_tripping_speed(df)
        return jsonify({"status": "success", "data": result}), 200

    except ValueError as exc:
        return jsonify({"status": "error", "message": str(exc)}), 400

    except Exception as exc:
        # Catch unexpected errors and return a safe message.
        app.logger.exception("Unhandled error in /calculate/tripping")
        return jsonify({"status": "error", "message": "Internal server error.", "detail": str(exc)}), 500


@app.post("/api/calculate/w2w")
def w2w():
    """
    Calculate weight-to-weight (W2W) from an uploaded Excel file.

    Request  : multipart/form-data  — field name: 'file'  (.xlsx / .xls)
    Response : application/json     — calculation results or error detail
    """
    try:
        df = _read_excel(request.files)
        result = calculate_w2w(df)
        return jsonify({"status": "success", "data": result}), 200

    except ValueError as exc:
        return jsonify({"status": "error", "message": str(exc)}), 400

    except Exception as exc:
        app.logger.exception("Unhandled error in /calculate/w2w")
        return jsonify({"status": "error", "message": "Internal server error.", "detail": str(exc)}), 500


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    debug_mode = os.getenv("FLASK_DEBUG", "false").lower() == "true"
    
    # Remplacer 5060 par 5000 (qui est un port sûr et autorisé par les navigateurs)
    port = int(os.getenv("PORT", 5000))

    # Lancement du serveur
    app.run(host="0.0.0.0", debug=debug_mode, port=port)