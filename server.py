"""
server.py — Flask REST API
Endpoints for Tripping Speed and Weight-to-Weight (W2W) + exports.
"""

import io
import os

from flask import Flask, request, render_template, jsonify, send_file
import pandas as pd

from calculations.Dropdowns import WELLS, RIGS, PHASES, ROTARY_SYSTEMS, DRILL_PIPES, BHA
from calculations.tripping_speed import calculate_tripping_speed
from calculations.w2w import calculate_w2w, parse_config, build_export

app = Flask(__name__)

ALLOWED_EXTENSIONS = {"xlsx", "xls", "csv"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def _read_dataframe(request_files) -> pd.DataFrame:
    """Reads the uploaded file (Excel or CSV) into a DataFrame."""
    if "file" not in request_files:
        raise ValueError("No file part found in the request. Use key 'file'.")

    file = request_files["file"]
    if file.filename == "":
        raise ValueError("No file selected.")
    if not _allowed_file(file.filename):
        raise ValueError(f"Unsupported file type. Allowed: {sorted(ALLOWED_EXTENSIONS)}")

    ext = file.filename.rsplit(".", 1)[1].lower()
    try:
        if ext == "csv":
            raw = file.read()
            if isinstance(raw, bytes):
                try:
                    text = raw.decode("utf-8-sig")
                except UnicodeDecodeError:
                    text = raw.decode("latin-1")
            else:
                text = raw
            sep = ";" if text.count(";") > text.count(",") else ","
            return pd.read_csv(io.StringIO(text), sep=sep)

        if ext == "xls":
            try:
                return pd.read_excel(file, engine="xlrd")
            except ImportError as ie:
                raise ValueError(
                    "Reading .xls files requires the 'xlrd' package. "
                    "Install it with: pip install xlrd"
                ) from ie

        return pd.read_excel(file)

    except ValueError:
        raise
    except Exception as exc:
        raise ValueError(f"Could not read file: {exc}") from exc


# ---------------------------------------------------------------------------
# Routes  (UNE SEULE fois par endpoint !)
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    """Renders the dashboard and injects the dropdown reference lists."""
    return render_template(
        "index.html",
        wells=WELLS,
        rigs=RIGS,
        phases=PHASES,
        rotary_systems=ROTARY_SYSTEMS,
        drill_pipes=DRILL_PIPES,
        bha_names=BHA,
    )


@app.post("/api/calculate/tripping")
def tripping():
    try:
        df = _read_dataframe(request.files)
        result = calculate_tripping_speed(df)
        return jsonify({"status": "success", "data": result}), 200
    except ValueError as exc:
        return jsonify({"status": "error", "message": str(exc)}), 400
    except Exception as exc:
        app.logger.exception("Unhandled error in /api/calculate/tripping")
        return jsonify({"status": "error", "message": "Internal server error.",
                        "detail": str(exc)}), 500


@app.post("/api/calculate/w2w")
def w2w():
    try:
        df = _read_dataframe(request.files)
        cfg = parse_config(request.form)
        result = calculate_w2w(df, cfg=cfg)
        return jsonify({"status": "success", "data": result}), 200
    except ValueError as exc:
        return jsonify({"status": "error", "message": str(exc)}), 400
    except Exception as exc:
        app.logger.exception("Unhandled error in /api/calculate/w2w")
        return jsonify({"status": "error", "message": "Internal server error.",
                        "detail": str(exc)}), 500


@app.post("/api/export/<fmt>")
def export(fmt):
    """Exports the (possibly filtered) cycles to CSV or XLSX."""
    if fmt not in ("csv", "xlsx"):
        return jsonify({"status": "error",
                        "message": "Unsupported export format."}), 400
    try:
        payload = request.get_json(silent=True) or {}
        data = payload.get("data") or {}
        meta = payload.get("meta") or {}
        blob, mimetype, filename = build_export(fmt, data, meta)
        return send_file(
            io.BytesIO(blob),
            mimetype=mimetype,
            as_attachment=True,
            download_name=filename,
        )
    except ValueError as exc:
        return jsonify({"status": "error", "message": str(exc)}), 400
    except Exception as exc:
        app.logger.exception("Unhandled error in /api/export/%s", fmt)
        return jsonify({"status": "error", "message": "Internal server error.",
                        "detail": str(exc)}), 500


if __name__ == "__main__":
    debug_mode = os.getenv("FLASK_DEBUG", "false").lower() == "true"
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", debug=debug_mode, port=port)
