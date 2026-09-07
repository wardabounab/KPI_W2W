from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
import warnings
import traceback
import os
import json


# ============================================================
# FLASK
# ============================================================

app = Flask(__name__)

app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024


# ============================================================
# CONFIGURATION
# ============================================================

DEBUG_W2W = True

WOB_ON_THRESHOLD = 1.0

GAP_THRESHOLD = pd.Timedelta(minutes=10)

# Tolérance minimale de mouvement du bit
BIT_MOVEMENT_TOL = 0.02

# Course BPOS attendue par le cahier des charges :
# 9 m pour une connexion joint et jusqu'à 31 m pour un stand.
MIN_BPOS_RISE = 9.0
MAX_BPOS_RISE = 31.0

# Position basse du bloc et tolérance de retour au fond (DBTM ~= DMEA).
LOWER_BPOS_MAX = 9.0
BOTTOM_DEPTH_TOL = 0.30

# Durée minimale d'une connexion
MIN_CONNECTION_SECONDS = 5

# Durée minimale W2W
MIN_W2W_MINUTES = 0.5

# Durée maximale W2W
MAX_W2W_MINUTES = 30.0

# Valeurs invalides fréquentes
BAD_VALUES = {
    -999,
    -9999,
    -999.25,
    -999.99,
    -9999.25,
    -9999.99,
    -99999,
}


# ============================================================
# HELPERS
# ============================================================

def debug(message):

    if DEBUG_W2W:
        print(f"[W2W] {message}")


def clean_column_name(column):

    return (
        str(column)
        .strip()
        .replace("\n", " ")
        .replace("\r", " ")
    )


# ============================================================
# DETECTION DES COLONNES
# ============================================================

def detect_columns(df):

    mapping = {}

    timestamp_col = None

    wob_col = None

    for original_col in df.columns:

        col = clean_column_name(original_col)

        lc = col.lower().strip()

        # --------------------------------------------------------
        # TIMESTAMP
        # --------------------------------------------------------

        if lc in [
            "index",
            "timestamp",
            "date time",
            "datetime",
            "date_time",
            "date-time",
        ]:

            timestamp_col = original_col

        # --------------------------------------------------------
        # DMEA / DEPTH
        # --------------------------------------------------------

        if (
            "dmea" in lc
            or "measured depth" in lc
            or "hole depth" in lc
            or lc == "depth(m)"
            or lc == "depth"
        ):

            mapping["dmea"] = original_col

        # --------------------------------------------------------
        # DBTM / BIT DEPTH
        # --------------------------------------------------------

        if (
            "dbtm" in lc
            or "bit depth" in lc
        ):

            mapping["dbtm"] = original_col

        # --------------------------------------------------------
        # BPOS
        # --------------------------------------------------------

        if (
            "bpos" in lc
            or "block position" in lc
            or "block pos" in lc
        ):

            mapping["bpos"] = original_col

        # --------------------------------------------------------
        # HOOKLOAD
        # --------------------------------------------------------

        if (
            "hkla" in lc
            or "hkld" in lc
            or "hookload" in lc
            or "hook load" in lc
        ):

            mapping["hkla"] = original_col

        # --------------------------------------------------------
        # WOB
        # --------------------------------------------------------

        if (
            "woba" in lc
            or lc == "wob"
            or "weight on bit" in lc
            or "weight-on-bit" in lc
            or "weight_on_bit" in lc
        ):

            wob_col = original_col

    if timestamp_col is not None:

        mapping["index"] = timestamp_col

    if wob_col is not None:

        mapping["wob"] = wob_col

    return mapping


# ============================================================
# PREPROCESSING
# ============================================================

def preprocess_data(raw_df):

    df = raw_df.copy()

    # --------------------------------------------------------
    # Nettoyage noms colonnes
    # --------------------------------------------------------

    df.columns = [
        clean_column_name(c)
        for c in df.columns
    ]

    debug("Détection des colonnes...")

    mapping = detect_columns(df)

    print("\nDetected columns:")

    for key, value in mapping.items():

        print(
            f"  {key:8s} -> {value}"
        )

    wob_available = "wob" in mapping

    print(
        f"\nWOB available: {wob_available}"
    )

    # --------------------------------------------------------
    # Colonnes obligatoires
    # --------------------------------------------------------

    required = [
        "index",
        "dmea",
        "dbtm",
        "bpos",
        "hkla",
    ]

    missing = [
        x
        for x in required
        if x not in mapping
    ]

    if missing:

        raise ValueError(
            "Missing required column(s): "
            f"{missing}. "
            f"Found columns: {list(df.columns)}"
        )

    # --------------------------------------------------------
    # Sélection
    # --------------------------------------------------------

    selected = {
        mapping["index"]: "ts",
        mapping["dmea"]: "dmea",
        mapping["dbtm"]: "dbtm",
        mapping["bpos"]: "bpos",
        mapping["hkla"]: "hkla",
    }

    if wob_available:

        selected[
            mapping["wob"]
        ] = "wob"

    df = (
        df[
            list(selected.keys())
        ]
        .rename(columns=selected)
    )

    # --------------------------------------------------------
    # Timestamp
    # --------------------------------------------------------

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # RTOM exports may use a mixed date/time representation.
        # dayfirst=True is safe for the DD/MM/YYYY form and explicit
        # format inference is avoided to remove the pandas warning.
        df["ts"] = pd.to_datetime(
            df["ts"],
            errors="coerce",
            dayfirst=True
        )

    # --------------------------------------------------------
    # Numérique
    # --------------------------------------------------------

    numeric_cols = [
        "dmea",
        "dbtm",
        "bpos",
        "hkla",
    ]

    if wob_available:

        numeric_cols.append("wob")

    for col in numeric_cols:

        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )

    # --------------------------------------------------------
    # Supprimer valeurs invalides
    # --------------------------------------------------------

    df[numeric_cols] = (
        df[numeric_cols]
        .replace(
            list(BAD_VALUES),
            np.nan
        )
    )

    # --------------------------------------------------------
    # Supprimer NaN
    # --------------------------------------------------------

    df = df.dropna(
        subset=[
            "ts",
            "dmea",
            "dbtm",
            "bpos",
            "hkla",
        ]
    )

    if wob_available:

        df = df.dropna(
            subset=["wob"]
        )

    # --------------------------------------------------------
    # Trier
    # --------------------------------------------------------

    df = (
        df
        .sort_values("ts")
        .reset_index(drop=True)
    )

    if len(df) == 0:

        raise ValueError(
            "No valid rows after preprocessing."
        )

    # --------------------------------------------------------
    # Timestamp comme index
    # --------------------------------------------------------

    df = df.set_index("ts")

    # --------------------------------------------------------
    # Supprimer doublons timestamp
    # --------------------------------------------------------

    df = df[
        ~df.index.duplicated(
            keep="first"
        )
    ]

    # --------------------------------------------------------
    # Hookload
    # --------------------------------------------------------

    df["hkla"] = df["hkla"].astype(float)

    # --------------------------------------------------------
    # Gap
    # --------------------------------------------------------

    time_diff = df.index.to_series().diff()

    df["large_gap"] = (
        time_diff > GAP_THRESHOLD
    )

    # --------------------------------------------------------
    # Seuil slips
    # --------------------------------------------------------

    slip_threshold = compute_slip_threshold(
        df["hkla"]
    )

    print(
        f"\nComputed slip threshold: "
        f"{slip_threshold:.3f}"
    )

    debug(
        f"Rows after preprocessing: {len(df)}"
    )

    debug(
        f"Hookload min: {df['hkla'].min():.3f}"
    )

    debug(
        f"Hookload max: {df['hkla'].max():.3f}"
    )

    debug(
        f"Slip threshold: {slip_threshold:.3f}"
    )

    return (
        df,
        slip_threshold,
        wob_available,
    )


# ============================================================
# CALCUL DU SEUIL HOOKLOAD
# ============================================================

def compute_slip_threshold(series):

    values = (
        series
        .dropna()
        .to_numpy()
    )

    if len(values) == 0:

        return 0.0

    if len(values) < 10:

        return float(
            np.median(values)
        )

    # --------------------------------------------------------
    # Initialisation
    # --------------------------------------------------------

    low = float(
        np.percentile(
            values,
            25
        )
    )

    high = float(
        np.percentile(
            values,
            75
        )
    )

    # --------------------------------------------------------
    # K-means 1D
    # --------------------------------------------------------

    for _ in range(30):

        distances_low = (
            np.abs(
                values - low
            )
        )

        distances_high = (
            np.abs(
                values - high
            )
        )

        low_group = values[
            distances_low
            <= distances_high
        ]

        high_group = values[
            distances_high
            < distances_low
        ]

        if len(low_group) == 0:
            break

        if len(high_group) == 0:
            break

        new_low = float(
            np.mean(low_group)
        )

        new_high = float(
            np.mean(high_group)
        )

        if (
            abs(new_low - low) < 0.001
            and
            abs(new_high - high) < 0.001
        ):

            break

        low = new_low
        high = new_high

    threshold = (
        low + high
    ) / 2

    return float(threshold)


# ============================================================
# W2W DETECTION
# ============================================================

def detect_w2w_cycles(
    df,
    slip_threshold,
    wob_available=False,
):
    """
    Detect W2W connection cycles from RTOM signals.

    Important for the current RTOM export:
    - WOB is not present, so the detector does NOT invent WOB values.
    - BPOS + HKLA are the primary connection signals.
    - DBTM/DMEA are used as consistency checks, not as a strict
      "DBTM must be perfectly stable during slips" condition.
    - After slip-out, the cycle is completed when the block returns
      to its lower position and the bit returns close to the depth
      recorded at pull-off/slip-in.
    """

    df = df.copy()

    dbtm = pd.to_numeric(df["dbtm"], errors="coerce").astype(float)
    dmea = pd.to_numeric(df["dmea"], errors="coerce").astype(float)
    bpos = pd.to_numeric(df["bpos"], errors="coerce").astype(float)
    hkla = pd.to_numeric(df["hkla"], errors="coerce").astype(float)

    # 5 samples is appropriate for the ~5 s RTOM sampling seen in the
    # supplied run, while keeping transitions reasonably sharp.
    dbtm_s = dbtm.rolling(5, center=True, min_periods=1).median()
    dmea_s = dmea.rolling(5, center=True, min_periods=1).median()
    bpos_s = bpos.rolling(5, center=True, min_periods=1).median()
    hkla_s = hkla.rolling(5, center=True, min_periods=1).median()

    bpos_diff = bpos_s.diff()
    dbtm_diff = dbtm_s.diff()

    block_up = bpos_diff > 0.03
    block_down = bpos_diff < -0.03
    bit_down = dbtm_diff > BIT_MOVEMENT_TOL
    bit_up = dbtm_diff < -BIT_MOVEMENT_TOL

    in_slips = hkla_s <= slip_threshold
    out_of_slips = hkla_s > slip_threshold

    # These are deliberately more permissive than the previous version.
    # The real data can have small depth differences between DMEA and DBTM.
    BOTTOM_TOL = max(BOTTOM_DEPTH_TOL, 0.50)
    POST_DBTM_TOL = 1.50
    SLIP_DBTM_TOL = 3.00
    MIN_PRE_SECONDS = 5
    MAX_PRE_SECONDS = 20 * 60
    MAX_POST_SECONDS = 20 * 60
    MIN_SLIP_SECONDS = MIN_CONNECTION_SECONDS

    LOWER_BPOS = LOWER_BPOS_MAX
    MIN_RISE = MIN_BPOS_RISE
    MAX_RISE = MAX_BPOS_RISE

    DRILLING = "DRILLING"
    PRE_CONNECTION = "PRE_CONNECTION"
    IN_SLIPS = "IN_SLIPS"
    POST_CONNECTION = "POST_CONNECTION"

    state = DRILLING
    cycles = []

    pre_start_idx = None
    slip_in_idx = None
    slip_out_idx = None
    peak_bpos = None
    cycle = None

    debug_candidates = 0
    debug_slip_in = 0
    debug_slip_out = 0

    def reset():
        return DRILLING, None, None, None, None, None

    for i in range(1, len(df)):
        current_time = df.index[i]
        current_bpos = float(bpos_s.iloc[i])
        current_dbtm = float(dbtm_s.iloc[i])
        current_dmea = float(dmea_s.iloc[i])
        current_hkla = float(hkla_s.iloc[i])

        if bool(df["large_gap"].iloc[i]):
            debug(f"Large data gap at {current_time}")
            (
                state, pre_start_idx, slip_in_idx,
                slip_out_idx, peak_bpos, cycle
            ) = reset()
            continue

        block_lower = current_bpos <= LOWER_BPOS
        off_bottom = current_dbtm < current_dmea - BOTTOM_TOL

        # ============================================================
        # DRILLING -> PRE-CONNECTION
        # ============================================================
        if state == DRILLING:
            if wob_available:
                wob = pd.to_numeric(df["wob"], errors="coerce").astype(float)
                wob_on = wob > WOB_ON_THRESHOLD

                pullout_condition = (
                    bool(wob_on.iloc[i - 1])
                    and not bool(wob_on.iloc[i])
                    and block_lower
                )
            else:
                # No WOB in this RTOM export:
                # start from an upward block movement while hookload is
                # still above slips. Do not require a single DBTM sample
                # to be "off bottom", because RTOM depth channels can lag.
                pullout_condition = (
                    bool(block_up.iloc[i])
                    and current_hkla > slip_threshold
                    and block_lower
                )

            if pullout_condition:
                debug_candidates += 1
                state = PRE_CONNECTION
                pre_start_idx = i
                peak_bpos = current_bpos
                cycle = {
                    "pre_start": i,
                    "pre_dbtm": current_dbtm,
                    "pre_dmea": current_dmea,
                }

                debug(
                    f"PRE-CONNECTION {current_time} | "
                    f"DBTM={current_dbtm:.2f} | "
                    f"BPOS={current_bpos:.2f} | "
                    f"HKLA={current_hkla:.2f}"
                )
            continue

        # ============================================================
        # PRE-CONNECTION -> IN-SLIPS
        # ============================================================
        if state == PRE_CONNECTION:
            if peak_bpos is None or current_bpos > peak_bpos:
                peak_bpos = current_bpos

            initial_bpos = float(bpos_s.iloc[pre_start_idx])
            bpos_rise = peak_bpos - initial_bpos

            elapsed_pre = (
                current_time - df.index[pre_start_idx]
            ).total_seconds()

            # A real connection requires a meaningful block travel.
            if in_slips.iloc[i] and MIN_RISE <= bpos_rise <= MAX_RISE:
                slip_in_idx = i
                cycle["slip_in"] = i
                cycle["bpos_rise"] = float(bpos_rise)
                cycle["slip_dbtm"] = current_dbtm

                state = IN_SLIPS
                debug_slip_in += 1

                debug(
                    f"SLIP-IN {current_time} | "
                    f"BPOS rise={bpos_rise:.2f} m | "
                    f"HKLA={current_hkla:.2f}"
                )
                continue

            # Do NOT cancel merely because DBTM moved.
            # Cancel only if the candidate lasts too long without reaching
            # slips, or if the block clearly returned to its lower position.
            if (
                elapsed_pre > MAX_PRE_SECONDS
                or (
                    block_lower
                    and elapsed_pre >= MIN_PRE_SECONDS
                    and not block_up.iloc[i]
                    and current_hkla > slip_threshold
                    and bpos_rise < MIN_RISE
                )
            ):
                debug(f"Cancelled PRE-CONNECTION at {current_time}")
                (
                    state, pre_start_idx, slip_in_idx,
                    slip_out_idx, peak_bpos, cycle
                ) = reset()
            continue

        # ============================================================
        # IN-SLIPS -> SLIP-OUT
        # ============================================================
        if state == IN_SLIPS:
            if current_bpos > peak_bpos:
                peak_bpos = current_bpos

            connection_seconds = (
                current_time - df.index[slip_in_idx]
            ).total_seconds()

            # IMPORTANT:
            # Do not reject because DBTM is not perfectly stable.
            # During a real connection the depth channels can move a little.
            if (
                connection_seconds >= MIN_SLIP_SECONDS
                and out_of_slips.iloc[i]
            ):
                slip_out_idx = i
                cycle["slip_out"] = i

                state = POST_CONNECTION
                debug_slip_out += 1

                debug(
                    f"SLIP-OUT {current_time} | "
                    f"Connection={connection_seconds:.1f}s"
                )
            continue

        # ============================================================
        # POST-CONNECTION -> RETURN TO BOTTOM
        # ============================================================
        if state == POST_CONNECTION:
            post_seconds = (
                current_time - df.index[slip_out_idx]
            ).total_seconds()

            # Reference depth is the depth at pull-off/slip-in.
            # We accept a small movement because DBTM/DMEA are not identical
            # channels and the RTOM export is sampled every few seconds.
            ref_dbtm = float(cycle.get("pre_dbtm", current_dbtm))

            depth_back_near_reference = (
                abs(current_dbtm - ref_dbtm) <= POST_DBTM_TOL
            )

            # Primary completion signal: block has come back down.
            # Secondary confirmation: bit depth is close to the pre-connection
            # depth OR the bit is moving down.
            drilling_resumed = (
                block_lower
                and (
                    depth_back_near_reference
                    or bool(bit_down.iloc[i])
                )
            )

            if drilling_resumed:
                post_end_idx = i

                pre_minutes = (
                    df.index[slip_in_idx] -
                    df.index[pre_start_idx]
                ).total_seconds() / 60.0

                connection_minutes = (
                    df.index[slip_out_idx] -
                    df.index[slip_in_idx]
                ).total_seconds() / 60.0

                post_minutes = (
                    df.index[post_end_idx] -
                    df.index[slip_out_idx]
                ).total_seconds() / 60.0

                w2w_minutes = (
                    pre_minutes +
                    connection_minutes +
                    post_minutes
                )

                abnormal_reasons = []

                if w2w_minutes < MIN_W2W_MINUTES:
                    abnormal_reasons.append("W2W duration too short")

                if w2w_minutes > MAX_W2W_MINUTES:
                    abnormal_reasons.append("W2W duration too long")

                bpos_rise = float(cycle.get("bpos_rise", 0.0))

                connection_type = (
                    "stand" if bpos_rise >= 15.0 else "joint"
                )

                dbtm_segment = dbtm_s.iloc[
                    slip_in_idx:slip_out_idx + 1
                ]

                dbtm_change = (
                    float(dbtm_segment.max() - dbtm_segment.min())
                    if len(dbtm_segment) else 0.0
                )

                cycle_result = {
                    "cycle_number": len(cycles) + 1,
                    "connection_type": connection_type,
                    "depth_dmea_m": float(
                        df["dmea"].iloc[pre_start_idx]
                    ),
                    "date_pre_start":
                        df.index[pre_start_idx].isoformat(),
                    "date_slip_in":
                        df.index[slip_in_idx].isoformat(),
                    "date_slip_out":
                        df.index[slip_out_idx].isoformat(),
                    "date_post_end":
                        df.index[post_end_idx].isoformat(),
                    "pre_connection_minutes": float(pre_minutes),
                    "connection_minutes": float(connection_minutes),
                    "post_connection_minutes": float(post_minutes),
                    "w2w_minutes": float(w2w_minutes),
                    "pre_bpos_change_m": float(bpos_rise),
                    "conn_dbtm_change_m": float(dbtm_change),
                    "abnormal": bool(abnormal_reasons),
                    "abnormal_reasons": abnormal_reasons,
                }

                cycles.append(cycle_result)

                debug(
                    f"CYCLE DETECTED #{len(cycles)} | "
                    f"W2W={w2w_minutes:.2f} min | "
                    f"Connection={connection_minutes:.2f} min | "
                    f"BPOS rise={bpos_rise:.2f} m"
                )

                (
                    state, pre_start_idx, slip_in_idx,
                    slip_out_idx, peak_bpos, cycle
                ) = reset()

            elif post_seconds > MAX_POST_SECONDS:
                debug(
                    f"Cancelled POST-CONNECTION at {current_time} "
                    f"(timeout={post_seconds:.0f}s)"
                )
                (
                    state, pre_start_idx, slip_in_idx,
                    slip_out_idx, peak_bpos, cycle
                ) = reset()

    debug(
        "Detection finished | "
        f"candidates={debug_candidates} | "
        f"slip-in={debug_slip_in} | "
        f"slip-out={debug_slip_out} | "
        f"cycles={len(cycles)}"
    )

    return cycles


# ============================================================
# GLOBAL METRICS
# ============================================================

def calculate_global_metrics(cycles):

    if not cycles:

        return {

            "total_cycles": 0,

            "normal_cycles": 0,

            "abnormal_cycles": 0,

            "avg_w2w_minutes": None,

            "avg_pre_connection_minutes": None,

            "avg_connection_minutes": None,

            "avg_post_connection_minutes": None,

            "total_w2w_minutes": 0,

            "min_w2w_minutes": None,

            "max_w2w_minutes": None,

            "min_connection_minutes": None,

            "max_connection_minutes": None,

            "best_connection_index": None,

            "worst_connection_index": None,
        }

    normal = [
        c
        for c in cycles
        if not c["abnormal"]
    ]

    abnormal = [
        c
        for c in cycles
        if c["abnormal"]
    ]

    def avg(key):

        values = [
            c[key]
            for c in normal
        ]

        if not values:

            return None

        return float(
            np.mean(values)
        )

    w2w_values = [
        c["w2w_minutes"]
        for c in normal
    ]

    connection_values = [
        c["connection_minutes"]
        for c in normal
    ]

    best_index = None

    worst_index = None

    if connection_values:

        best_position = int(
            np.argmin(
                connection_values
            )
        )

        worst_position = int(
            np.argmax(
                connection_values
            )
        )

        best_index = normal[
            best_position
        ]["cycle_number"]

        worst_index = normal[
            worst_position
        ]["cycle_number"]

    return {

        "total_cycles":
            len(cycles),

        "normal_cycles":
            len(normal),

        "abnormal_cycles":
            len(abnormal),

        "avg_w2w_minutes":
            avg("w2w_minutes"),

        "avg_pre_connection_minutes":
            avg(
                "pre_connection_minutes"
            ),

        "avg_connection_minutes":
            avg(
                "connection_minutes"
            ),

        "avg_post_connection_minutes":
            avg(
                "post_connection_minutes"
            ),

        "total_w2w_minutes":
            float(
                sum(w2w_values)
            )
            if w2w_values
            else 0,

        "min_w2w_minutes":
            float(
                min(w2w_values)
            )
            if w2w_values
            else None,

        "max_w2w_minutes":
            float(
                max(w2w_values)
            )
            if w2w_values
            else None,

        "min_connection_minutes":
            float(
                min(connection_values)
            )
            if connection_values
            else None,

        "max_connection_minutes":
            float(
                max(connection_values)
            )
            if connection_values
            else None,

        "best_connection_index":
            best_index,

        "worst_connection_index":
            worst_index,
    }


# ============================================================
# CALCUL W2W PRINCIPAL
# ============================================================

def calculate_w2w(raw_df):

    df, slip_threshold, wob_available = (
        preprocess_data(
            raw_df
        )
    )

    cycles = detect_w2w_cycles(
        df,
        slip_threshold,
        wob_available,
    )

    metrics = calculate_global_metrics(
        cycles
    )

    return {

        "input_rows":
            int(len(raw_df)),

        "clean_rows":
            int(len(df)),

        "dropped_rows":
            int(
                len(raw_df)
                -
                len(df)
            ),

        "wob_available":
            bool(wob_available),

        "slip_threshold_ton":
            float(
                slip_threshold
            ),

        "wob_threshold_ton":
            (
                float(
                    WOB_ON_THRESHOLD
                )
                if wob_available
                else None
            ),

        "cycles":
            cycles,

        "global_metrics":
            metrics,
    }


# ============================================================
# API
# ============================================================

@app.route("/")
def index():

    return render_template(
        "index.html"
    )


# ============================================================
# W2W API
# ============================================================

@app.route(
    "/api/calculate/w2w",
    methods=["POST"]
)
def calculate_w2w_api():

    try:

        # --------------------------------------------------------
        # File
        # --------------------------------------------------------

        file = request.files.get(
            "file"
        )

        if file is None:

            return jsonify({

                "success": False,

                "error":
                    "No file uploaded."

            }), 400

        if file.filename == "":

            return jsonify({

                "success": False,

                "error":
                    "No file selected."

            }), 400

        filename = file.filename.lower()

        if not (
            filename.endswith(".xls")
            or
            filename.endswith(".xlsx")
        ):

            return jsonify({

                "success": False,

                "error":
                    "Only .xls and .xlsx files are supported."

            }), 400

        print("\n")
        print("=" * 80)

        print(
            "W2W CALCULATION"
        )

        print(
            f"File: {file.filename}"
        )

        print("=" * 80)

        # --------------------------------------------------------
        # Read Excel
        # --------------------------------------------------------

        try:

            raw_df = pd.read_excel(
                file
            )

        except ImportError as e:

            return jsonify({

                "success": False,

                "error":
                    "Excel engine missing. "
                    "Install xlrd for .xls and "
                    "openpyxl for .xlsx. "
                    f"Details: {str(e)}"

            }), 500

        # --------------------------------------------------------
        # Afficher colonnes
        # --------------------------------------------------------

        print("\nExcel columns:")

        for i, column in enumerate(
            raw_df.columns,
            start=1
        ):

            print(
                f"  {i:02d}. {column}"
            )

        # --------------------------------------------------------
        # Calcul
        # --------------------------------------------------------

        result = calculate_w2w(
            raw_df
        )

        # --------------------------------------------------------
        # JSON serialisable
        # --------------------------------------------------------

        result_json = json.loads(
            json.dumps(
                result,
                default=str
            )
        )

        # --------------------------------------------------------
        # Résumé serveur
        # --------------------------------------------------------

        metrics = (
            result_json[
                "global_metrics"
            ]
        )

        print("\n")
        print("=" * 80)

        print(
            "RESULT"
        )

        print("=" * 80)

        print(
            "Total cycles:",
            metrics[
                "total_cycles"
            ]
        )

        print(
            "Normal:",
            metrics[
                "normal_cycles"
            ]
        )

        print(
            "Abnormal:",
            metrics[
                "abnormal_cycles"
            ]
        )

        print(
            "Average W2W:",
            metrics[
                "avg_w2w_minutes"
            ]
        )

        print("=" * 80)

        # --------------------------------------------------------
        # Response
        # --------------------------------------------------------

        return jsonify({

            "success": True,

            "data": result_json

        })

    except Exception as e:

        traceback.print_exc()

        return jsonify({

            "success": False,

            "error": str(e),

            "details":
                traceback.format_exc()

        }), 500


# ============================================================
# HEALTH
# ============================================================

@app.route(
    "/api/health",
    methods=["GET"]
)
def health():

    return jsonify({

        "status": "ok",

        "module":
            "RTOM Weight to Weight",

    })


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    print("\n")
    print("=" * 80)

    print(
        "RTOM WEIGHT TO WEIGHT MODULE"
    )

    print(
        "Starting Flask server..."
    )

    print("=" * 80)

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=False,
    )
