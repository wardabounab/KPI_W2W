"""
RTOM - Weight to Weight (W2W) module  (Drill State)

Détection des connexions de forage à partir des signaux RTOM :
    WOB, HKLD, BPOS, DBTM, DMEA
Séquence : Pre-connection (weight -> slips)
         + Connection     (slips -> slips)
         + Post-connection(slips -> weight)
"""

import io
import logging
import math
import os
import sys
import time
import warnings
from datetime import datetime
from logging.handlers import RotatingFileHandler

import numpy as np
import pandas as pd

# ============================================================
# LOGGING
# ============================================================

DEBUG_W2W = os.environ.get("W2W_DEBUG", "0") == "1"
LOG_DIR = os.environ.get("W2W_LOG_DIR", "logs")
os.makedirs(LOG_DIR, exist_ok=True)

logger = logging.getLogger("w2w")
logger.setLevel(logging.DEBUG if DEBUG_W2W else logging.INFO)
if not logger.handlers:
    _fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    _fh = RotatingFileHandler(
        os.path.join(LOG_DIR, "w2w.log"),
        maxBytes=2_000_000, backupCount=5, encoding="utf-8",
    )
    _fh.setFormatter(_fmt)
    _sh = logging.StreamHandler(sys.stdout)
    _sh.setFormatter(_fmt)
    logger.addHandler(_fh)
    logger.addHandler(_sh)

# ============================================================
# CONFIGURATION
# ============================================================

DEFAULTS = {
    "slip_threshold": None,
    "wob_threshold": 1.0,
    "lower_bpos": 9.0,
    "upper_bpos": 9.0,
    "max_rise": 31.0,
    "rise_tol": 1.0,
    "stand_rise_min": 15.0,
    "bottom_tol": 0.5,
    "post_dbtm_tol": 0.5,
    "conn_dbtm_tol": 1.0,
    "return_tol": 1.0,
    "min_connection_s": 5,
    "min_pre_s": 30,
    "no_rise_s": 180,
    "max_pre_s": 120 * 60,
    "max_slip_s": 120 * 60,
    "max_post_s": 30 * 60,
    "min_w2w": 0.5,
    "max_w2w": 120.0,
    "benchmark": None,
    "long_factor": 2.0,
    "short_factor": 0.3,
    "gap_s": 600,
    "smoothing": 5,
    "bpos_floor": -1.0,
    "day_start_h": 6,
    "day_end_h": 18,
}

CONFIG_TYPES = {
    "slip_threshold": float, "wob_threshold": float, "lower_bpos": float,
    "upper_bpos": float, "max_rise": float, "benchmark": float,
    "min_w2w": float, "max_w2w": float,
}

BAD_VALUES = {-999, -9999, -999.25, -999.99, -9999.25, -9999.99, -99999}

REQUIRED = ["ts", "dmea", "dbtm", "bpos", "hkla"]

# Constantes de fin de post-connexion (évite les string-literals fragiles)
END_REASON_WOB = "WOB > threshold"
END_REASON_DBTM_DMEA = "DBTM = DMEA"
END_REASON_DBTM_BACK = "DBTM back at pre-connection depth"


def parse_config(form):
    cfg = dict(DEFAULTS)
    for key, caster in CONFIG_TYPES.items():
        raw = form.get(key)
        if raw is None or str(raw).strip() == "":
            continue
        try:
            cfg[key] = caster(str(raw).strip().replace(",", "."))
        except ValueError:
            raise ValueError(f"Invalid value for '{key}': {raw!r}")
    if cfg["upper_bpos"] > cfg["max_rise"]:
        raise ValueError("Upper BPOS limit cannot exceed the maximum rise.")
    if cfg["benchmark"] is not None and cfg["benchmark"] <= 0:
        cfg["benchmark"] = None
    return cfg


# ============================================================
# HELPERS
# ============================================================

def clean_column_name(column):
    return str(column).strip().replace("\n", " ").replace("\r", " ")


def to_native(o):
    if isinstance(o, dict):
        return {str(k): to_native(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [to_native(v) for v in o]
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, (np.floating, float)):
        f = float(o)
        return f if math.isfinite(f) else None
    if isinstance(o, (pd.Timestamp, datetime)):
        return o.isoformat()
    return o


def _to_numeric(series):
    if series.dtype == object:
        series = (
            series.astype(str).str.strip().str.replace(",", ".", regex=False)
        )
    return pd.to_numeric(series, errors="coerce")


def _parse_ts(series):
    if pd.api.types.is_datetime64_any_dtype(series):
        out = series
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out = pd.to_datetime(series, errors="coerce", dayfirst=True)
    if getattr(out.dt, "tz", None) is not None:
        out = out.dt.tz_localize(None)
    return out


# ============================================================
# DETECTION DES COLONNES
# ============================================================

TS_NAMES = {
    "index", "timestamp", "date time", "datetime", "date_time",
    "date-time", "time", "date", "heure",
}


def detect_columns(df):
    mapping = {}
    for col in df.columns:
        lc = col.lower().strip()

        if "ts" not in mapping and (
            lc in TS_NAMES or lc.startswith(("time", "date"))
        ):
            mapping["ts"] = col
        elif "dbtm" in lc or "bit depth" in lc:
            mapping.setdefault("dbtm", col)
        elif (
            "dmea" in lc or "measured depth" in lc or "hole depth" in lc
            or lc in ("depth(m)", "depth")
        ):
            mapping.setdefault("dmea", col)
        elif "bpos" in lc or "block position" in lc or "block pos" in lc:
            mapping.setdefault("bpos", col)
        elif (
            "hkla" in lc or "hkld" in lc or "hookload" in lc
            or "hook load" in lc
        ):
            mapping.setdefault("hkla", col)
        elif (
            "woba" in lc or lc == "wob" or "weight on bit" in lc
            or "weight-on-bit" in lc or "weight_on_bit" in lc
        ):
            mapping.setdefault("wob", col)
    return mapping


def _getcol(df, name):
    s = df[name]
    return s.iloc[:, 0] if isinstance(s, pd.DataFrame) else s


# ============================================================
# SEUIL HOOKLOAD AUTOMATIQUE (k-means 1D, 2 classes)
# ============================================================

def compute_slip_threshold(series):
    values = series.dropna().to_numpy(dtype=float)
    if len(values) == 0:
        return 0.0, None, None
    if len(values) < 10:
        return float(np.median(values)), None, None

    lo_clip, hi_clip = np.percentile(values, [1, 99])
    values = values[(values >= lo_clip) & (values <= hi_clip)]

    # Garde-fou : après clipping, il faut au moins 2 points
    if len(values) < 2:
        return float(np.median(series.dropna())), None, None

    low = float(np.percentile(values, 25))
    high = float(np.percentile(values, 75))

    for _ in range(50):
        d_low = np.abs(values - low)
        d_high = np.abs(values - high)
        low_g = values[d_low <= d_high]
        high_g = values[d_high < d_low]
        if len(low_g) == 0 or len(high_g) == 0:
            break
        new_low, new_high = float(low_g.mean()), float(high_g.mean())
        if abs(new_low - low) < 1e-3 and abs(new_high - high) < 1e-3:
            low, high = new_low, new_high
            break
        low, high = new_low, new_high

    return (low + high) / 2.0, low, high


# ============================================================
# PREPROCESSING + CONTROLE QUALITE
# ============================================================

def preprocess_data(raw_df, cfg):
    df0 = raw_df.copy()
    df0.columns = [clean_column_name(c) for c in df0.columns]
    mapping = detect_columns(df0)

    if "ts" not in mapping and len(df0.columns):
        first = df0.columns[0]
        if _parse_ts(_getcol(df0, first)).notna().mean() > 0.5:
            mapping["ts"] = first

    missing = [k for k in REQUIRED if k not in mapping]
    if missing:
        raise ValueError(
            f"Missing required column(s): {missing}. "
            f"Found columns: {list(df0.columns)}"
        )

    quality = {
        "columns_detected": {k: str(v) for k, v in mapping.items()},
        "input_rows": int(len(df0)),
        "warnings": [],
    }

    df = pd.DataFrame({k: _getcol(df0, v) for k, v in mapping.items()})

    df["ts"] = _parse_ts(df["ts"])
    quality["invalid_timestamps"] = int(df["ts"].isna().sum())

    num_cols = ["dmea", "dbtm", "bpos", "hkla"]
    if "wob" in df.columns:
        num_cols.append("wob")
    for c in num_cols:
        df[c] = _to_numeric(df[c])

    quality["bad_values_replaced"] = int(
        df[num_cols].isin(list(BAD_VALUES)).to_numpy().sum()
    )
    df[num_cols] = df[num_cols].replace(list(BAD_VALUES), np.nan)

    wob_available = False
    if "wob" in df.columns:
        if df["wob"].notna().mean() >= 0.2:
            wob_available = True
        else:
            quality["warnings"].append(
                "WOB column found but mostly empty: ignored."
            )
            df = df.drop(columns=["wob"])
            num_cols.remove("wob")
    else:
        quality["warnings"].append(
            "WOB column not found: pre-connection start uses BPOS/HKLD only."
        )
    quality["wob_available"] = wob_available

    quality["missing_values"] = {
        c: int(df[c].isna().sum()) for c in ["ts"] + num_cols
    }

    before = len(df)
    df = df.dropna(subset=REQUIRED)
    quality["rows_dropped_invalid"] = int(before - len(df))
    if len(df) == 0:
        raise ValueError("No valid rows after preprocessing.")

    df = df.sort_values("ts").reset_index(drop=True)

    dup = df["ts"].duplicated(keep="first")
    quality["duplicate_timestamps"] = int(dup.sum())
    df = df[~dup].set_index("ts")

    if wob_available:
        df["wob"] = df["wob"].ffill(limit=3)

    dt = df.index.to_series().diff()
    thr = pd.Timedelta(seconds=cfg["gap_s"])
    df["large_gap"] = (dt > thr).to_numpy()
    gaps = []
    for end, delta in dt[dt > thr].items():
        gaps.append({
            "start": (end - delta).isoformat(),
            "end": end.isoformat(),
            "minutes": float(delta.total_seconds() / 60.0),
        })
    quality["data_gap_count"] = len(gaps)
    quality["data_gaps"] = gaps[:100]
    med = dt.dropna().dt.total_seconds().median()
    quality["median_sampling_s"] = float(med) if pd.notna(med) else None

    quality["inconsistent_values"] = {
        "dbtm_greater_than_dmea": int((df["dbtm"] > df["dmea"] + 1.0).sum()),
        "bpos_below_floor": int((df["bpos"] < cfg["bpos_floor"]).sum()),
        "negative_hookload": int((df["hkla"] < 0).sum()),
    }
    if quality["inconsistent_values"]["bpos_below_floor"]:
        quality["warnings"].append(
            "BPOS values below rig floor: check sensor calibration / zero."
        )

    if cfg["slip_threshold"] is not None:
        slip_threshold, low, high = float(cfg["slip_threshold"]), None, None
        source = "manual"
    else:
        slip_threshold, low, high = compute_slip_threshold(df["hkla"])
        source = "auto"
        if low is not None and high is not None and (high - low) < 0.2 * abs(high):
            quality["warnings"].append(
                "Hookload shows no clear 'in slips' / 'out of slips' "
                "separation: set the slip threshold manually."
            )
    quality["slip_threshold_source"] = source
    quality["hookload_clusters"] = [low, high]

    logger.info(
        "Preprocessed: %d -> %d rows | slip threshold %.3f (%s) | WOB=%s | gaps=%d",
        len(raw_df), len(df), slip_threshold, source, wob_available, len(gaps),
    )
    return df, slip_threshold, wob_available, quality


# ============================================================
# DETECTION W2W
# ============================================================

def detect_w2w_cycles(df, cfg, slip_threshold, wob_available):
    incomplete = []
    n = len(df)
    if n < 10:
        return [], incomplete

    idx = df.index
    t = idx.values.astype("datetime64[ns]").astype("int64") / 1e9
    k = max(1, int(cfg["smoothing"]))

    def smooth(col):
        return (
            df[col].rolling(k, center=True, min_periods=1)
            .median().to_numpy(dtype=float)
        )

    dbtm, dmea, bpos, hkla = (smooth(c) for c in ("dbtm", "dmea", "bpos", "hkla"))
    wob = smooth("wob") if wob_available else None
    gap = df["large_gap"].to_numpy(dtype=bool)

    block_up = np.diff(bpos, prepend=bpos[0]) > 0.03
    in_slips = hkla <= slip_threshold

    lower = cfg["lower_bpos"]
    min_rise = cfg["upper_bpos"]
    max_rise = cfg["max_rise"]
    wob_thr = cfg["wob_threshold"]
    bench = cfg["benchmark"]

    DRILLING, PRE, IN_SLIPS, POST = range(4)
    state = DRILLING
    cycles = []
    pre_i = det_i = slip_in = slip_out = None
    peak = 0.0
    hk_drop = False
    last_end = -1

    def log_inc(reason, i, start=None):
        start = pre_i if start is None else start
        incomplete.append({
            "reason": reason,
            "start": idx[start].isoformat() if start is not None else None,
            "at": idx[i].isoformat(),
            "depth_dmea_m": float(dmea[start]) if start is not None else None,
        })
        logger.debug("Incomplete connection: %s at %s", reason, idx[i])

    def make_cycle(number, end_i, end_reason):
        pre_min = (t[slip_in] - t[pre_i]) / 60.0
        conn_min = (t[slip_out] - t[slip_in]) / 60.0
        post_min = (t[end_i] - t[slip_out]) / 60.0
        w2w = pre_min + conn_min + post_min
        rise = float(peak - bpos[pre_i])
        dbtm_change = float(np.ptp(dbtm[slip_in:slip_out + 1]))

        reasons, warns = [], []

        if w2w > cfg["max_w2w"] or (bench and w2w > bench * cfg["long_factor"]):
            reasons.append("Abnormally long connection")
        if w2w < cfg["min_w2w"] or (bench and w2w < bench * cfg["short_factor"]):
            reasons.append("Abnormally short connection")
        if hk_drop:
            reasons.append("Hook load drop during post-connection")
        if wob is not None and end_reason != END_REASON_WOB:
            tail = wob[end_i:min(n, end_i + 12)]
            if not np.any(tail > wob_thr):
                reasons.append("No WOB at end of post-connection")
        if np.nanmin(bpos[pre_i:end_i + 1]) < cfg["bpos_floor"]:
            reasons.append("BPOS below rig floor")

        if dbtm_change > cfg["conn_dbtm_tol"]:
            warns.append(f"DBTM varied {dbtm_change:.2f} m while in slips")
        if peak < min_rise:
            warns.append("Upper BPOS limit not reached")
        if end_reason != END_REASON_WOB and wob is None:
            warns.append("End of post-connection inferred without WOB")

        h = idx[slip_in].hour
        shift = "Day" if cfg["day_start_h"] <= h < cfg["day_end_h"] else "Night"

        return {
            "cycle_number": number,
            "connection_type": "stand" if rise >= cfg["stand_rise_min"] else "joint",
            "date": idx[slip_in].date().isoformat(),
            "shift": shift,
            "depth_dmea_m": float(dmea[pre_i]),
            "date_pre_start": idx[pre_i].isoformat(),
            "date_slip_in": idx[slip_in].isoformat(),
            "date_slip_out": idx[slip_out].isoformat(),
            "date_post_end": idx[end_i].isoformat(),
            "pre_connection_minutes": float(pre_min),
            "connection_minutes": float(conn_min),
            "post_connection_minutes": float(post_min),
            "w2w_minutes": float(w2w),
            "pre_bpos_change_m": rise,
            "conn_dbtm_change_m": dbtm_change,
            "end_reason": end_reason,
            "above_benchmark": bool(bench and w2w > bench),
            "benchmark_delta_min": float(w2w - bench) if bench else None,
            "abnormal": bool(reasons),
            "abnormal_reasons": reasons,
            "warnings": warns,
        }

    for i in range(1, n):

        # ---------------- trou de données ----------------
        if gap[i]:
            if state in (IN_SLIPS, POST) or (
                state == PRE and peak - bpos[pre_i] >= min_rise
            ):
                log_inc("Data gap during connection", i)
            state = DRILLING
            last_end = i
            continue

        # ---------------- DRILLING -> PRE ----------------
        if state == DRILLING:
            wob_off = wob is None or wob[i] <= wob_thr
            if (
                block_up[i] and bpos[i] <= lower
                and hkla[i] > slip_threshold and wob_off
            ):
                start = i
                if wob is not None:
                    while (
                        start - 1 > last_end
                        and wob[start - 1] <= wob_thr
                        and t[i] - t[start - 1] <= 300
                    ):
                        start -= 1
                pre_i, det_i = start, i
                peak = bpos[i]
                hk_drop = False          # reset à chaque nouveau candidat
                state = PRE
            continue

        # ---------------- PRE -> IN_SLIPS ----------------
        if state == PRE:
            peak = max(peak, bpos[i])
            rise = peak - bpos[pre_i]
            since = t[i] - t[det_i]
            elapsed = t[i] - t[pre_i]
            risen = rise > cfg["return_tol"]

            if in_slips[i] and rise >= min_rise:
                if rise <= max_rise + cfg["rise_tol"]:
                    slip_in = i
                    state = IN_SLIPS
                else:
                    log_inc(
                        f"BPOS rise {rise:.1f} m exceeds maximum {max_rise:.0f} m", i
                    )
                    state, last_end = DRILLING, i
                continue

            if elapsed > cfg["max_pre_s"]:
                if rise >= min_rise:
                    log_inc("Pre-connection timeout: slip-in never detected", i)
                state, last_end = DRILLING, i
                continue

            drilling_back = (
                wob is not None and wob[i] > wob_thr and since >= cfg["min_pre_s"]
            )
            returned = (
                risen and bpos[i] <= bpos[pre_i] + cfg["return_tol"]
                and since >= cfg["min_pre_s"]
            )
            stalled = (not risen) and since > cfg["no_rise_s"]
            if rise < min_rise and (drilling_back or returned or stalled):
                state, last_end = DRILLING, i
            continue

        # ---------------- IN_SLIPS -> POST ----------------
        if state == IN_SLIPS:
            peak = max(peak, bpos[i])
            dur = t[i] - t[slip_in]
            if (not in_slips[i]) and dur >= cfg["min_connection_s"]:
                slip_out = i
                hk_drop = False
                state = POST
            elif dur > cfg["max_slip_s"]:
                log_inc("Connection exceeds maximum duration in slips", i)
                state, last_end = DRILLING, i
            continue

        # ---------------- POST -> fin ----------------
        if state == POST:
            post_s = t[i] - t[slip_out]
            if in_slips[i] and post_s >= 10:
                hk_drop = True

            ref_dbtm = dbtm[pre_i]
            reason = None
            if wob is not None and wob[i] > wob_thr:
                reason = END_REASON_WOB
            elif dmea[i] - dbtm[i] <= cfg["bottom_tol"]:
                reason = END_REASON_DBTM_DMEA
            elif abs(dbtm[i] - ref_dbtm) <= cfg["post_dbtm_tol"]:
                reason = END_REASON_DBTM_BACK

            if reason:
                cyc = make_cycle(len(cycles) + 1, i, reason)
                cycles.append(cyc)
                logger.debug(
                    "CYCLE #%d W2W=%.2f min conn=%.2f min",
                    cyc["cycle_number"], cyc["w2w_minutes"], cyc["connection_minutes"],
                )
                state, last_end = DRILLING, i
            elif post_s > cfg["max_post_s"]:
                log_inc("Post-connection timeout: bottom never reached", i)
                state, last_end = DRILLING, i

    logger.info(
        "Detection finished: %d cycles, %d incomplete", len(cycles), len(incomplete)
    )
    return cycles, incomplete


# ============================================================
# METRIQUES
# ============================================================

def is_included(c):
    return not c.get("excluded", c.get("abnormal", False))


def find_degradation(included, window=5, factor=1.3):
    if len(included) < window + 2:
        return []
    w = np.array([c["w2w_minutes"] for c in included], dtype=float)
    ref = float(np.median(w))
    roll = pd.Series(w).rolling(window).mean().to_numpy()
    flag = np.nan_to_num(roll, nan=0.0) > ref * factor
    periods, i = [], 0
    while i < len(w):
        if flag[i]:
            j = i
            while j + 1 < len(w) and flag[j + 1]:
                j += 1
            s = max(0, i - window + 1)
            periods.append({
                "from_cycle": included[s]["cycle_number"],
                "to_cycle": included[j]["cycle_number"],
                "from": included[s]["date_slip_in"],
                "to": included[j]["date_post_end"],
                "avg_w2w_minutes": float(w[s:j + 1].mean()),
                "reference_minutes": ref,
            })
            i = j + 1
        else:
            i += 1
    return periods


def calculate_global_metrics(cycles):
    empty = {
        "total_cycles": 0, "normal_cycles": 0, "abnormal_cycles": 0,
        "excluded_cycles": 0, "avg_w2w_minutes": None, "median_w2w_minutes": None,
        "avg_pre_connection_minutes": None, "avg_connection_minutes": None,
        "avg_post_connection_minutes": None, "total_w2w_minutes": 0,
        "min_w2w_minutes": None, "max_w2w_minutes": None,
        "min_connection_minutes": None, "max_connection_minutes": None,
        "best_connection_index": None, "worst_connection_index": None,
        "slowest_connections": [], "joint_count": 0, "stand_count": 0,
        "degradation_periods": [], "histogram": {"edges": [], "counts": []},
    }
    if not cycles:
        return empty

    inc = [c for c in cycles if is_included(c)]
    out = dict(empty)
    out.update({
        "total_cycles": len(cycles),
        "normal_cycles": sum(1 for c in cycles if not c["abnormal"]),
        "abnormal_cycles": sum(1 for c in cycles if c["abnormal"]),
        "excluded_cycles": len(cycles) - len(inc),
    })
    if not inc:
        return out

    def col(key):
        return np.array([c[key] for c in inc], dtype=float)

    w2w, conn = col("w2w_minutes"), col("connection_minutes")
    out.update({
        "avg_w2w_minutes": float(w2w.mean()),
        "median_w2w_minutes": float(np.median(w2w)),
        "avg_pre_connection_minutes": float(col("pre_connection_minutes").mean()),
        "avg_connection_minutes": float(conn.mean()),
        "avg_post_connection_minutes": float(col("post_connection_minutes").mean()),
        "total_w2w_minutes": float(w2w.sum()),
        "min_w2w_minutes": float(w2w.min()),
        "max_w2w_minutes": float(w2w.max()),
        "min_connection_minutes": float(conn.min()),
        "max_connection_minutes": float(conn.max()),
        "best_connection_index": inc[int(np.argmin(w2w))]["cycle_number"],
        "worst_connection_index": inc[int(np.argmax(w2w))]["cycle_number"],
        "slowest_connections": [
            {"cycle_number": c["cycle_number"], "w2w_minutes": c["w2w_minutes"]}
            for c in sorted(inc, key=lambda x: -x["w2w_minutes"])[:5]
        ],
        "joint_count": sum(1 for c in inc if c["connection_type"] == "joint"),
        "stand_count": sum(1 for c in inc if c["connection_type"] == "stand"),
        "degradation_periods": find_degradation(inc),
    })
    bins = int(min(12, max(4, math.ceil(math.sqrt(len(w2w))))))
    counts, edges = np.histogram(w2w, bins=bins)
    out["histogram"] = {"edges": edges.tolist(), "counts": counts.tolist()}
    return out


def daily_summary(cycles):
    days = {}
    for c in cycles:
        days.setdefault(c.get("date") or c["date_slip_in"][:10], []).append(c)
    rows = []
    for day in sorted(days):
        cs = days[day]
        inc = [c for c in cs if is_included(c)]
        w = [c["w2w_minutes"] for c in inc]
        rows.append({
            "date": day,
            "connections": len(cs),
            "included": len(inc),
            "avg_w2w_minutes": float(np.mean(w)) if w else None,
            "min_w2w_minutes": float(min(w)) if w else None,
            "max_w2w_minutes": float(max(w)) if w else None,
            "avg_connection_minutes": (
                float(np.mean([c["connection_minutes"] for c in inc])) if inc else None
            ),
        })
    return rows


# ============================================================
# CALCUL PRINCIPAL
# ============================================================

def calculate_w2w(raw_df, cfg=None):
    cfg = dict(DEFAULTS) if cfg is None else {**DEFAULTS, **cfg}
    t0 = time.time()
    df, slip_threshold, wob_available, quality = preprocess_data(raw_df, cfg)
    cycles, incomplete = detect_w2w_cycles(df, cfg, slip_threshold, wob_available)

    if not cycles:
        quality["warnings"].append(
            "No connection detected: check slip threshold and BPOS limits."
        )

    result = {
        "input_rows": int(len(raw_df)),
        "clean_rows": int(len(df)),
        "dropped_rows": int(len(raw_df) - len(df)),
        "wob_available": bool(wob_available),
        "slip_threshold_ton": float(slip_threshold),
        "wob_threshold_ton": float(cfg["wob_threshold"]) if wob_available else None,
        "config": {k: cfg[k] for k in (
            "lower_bpos", "upper_bpos", "max_rise", "benchmark",
            "max_w2w", "min_w2w", "long_factor", "short_factor",
        )},
        "period": {
            "start": df.index[0].isoformat(), "end": df.index[-1].isoformat(),
        },
        "cycles": cycles,
        "incomplete_connections": incomplete,
        "quality": quality,
        "global_metrics": calculate_global_metrics(cycles),
        "daily_summary": daily_summary(cycles),
    }
    result["elapsed_seconds"] = round(time.time() - t0, 3)
    return result


# ============================================================
# EXPORTS (CSV / Excel)
# ============================================================

CYCLE_COLUMNS = [
    ("cycle_number", "Conn No."), ("date", "Date"), ("shift", "Shift"),
    ("connection_type", "Type"), ("depth_dmea_m", "Depth (m)"),
    ("date_pre_start", "Pre start"), ("date_slip_in", "Slip in"),
    ("date_slip_out", "Slip out"), ("date_post_end", "Post end"),
    ("pre_connection_minutes", "Pre-conn (min)"),
    ("connection_minutes", "Conn (min)"),
    ("post_connection_minutes", "Post-conn (min)"),
    ("w2w_minutes", "W2W (min)"), ("pre_bpos_change_m", "BPOS rise (m)"),
    ("conn_dbtm_change_m", "DBTM change (m)"), ("benchmark_delta_min", "vs benchmark (min)"),
    ("end_reason", "Post end reason"), ("status", "Status"), ("comment", "Comment"),
]


def cycles_frame(cycles):
    rows = []
    for c in cycles:
        r = dict(c)
        r["status"] = "Excluded" if not is_included(c) else "Included"
        if c.get("abnormal"):
            r["status"] = "Abnormal" + (" (excluded)" if not is_included(c) else " (included)")
        r["comment"] = "; ".join(
            list(c.get("abnormal_reasons", [])) + list(c.get("warnings", []))
        )
        rows.append(r)
    df = pd.DataFrame(rows)
    for key, _ in CYCLE_COLUMNS:
        if key not in df.columns:
            df[key] = None
    return df[[k for k, _ in CYCLE_COLUMNS]].rename(columns=dict(CYCLE_COLUMNS))


def build_export(fmt, data, meta):
    """Retourne (bytes, mimetype, nom_de_fichier). Lève ValueError si rien à exporter."""
    cycles = (data or {}).get("cycles") or []
    if not cycles:
        raise ValueError("No connection to export.")
    meta = meta or {}
    metrics = calculate_global_metrics(cycles)
    daily = daily_summary(cycles)
    frame = cycles_frame(cycles)
    stamp = datetime.now().strftime("%Y%m%d_%H%M")
    well = "".join(
        ch for ch in str(meta.get("Well name", "well")) if ch.isalnum() or ch in "-_"
    ) or "well"

    if fmt == "csv":
        return (
            frame.to_csv(index=False).encode("utf-8-sig"),
            "text/csv", f"W2W_{well}_{stamp}.csv",
        )
    if fmt != "xlsx":
        raise ValueError("Unsupported export format.")

    summary = [("Generated on", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))]
    summary += [(k, v if v not in (None, "") else "N/A") for k, v in meta.items()]
    summary += [("", "")]
    for k, v in metrics.items():
        if not isinstance(v, (list, dict)):
            summary.append((k, "N/A" if v is None else v))
    q = (data or {}).get("quality") or {}
    quality_rows = [
        (k, str(v)) for k, v in q.items()
        if k != "data_gaps" and not isinstance(v, (list, dict))
    ]
    quality_rows += [("warning", w) for w in q.get("warnings", [])]
    quality_rows += [
        ("incomplete connection", f'{i.get("at")} - {i.get("reason")}')
        for i in (data or {}).get("incomplete_connections", [])
    ]

    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as xw:
        pd.DataFrame(summary, columns=["Field", "Value"]).to_excel(xw, sheet_name="Summary", index=False)
        frame.to_excel(xw, sheet_name="Connections", index=False)
        pd.DataFrame(daily).to_excel(xw, sheet_name="Daily", index=False)
        pd.DataFrame(quality_rows, columns=["Check", "Value"]).to_excel(xw, sheet_name="Data quality", index=False)
        for ws in xw.book.worksheets:
            for column in ws.columns:
                width = max(len(str(c.value)) if c.value is not None else 0 for c in column)
                ws.column_dimensions[column[0].column_letter].width = min(max(width + 2, 10), 48)
    return (
        buf.getvalue(),
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        f"W2W_{well}_{stamp}.xlsx",
    )