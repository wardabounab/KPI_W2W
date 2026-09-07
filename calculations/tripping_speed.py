import pandas as pd
import numpy as np
import math
from typing import List

# ---------------------------------------------------------------------------
# Constants — update these to match your actual Excel column headers.
# ---------------------------------------------------------------------------

REQUIRED_COLUMNS = [
    "index",          # date time : dd/mm/yyyy hh:mm:ss
    "DBTM (m)",          
    "BPOS (m)",  
    "HKLA (N)"
]

# Sentinel / garbage values to treat as NaN
BAD_VALUES = {-999, -9999, -999.25, -999.99, -9999.25, -9999.99, -99999}

# Conversion factor: Newtons → tonnes-force (tonf)
N_TO_TONF = 1.0 / 9806.65

GAP_THRESHOLD = pd.Timedelta(seconds=6)

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def calculate_tripping_speed(df: pd.DataFrame) -> dict:

    # 1. Validate columns
    validate_columns(df, REQUIRED_COLUMNS)

    # 2. Select and copy only the columns we need
    df = df[REQUIRED_COLUMNS].copy()

    # 4. Drop fully empty rows
    before = len(df)
    df = preprocess_data(df)
    dropped = before - len(df)
    
    print(f"{dropped} row(s) were ignored.")
    print(df.head())

    gaps = detect_gaps(df)
    print_gap_report(gaps)

    hkla_threshold = find_hkla_threshold(df)

    return {
        'dropped_rows':dropped,
        "total_gap":  gaps['total_gap'],
        "total_span": gaps['total_span'],
        "gap_pct":    gaps['gap_pct'],
        "hkla_threshold": hkla_threshold
    }



# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def validate_columns(df: pd.DataFrame, required: List[str]) -> None:
    """Raise ValueError if any required column is absent (case-insensitive)."""
    actual = {c.strip().lower() for c in df.columns}
    missing = [col for col in required if col.lower() not in actual]
    if missing:
        raise ValueError(
            f"Missing required column(s): {missing}. "
            f"Found columns: {list(df.columns)}"
        )

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:

    data_columns = [c for c in REQUIRED_COLUMNS if c != "index"]

    df["index"] = pd.to_datetime(df["index"], dayfirst=True, errors="coerce")
    df = df.set_index("index")

    df = df.sort_index()


    # 4. Cast data columns to float64
    for col in data_columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")  # non-numeric → NaN

    # 5. Replace sentinel / bad values with NaN
    df[data_columns] = df[data_columns].where(
        ~df[data_columns].isin(BAD_VALUES), other=np.nan
    )

    # 6. Drop rows with any NaN / null (including unparseable timestamps)
    df = df[df.index.notna()]   # drop rows where datetime parsing failed
    df = df.dropna(subset=data_columns)
 
    # 7. Convert HKLA: N → tonf, rename column
    df["HKLA (tonf)"] = df["HKLA (N)"] * N_TO_TONF
    df = df.drop(columns=["HKLA (N)"])
 
    return df


def detect_gaps(df: pd.DataFrame, threshold: pd.Timedelta = GAP_THRESHOLD) -> dict:
    
    if len(df) < 2:
        raise ValueError("DataFrame must have at least 2 rows to detect gaps.")
 
    # Time difference between consecutive rows
    deltas = df.index.to_series().diff().iloc[1:]  # first value is NaT
 
    # Boolean mask of gap positions (index = the row AFTER the gap)
    gap_mask = deltas > threshold
 
    gaps_raw = []
    for gap_id, (end_ts, duration) in enumerate(
        deltas[gap_mask].items(), start=1
    ):
        # The row just before 'end_ts' in the index is the gap start
        pos = df.index.get_loc(end_ts)
        start_ts = df.index[pos - 1]
        gaps_raw.append({
            "gap_id":   gap_id,
            "start":    start_ts,
            "end":      end_ts,
            "duration": duration,
        })
 
    gaps_df = pd.DataFrame(
        gaps_raw,
        columns=["gap_id", "start", "end", "duration"]
    ) if gaps_raw else pd.DataFrame(columns=["gap_id", "start", "end", "duration"])
 
    total_gap  = gaps_df["duration"].sum() if not gaps_df.empty else pd.Timedelta(0)
    total_span = df.index[-1] - df.index[0]
    gap_pct    = (total_gap / total_span * 100) if total_span.total_seconds() > 0 else 0.0
 
    return {
        "gaps":       gaps_df,
        "total_gap":  total_gap,
        "total_span": total_span,
        "gap_pct":    round(gap_pct, 4),
        "n_gaps":     len(gaps_df),
    }
 
def find_hkla_threshold(df: pd.DataFrame) -> int:
    if "HKLA (tonf)" not in df.columns:
        raise KeyError("Column 'HKLA (tonf)' not found in dataframe.")
 
    hkla = df[["HKLA (tonf)"]].copy()
    hkla["HKLA (tonf)"] = hkla["HKLA (tonf)"].apply(math.ceil)
    hkla = hkla.sort_values("HKLA (tonf)").reset_index(drop=True)
 
    return int(hkla["HKLA (tonf)"].quantile(0.25))

 
def print_gap_report(result: dict) -> None:
    """Pretty-print the output of detect_gaps()."""
    print("=" * 55)
    print("GAP DETECTION REPORT")
    print("=" * 55)
    print(f"  Total span  : {result['total_span']}")
    print(f"  Gaps found  : {result['n_gaps']}")
    print(f"  Total gap   : {result['total_gap']}")
    print(f"  Gap coverage: {result['gap_pct']} %")
    print("-" * 55)
 
    if result["gaps"].empty:
        print("  No gaps detected.")
    else:
        print(f"  {'#':<5} {'Start':<25} {'End':<25} {'Duration'}")
        print(f"  {'-'*5} {'-'*24} {'-'*24} {'-'*15}")
        for _, row in result["gaps"].iterrows():
            print(
                f"  {int(row.gap_id):<5} "
                f"{str(row.start):<25} "
                f"{str(row.end):<25} "
                f"{row.duration}"
            )
    print("=" * 55)


# this part is only for testing the code to be removed when running the server
# run the scrpit with to test the logic python tripping_speed.py "C:\path\to\file.xlsx"
import sys
if __name__ == "__main__":
    filepath = sys.argv[1]
    df = pd.read_excel(filepath)
    res = calculate_tripping_speed(df)
    print(res)