import pandas as pd
import numpy as np
import glob, os, re
from pathlib import Path
from typing import Optional, List, Dict

# ========= Einstellungen =========
CSV_DIR   = r"F:\Masterarbeit\Daten Rotmoos\Final Alle\Kacheln Vorne"   # *_PebbleCountsAuto_CSV.csv
OUT_DIR   = r"F:\Masterarbeit\Daten Rotmoos\Final Alle\Kacheln Vorne"   # sedinet_*.csv Ziel
CSV_SUFFIX_TO_STRIP = "_PebbleCountsAuto_CSV"  # aus CSV-Basename entfernen
TRAIN_FRAC   = 0.8
RANDOM_SEED  = 42

# Längen-Spalten-Priorität (in Metern)
LENGTH_PRIORITY: List[str] = ["b (m)", "b_m", "bm", "a (m)", "a_m", "am"]
MIN_VALID_PER_KACHEL = 10
# =================================

def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", s.lower())

def robust_read_csv(csv_path: Path) -> Optional[pd.DataFrame]:
    for sep in [",",";","\t"]:
        try:
            df = pd.read_csv(csv_path, sep=sep)
            if df.shape[1] >= 3:
                return df
        except Exception:
            pass
    try:
        txt = csv_path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        txt = csv_path.read_text(errors="ignore")
    lines = txt.splitlines(True)
    header_idx = None
    for i, line in enumerate(lines):
        low = line.lower()
        if ("utm" in low and "(m)" in low) or ("a (m)" in low) or ("b (m)" in low):
            header_idx = i; break
    if header_idx is None:
        return None
    chunk = "".join(lines[header_idx:])
    from io import StringIO
    for sep in [",",";","\t"]:
        try:
            df = pd.read_csv(StringIO(chunk), sep=sep)
            if df.shape[1] >= 3:
                return df
        except Exception:
            continue
    return None

def pick_length_series_in_m(df: pd.DataFrame) -> Optional[pd.Series]:
    norm_map = {_norm(c): c for c in df.columns}
    for cand in LENGTH_PRIORITY:
        key = _norm(cand)
        if key in norm_map:
            col = norm_map[key]
            s = df[col].astype(str).str.replace(",", ".", regex=False)
            vals = pd.to_numeric(s, errors="coerce")
            if vals.notna().sum() >= MIN_VALID_PER_KACHEL:
                return vals
    return None

def csv_basename(csv_path: Path) -> str:
    base = csv_path.stem
    if base.endswith(CSV_SUFFIX_TO_STRIP):
        base = base[:-len(CSV_SUFFIX_TO_STRIP)]
    return base

def to_image_relpath_from_base(base: str) -> str:
    """images/<Basename>.tif"""
    return f"images/{base}.tif"

def collect_rows() -> pd.DataFrame:
    rows = []
    csv_files = sorted(glob.glob(os.path.join(CSV_DIR, "*.csv")))
    for fp in csv_files:
        p = Path(fp)
        df = robust_read_csv(p)
        if df is None or df.empty:
            continue
        length_m = pick_length_series_in_m(df)
        if length_m is None:
            continue

        # ---- D50 berechnen (Median in m -> mm) ----
        d50_m = float(np.nanpercentile(length_m.values, 50))
        d50_mm = d50_m * 1000.0

        base = csv_basename(p)
        rows.append({
            "base": base,
            "files": to_image_relpath_from_base(base),
            "P50": round(d50_mm, 3)
        })
    return pd.DataFrame(rows)

def assign_stable_ids(df: pd.DataFrame) -> pd.DataFrame:
    bases_sorted = sorted(df["base"].unique())
    id_map: Dict[str, int] = {b: i+1 for i, b in enumerate(bases_sorted)}
    df = df.copy()
    df["ID"] = df["base"].map(id_map)
    return df[["ID", "files", "P50", "base"]]

def write_with_header(df: pd.DataFrame, out_path: Path):
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(",files,P50\n")
        for r in df.itertuples(index=False):
            f.write(f"{r.ID},{r.files},{r.P50}\n")

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    out_dir = Path(OUT_DIR)

    df_all = collect_rows()
    if df_all.empty:
        print("Keine verwertbaren Kacheln gefunden."); return

    df_all = assign_stable_ids(df_all)
    df_train = df_all.sample(frac=TRAIN_FRAC, random_state=RANDOM_SEED)
    df_test  = df_all.drop(df_train.index)

    write_with_header(df_all.sort_values("ID"),  out_dir / "sedinet_all.csv")
    write_with_header(df_train.sort_values("ID"), out_dir / "sedinet_train.csv")
    write_with_header(df_test.sort_values("ID"),  out_dir / "sedinet_test.csv")

    print(f"✅ Fertig. Gesamt: {len(df_all)} | Train: {len(df_train)} | Test: {len(df_test)}")
    print(f"→ {out_dir/'sedinet_all.csv'}")
    print(f"→ {out_dir/'sedinet_train.csv'}")
    print(f"→ {out_dir/'sedinet_test.csv'}")

if __name__ == "__main__":
    main()
