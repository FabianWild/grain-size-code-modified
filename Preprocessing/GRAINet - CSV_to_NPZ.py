# -*- coding: utf-8 -*-
"""
PebbleCounts-CSV(s) -> GrainNet-NPZ

Modi:
- per_stone_spike:   ein Patch pro Stein (Spike-CDF), dm= b(cm)
- per_csv_distribution: ein Patch pro CSV  (CDF aller b),  dm= D50(cm)
- per_window_d50:    Fenster/Stride über dem Ortho; pro Fenster CDF & D50 aus allen Steinen im Fenster

Ausgabe-Keys:
  images:     (N,H,W,3) uint8
  dm:         (N,) float32               # in cm
  histograms: (N,22) float32             # kumulative CDF (22 Knoten -> 21 Klassen)
  tile_names: (N,) object                # Info/Debug

Voraussetzungen:
- Ortho ist georeferenziert (UTM), CSVs enthalten UTM X (m), UTM Y (m).
- GDAL/OSGeo in Env verfügbar.
"""

# ========= USER SETTINGS =========
ORTHO_PATH     = r"F:\Masterarbeit\Daten Rotmoos\Final Alle\Spot Vorn.tif"   # dein großes Ortho
CSV_INPUT      = r"F:\Masterarbeit\Daten Rotmoos\Final Alle\Kacheln Vorne\pebblecounts_merged_FINAL.csv"  # Datei ODER Ordner
IS_FOLDER      = False   # True: CSV_INPUT ist ein Ordner (alle *.csv einlesen); False: eine einzelne CSV

OUT_NPZ_PATH   = r"F:\Masterarbeit\Daten Rotmoos\Final Alle\Vorne_224.npz"

SAMPLE_MODE    = "per_window_d50"   # "per_window_d50" | "per_stone_spike" | "per_csv_distribution"

# CRS:
CSV_EPSG       = 32632     # anpassen falls Zone 33 -> 32633
FORCE_SAME_CRS = True     # auf True setzen, wenn CSV & Ortho sicher gleiches CRS

# Fenster/Stride (nur für per_window_d50)
PATCH_SIZE     = 224       # Pixel
STRIDE         = 64        # Pixel (typisch 32-64)
BANDS          = (1,2,3)   # RGB-Bänder im Ortho

# Aggregation (nur per_window_d50)
MIN_STONES     = 10        # Mindestanzahl Steine im Fenster, sonst Sample verwerfen

# Deduplizierung (per_stone_spike)
DEDUP_BY_TILE  = True
DEDUP_BY_HASH  = True

# Klassen-/Kantendefinition (Meter) – 22 Knoten wie im GRAINet-Setup
import numpy as np
EDGES_M = np.array(
    [0.00, 0.01, 0.02, 0.03, 0.04, 0.06, 0.08, 0.10, 0.12, 0.15,
     0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.60, 0.80, 1.00, 1.20, 1.50, 2.00],
    dtype=np.float32
)
# =================================

import os, io, glob, math, hashlib
import pandas as pd
from pathlib import Path
from osgeo import gdal, ogr, osr

gdal.UseExceptions()

def ensure_gdal_env():
    # Optional: falls nötig, Pfade setzen (hier auskommentiert – bei Bedarf anpassen)
    # env = r"F:\Anaconda3\envs\grainenv"
    # os.environ.setdefault("PROJ_LIB",  rf"{env}\Library\share\proj")
    # os.environ.setdefault("GDAL_DATA",  rf"{env}\Library\share\gdal")
    pass

def inv_gt(gt):
    if gt is None or len(gt) != 6:
        raise RuntimeError("Dataset hat keine gültige GeoTransform (affin).")
    res = gdal.InvGeoTransform(gt)
    # GDAL-Python ist uneinheitlich: manchmal (ok, inv), manchmal nur inv
    if isinstance(res, tuple):
        # Fall 1: (ok, inv)
        if len(res) == 2 and isinstance(res[0], (bool, int)) and hasattr(res[1], "__len__"):
            ok, inv = res
            if not ok:
                raise RuntimeError("InvGeoTransform() fehlgeschlagen.")
            return inv
        # Fall 2: direkt die 6 Werte zurückgegeben
        if len(res) == 6:
            return res
    # manche Wrapper geben eine Liste zurück
    if isinstance(res, list) and len(res) == 6:
        return tuple(res)
    raise RuntimeError(f"Unerwartete Rückgabe von InvGeoTransform: {type(res)} / {res}")

def map_to_pix(inv_gt, x, y):
    col, row = gdal.ApplyGeoTransform(inv_gt, x, y)
    return int(np.floor(col)), int(np.floor(row))

def read_patch_rgb(ds, r0, c0, size, bands=(1,2,3)):
    tiles = []
    for b in bands:
        arr = ds.GetRasterBand(int(b)).ReadAsArray(c0, r0, size, size)
        if arr is None:
            return None
        tiles.append(arr)
    patch = np.stack(tiles, axis=-1)
    # auf uint8 bringen
    if patch.dtype != np.uint8:
        if np.issubdtype(patch.dtype, np.floating):
            scale = 255.0 if patch.max() <= 1.0 else 1.0
            patch = np.clip(patch * scale, 0, 255).astype(np.uint8)
        else:
            patch = np.clip(patch, 0, 255).astype(np.uint8)
    return patch

def img_sig(arr: np.ndarray) -> str:
    return hashlib.blake2b(arr.tobytes(), digest_size=16).hexdigest()

def one_spike_cdf(length_m: float) -> np.ndarray:
    # 22 Werte: [0, CDF_1, ..., CDF_21]
    idx = np.clip(np.searchsorted(EDGES_M[1:], float(length_m), side="right"), 0, len(EDGES_M)-2)
    pdf = np.zeros(len(EDGES_M)-1, dtype=np.float32)
    pdf[idx] = 1.0
    return np.concatenate([[0.0], np.cumsum(pdf)]).astype(np.float32)

def cdf_from_values_m(vals_m: np.ndarray) -> np.ndarray:
    x = np.asarray(vals_m, dtype=np.float32)
    x = x[np.isfinite(x) & (x >= 0)]
    if x.size == 0:
        return None
    counts, _ = np.histogram(x, bins=EDGES_M)
    return np.concatenate([[0.0], np.cumsum(counts)/max(1, counts.sum())]).astype(np.float32)

def pctl(arr, q):
    try:    return float(np.percentile(arr, q, method="linear"))
    except: return float(np.percentile(arr, q, interpolation="linear"))

def detect_delimiter(path):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        head = "".join([next(f) for _ in range(2)])
    return ";" if head.count(";") > head.count(",") else ","

def read_pebble_csv(path: str) -> pd.DataFrame:
    delim = detect_delimiter(path)
    df = pd.read_csv(path, delimiter=delim, encoding="utf-8", engine="python")
    # weiche Normalisierung der Spaltennamen
    cols = {c.strip(): c for c in df.columns}
    need = ["UTM X (m)", "UTM Y (m)", "b (m)"]
    for n in need:
        if n not in cols:
            # Versuch: toleranter Match
            lc = [c for c in df.columns if n.replace(" ", "").lower() == c.replace(" ", "").lower()]
            if lc:
                cols[n] = lc[0]
    missing = [n for n in need if n not in cols]
    if missing:
        raise KeyError(f"{Path(path).name}: Spalten fehlen: {missing}")
    out = df[[cols["UTM X (m)"], cols["UTM Y (m)"], cols["b (m)"]]].copy()
    out.columns = ["UTM X (m)", "UTM Y (m)", "b (m)"]
    # Strings -> float
    for c in out.columns:
        if out[c].dtype == object:
            out[c] = (out[c].astype(str)
                            .str.replace(",", ".", regex=False)
                            .str.replace("\u00a0","", regex=False)
                            .str.strip())
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out.dropna()

def load_all_csvs(csv_input: str, is_folder: bool):
    files = []
    if is_folder:
        files = sorted(glob.glob(os.path.join(csv_input, "*.csv")))
    else:
        files = [csv_input]
    dfs = []
    for fp in files:
        if not os.path.isfile(fp): continue
        try:
            d = read_pebble_csv(fp)
            if len(d):
                d["__source__"] = os.path.basename(fp)
                dfs.append(d)
        except Exception as e:
            print("WARN CSV skip:", os.path.basename(fp), "|", e)
    if not dfs:
        raise RuntimeError("Keine gültigen CSVs gefunden.")
    big = pd.concat(dfs, ignore_index=True)
    return big

def make_transform(srs_src: osr.SpatialReference, srs_dst: osr.SpatialReference):
    try:
        if srs_src and srs_dst and srs_src.IsSame(srs_dst):
            return None
    except Exception:
        pass
    try:
        return osr.CoordinateTransformation(srs_src, srs_dst)
    except Exception:
        return None

def main():
    ensure_gdal_env()

    # Ortho öffnen
    rds = gdal.Open(ORTHO_PATH, gdal.GA_ReadOnly)
    if rds is None:
        raise FileNotFoundError(f"Raster nicht gefunden: {ORTHO_PATH}")
    W, H = rds.RasterXSize, rds.RasterYSize
    if rds.RasterCount < max(BANDS):
        raise RuntimeError(f"Raster hat nur {rds.RasterCount} Bänder, BANDS={BANDS} nicht möglich.")
    gt = rds.GetGeoTransform()
    inv = inv_gt(gt)
    srs_r = osr.SpatialReference(); srs_r.ImportFromWkt(rds.GetProjection())

    # CSV(s) laden
    df = load_all_csvs(CSV_INPUT, IS_FOLDER)
    srs_c = osr.SpatialReference(); srs_c.ImportFromEPSG(int(CSV_EPSG))
    x = df["UTM X (m)"].to_numpy(np.float64)
    y = df["UTM Y (m)"].to_numpy(np.float64)
    b = df["b (m)"].to_numpy(np.float32)

    # Reprojektion wenn nötig
    do_tr = None
    if not FORCE_SAME_CRS:
        do_tr = make_transform(srs_c, srs_r)
        if do_tr is None and not srs_c.IsSame(srs_r):
            raise RuntimeError("CRS der CSV unterscheidet sich, aber keine Transformation möglich. PROJ/GDAL prüfen oder FORCE_SAME_CRS=True setzen.")

    def to_raster_xy(ix):
        X, Y = float(x[ix]), float(y[ix])
        if do_tr is not None and not FORCE_SAME_CRS:
            pt = ogr.Geometry(ogr.wkbPoint); pt.AddPoint(X, Y)
            pt.Transform(do_tr)
            X, Y = pt.GetX(), pt.GetY()
        return X, Y

    images, dm_cm, cdfs, names = [], [], [], []

    if SAMPLE_MODE == "per_stone_spike":
        seen_tiles, seen_hash = set(), set()
        kept = dup_tile = dup_hash = oob = bad = 0
        for i in range(len(df)):
            try:
                Lm = float(b[i])
                if not np.isfinite(Lm) or Lm < 0: continue
                X, Y = to_raster_xy(i)
                c, r = map_to_pix(inv, X, Y)
                r0 = int(np.clip(r - PATCH_SIZE//2, 0, H-PATCH_SIZE))
                c0 = int(np.clip(c - PATCH_SIZE//2, 0, W-PATCH_SIZE))
                if r0 < 0 or c0 < 0: continue

                if DEDUP_BY_TILE:
                    key = (r0, c0)
                    if key in seen_tiles: dup_tile += 1; continue
                    seen_tiles.add(key)

                patch = read_patch_rgb(rds, r0, c0, PATCH_SIZE, bands=BANDS)
                if patch is None: continue

                if DEDUP_BY_HASH:
                    sig = img_sig(patch)
                    if sig in seen_hash: dup_hash += 1; continue
                    seen_hash.add(sig)

                images.append(patch)
                dm_cm.append(Lm*100.0)
                cdfs.append(one_spike_cdf(Lm))
                base = os.path.splitext(os.path.basename(ORTHO_PATH))[0]
                names.append(f"{base}_r{r0:05d}_c{c0:05d}_stone{i:06d}")
                kept += 1
            except Exception:
                bad += 1
                continue
        print(f"[per_stone_spike] kept={len(images)}, dup_tile={dup_tile}, dup_hash={dup_hash}")

    elif SAMPLE_MODE == "per_csv_distribution":
        # Ein Patch pro CSV (hier: wir nutzen Mediane aller Punkte für Patchlage)
        if "__source__" not in df.columns:
            df["__source__"] = "merged"
        for src, group in df.groupby("__source__"):
            vals = group["b (m)"].to_numpy(np.float32)
            vals = vals[np.isfinite(vals) & (vals >= 0)]
            if vals.size == 0: continue
            xm = float(np.median(group["UTM X (m)"].values))
            ym = float(np.median(group["UTM Y (m)"].values))
            # reproj
            if do_tr is not None and not FORCE_SAME_CRS:
                pt = ogr.Geometry(ogr.wkbPoint); pt.AddPoint(xm, ym)
                pt.Transform(do_tr)
                xm, ym = pt.GetX(), pt.GetY()
            c, r = map_to_pix(inv, xm, ym)
            r0 = int(np.clip(r - PATCH_SIZE//2, 0, H-PATCH_SIZE))
            c0 = int(np.clip(c - PATCH_SIZE//2, 0, W-PATCH_SIZE))
            patch = read_patch_rgb(rds, r0, c0, PATCH_SIZE, bands=BANDS)
            if patch is None: continue
            cdf = cdf_from_values_m(vals)
            d50m = pctl(vals, 50.0)
            images.append(patch)
            dm_cm.append(d50m*100.0)
            cdfs.append(cdf)
            base = os.path.splitext(os.path.basename(ORTHO_PATH))[0]
            names.append(f"{base}_r{r0:05d}_c{c0:05d}_{src}")

        print(f"[per_csv_distribution] kept={len(images)}")

    elif SAMPLE_MODE == "per_window_d50":
        # Fenster/Stride über gesamtem Ortho; b-Werte innerhalb der Fenster-BBox sammeln
        # 1) Punkte einmal in Pixelindices umrechnen, damit Fenstertest schnell wird
        pix = []
        for i in range(len(df)):
            X, Y = to_raster_xy(i)
            c, r = map_to_pix(inv, X, Y)
            if 0 <= r < H and 0 <= c < W:
                pix.append((r, c, float(b[i])))
        if not pix:
            raise RuntimeError("Keine Punkte liegen im Ortho-Auschnitt.")

        # 2) Grobes Bin-Index: Zellen von PATCH_SIZE/8, damit Fenster-Nachbarn schnell auffindbar
        bin_sz = max(8, PATCH_SIZE//8)
        grid = {}
        for r, c, bv in pix:
            br, bc = r//bin_sz, c//bin_sz
            grid.setdefault((br,bc), []).append((r,c,bv))

        def candidates(r0, c0, r1, c1):
            br0, bc0 = r0//bin_sz, c0//bin_sz
            br1, bc1 = r1//bin_sz, c1//bin_sz
            out = []
            for br in range(br0, br1+1):
                for bc in range(bc0, bc1+1):
                    out.extend(grid.get((br,bc), []))
            return out

        kept = 0
        base = os.path.splitext(os.path.basename(ORTHO_PATH))[0]
        for r0 in range(0, H-PATCH_SIZE+1, STRIDE):
            r1 = r0 + PATCH_SIZE - 1
            for c0 in range(0, W-PATCH_SIZE+1, STRIDE):
                c1 = c0 + PATCH_SIZE - 1
                cand = candidates(r0, c0, r1, c1)
                if not cand: continue
                # Punkte im Fenster filtern
                vals = [bv for (rr,cc,bv) in cand if (r0 <= rr <= r1 and c0 <= cc <= c1 and np.isfinite(bv) and bv>=0)]
                if len(vals) < MIN_STONES:
                    continue
                patch = read_patch_rgb(rds, r0, c0, PATCH_SIZE, bands=BANDS)
                if patch is None: continue
                cdf = cdf_from_values_m(np.array(vals, np.float32))
                d50m = pctl(np.array(vals, np.float32), 50.0)
                images.append(patch)
                dm_cm.append(d50m*100.0)
                cdfs.append(cdf)
                names.append(f"{base}_r{r0:05d}_c{c0:05d}_win{PATCH_SIZE}x{PATCH_SIZE}_s{STRIDE}")
                kept += 1
        print(f"[per_window_d50] kept={kept}, window={PATCH_SIZE}, stride={STRIDE}, min_stones={MIN_STONES}")

    else:
        raise ValueError("SAMPLE_MODE unbekannt")

    if len(images) == 0:
        raise RuntimeError("Keine Samples erzeugt. Prüfe CRS, Fenster/Stride, MIN_STONES, CSV-Inhalt.")

    images = np.stack(images, axis=0).astype(np.uint8)
    dm_cm  = np.asarray(dm_cm, dtype=np.float32).reshape(-1)
    cdfs   = np.stack(cdfs, axis=0).astype(np.float32)
    names  = np.asarray(names, dtype=object)

    np.savez_compressed(
        OUT_NPZ_PATH,
        images=images,
        dm=dm_cm,
        histograms=cdfs,
        tile_names=names
    )
    print("\nGespeichert:", OUT_NPZ_PATH)
    print("images     :", images.shape, images.dtype)
    print("dm (cm)    :", dm_cm.shape, f"min/max {dm_cm.min():.2f}/{dm_cm.max():.2f}")
    print("histograms :", cdfs.shape)
    print("tile_names :", names.shape)

if __name__ == "__main__":
    main()
