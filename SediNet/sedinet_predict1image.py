## Modified sedinet_predict1image.py — adds tiled inference & GeoTIFF output
## Keeps original single-image behavior intact
## Daniel Buscombe (orig) + minimal extensions

# sedinet_predict1image.py
# Single-image prediction + tiled GeoTIFF inference for SediNet
# Keeps original behavior, adds:
#  - GPU memory growth safety
#  - Tiled inference for large GeoTIFFs
#  - Proper SIMO/SISO mode detection from config
#  - Selection of P50 (or first available var) for map writing
#  - LZW-compressed GeoTIFF output with NoData

import sys, getopt, json, os, shutil, tempfile
from numpy import any as npany
import numpy as np

# ---------------- GPU setup (safe growth) ----------------
USE_GPU = True
os.environ['CUDA_VISIBLE_DEVICES'] = '0' if USE_GPU else '-1'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
try:
    import tensorflow as tf
    for g in tf.config.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(g, True)
except Exception:
    pass

# ---------------- SediNet imports ----------------
from sedinet_eval import *  # uses estimate_siso_simo_1image / estimate_categorical_1image

# ---------------- GeoTIFF I/O ----------------
from osgeo import gdal
from PIL import Image

# =========================================================
# Helpers
# =========================================================
def _read_rgb(ds, rgb_idx=(1, 2, 3)):
    """Read HxWx3 as uint8, robustly scaled from input range to 0..255."""
    arrs = []
    for b in rgb_idx:
        band = ds.GetRasterBand(b)
        A = band.ReadAsArray().astype(np.float32)
        # robust scale to 0..255 using 1..99 percentile
        finite = np.isfinite(A)
        if not np.any(finite):
            A = np.zeros_like(A, dtype=np.float32)
        else:
            lo, hi = np.percentile(A[finite], [1, 99])
            if hi <= lo:
                A = np.clip(A, 0, 255)
            else:
                A = (A - lo) / (hi - lo) * 255.0
                A = np.clip(A, 0, 255)
        arrs.append(A)
    img = np.stack(arrs, axis=-1).astype(np.uint8)
    return img

def _tiles(h, w, tile_r, tile_c, stride):
    """Return top-left indices for tiling, covering edges."""
    rows = list(range(0, max(0, h - tile_r + 1), stride))
    cols = list(range(0, max(0, w - tile_c + 1), stride))
    if rows and rows[-1] != h - tile_r:
        rows.append(h - tile_r)
    if cols and cols[-1] != w - tile_c:
        cols.append(w - tile_c)
    return rows, cols

def _save_geotiff_like(out_path, ref_ds, array2d, x_res_m, y_res_m):
    """Write 2D float map with same origin/CRS; pixel size = tile stride (map units)."""
    gt = list(ref_ds.GetGeoTransform())
    gt[1] = float(x_res_m)
    gt[5] = -abs(float(y_res_m))

    drv = gdal.GetDriverByName('GTiff')
    dst = drv.Create(
        out_path,
        int(array2d.shape[1]),
        int(array2d.shape[0]),
        1,
        gdal.GDT_Float32,
        options=['COMPRESS=LZW', 'PREDICTOR=3', 'TILED=YES']
    )
    dst.SetGeoTransform(tuple(gt))
    dst.SetProjection(ref_ds.GetProjection())
    band = dst.GetRasterBand(1)
    band.WriteArray(array2d.astype(np.float32))
    band.SetNoDataValue(np.nan)
    band.FlushCache()
    dst = None

def _vars_from_config(config):
    """Extract output variables (e.g., P5..P95) in sorted order from config."""
    vars = [k for k in config.keys() if not npany([
        k.startswith('base'), k.startswith('MAX_LR'),
        k.startswith('MIN_LR'), k.startswith('DO_AUG'),
        k.startswith('SHALLOW'), k.startswith('res_folder'),
        k.startswith('train_csvfile'), k.startswith('csvfile'),
        k.startswith('test_csvfile'), k.startswith('name'),
        k.startswith('greyscale'), k.startswith('aux_in'),
        k.startswith('dropout'), k.startswith('N'),
        k.startswith('scale'), k.startswith('numclass')
    ])]
    return sorted(vars)

def _coerce_weights_for_eval(weights_path, BATCH_SIZE):
    # sedinet_eval erwartet bei single-batch einen String, keine Liste
    if isinstance(weights_path, list):
        if not isinstance(BATCH_SIZE, list):
            # nur ein Batch -> erste Datei nehmen
            return weights_path[0]
        # Falls BATCH_SIZE Liste ist, muss auch weights_path eine Liste gleicher Länge sein.
        # In deinem Fall nicht relevant – lassen wir dann als Liste stehen.
    return weights_path

# =========================================================
# Single image prediction (original flow) with auto SIMO/SISO
# =========================================================
def _predict_single_image(image_path, config, weights_path, name, BATCH_SIZE):
    vars      = _vars_from_config(config)
    greyscale = config.get('greyscale', 'true')
    dropout   = config['dropout']
    numclass  = config.get('numclass', 0)
    scale     = config['scale']
    res_folder= config['res_folder']

    # SIMO wenn mehrere Zielvariablen in der Config stehen
    mode = 'simo' if len(vars) > 1 else 'siso'

    # Falls BATCH_SIZE eine Liste ist, müssen auch mehrere Weight-Dateien kommen
    if isinstance(BATCH_SIZE, list) and not isinstance(weights_path, list):
        print("Please specify one weights file per batch size in the list ... exiting")
        sys.exit(2)

    # Robust: Einzel-/Mehrfach-Weights passend machen
    weights_path = _coerce_weights_for_eval(weights_path, BATCH_SIZE)
    # (_coerce_weights_for_eval soll im Single-Batch-Fall einen String zurückgeben,
    #  im Multi-Batch-Fall eine Liste gleicher Länge wie BATCH_SIZE.)

    if numclass > 0:
        return estimate_categorical_1image(
            vars, image_path, res_folder, dropout, numclass, greyscale, name, mode, weights_path
        )
    else:
        return estimate_siso_simo_1image(
            vars, image_path, greyscale, dropout, numclass, scale, name, mode,
            res_folder, BATCH_SIZE, weights_path
        )

# =========================================================
# Tiled prediction on GeoTIFF; outputs a map of P50 (or first var)
# =========================================================
def _tiled_predict_geotiff(tif_path, config, weights_path, name,
                           tile_rows, tile_cols, stride_px,
                           tmpdir, batch_pred, save_map_path):
    ds = gdal.Open(tif_path)
    if ds is None:
        raise RuntimeError(f"Cannot open {tif_path}")

    H, W = ds.RasterYSize, ds.RasterXSize
    gt = ds.GetGeoTransform()  # (originX, pxW, 0, originY, 0, -pxH)
    px_w, px_h = gt[1], abs(gt[5])
    step_x_m = px_w * stride_px
    step_y_m = px_h * stride_px

    full = _read_rgb(ds)
    rows_idx, cols_idx = _tiles(H, W, tile_rows, tile_cols, stride_px)
    nR, nC = len(rows_idx), len(cols_idx)

    os.makedirs(tmpdir, exist_ok=True)

    # Decide which variable to map: prefer P50, else first var
    vars_in_cfg = _vars_from_config(config)
    target_var = 'P50' if 'P50' in vars_in_cfg else (vars_in_cfg[0] if vars_in_cfg else 'P50')
    target_idx = vars_in_cfg.index(target_var) if target_var in vars_in_cfg else 0

    vals = np.full((nR, nC), np.nan, dtype=np.float32)

    count = 0
    total = nR * nC
    for ir, r0 in enumerate(rows_idx):
        for ic, c0 in enumerate(cols_idx):
            r1, c1 = r0 + tile_rows, c0 + tile_cols
            tile = full[r0:r1, c0:c1, :]
            if tile.shape[0] != tile_rows or tile.shape[1] != tile_cols:
                continue
            tmp_img = os.path.join(tmpdir, f"tile_r{r0}_c{c0}.jpg")
            Image.fromarray(tile).save(tmp_img, quality=92)

            res = _predict_single_image(tmp_img, config, weights_path, name, batch_pred)

            # Zielvariable bestimmen (z.B. 'P50'); falls nur 1 Var trainiert wurde → Index 0
            vars_in_cfg = [k for k in config.keys() if not npany([k.startswith('base'), k.startswith('MAX_LR'),
                k.startswith('MIN_LR'), k.startswith('DO_AUG'), k.startswith('SHALLOW'),
                k.startswith('res_folder'), k.startswith('train_csvfile'), k.startswith('csvfile'),
                k.startswith('test_csvfile'), k.startswith('name'), k.startswith('greyscale'),
                k.startswith('aux_in'), k.startswith('dropout'), k.startswith('N'),
                k.startswith('scale'), k.startswith('numclass')])]
            vars_in_cfg = sorted(vars_in_cfg)
            target_var = vars_in_cfg[0] if vars_in_cfg else 'P50'

            arr = np.array(res, dtype='float32').ravel()
            if len(vars_in_cfg) > 1 and target_var in vars_in_cfg:
                idx = vars_in_cfg.index(target_var)
                val = float(arr[idx]) if arr.size > idx else float(arr[0])
            else:
                val = float(arr[0])

            vals[ir, ic] = val

            count += 1
            if count % 50 == 0:
                print(f" ... {count}/{total} tiles")

    if save_map_path:
        _save_geotiff_like(save_map_path, ds, vals, step_x_m, step_y_m)

    # cleanup temp tiles
    try:
        shutil.rmtree(tmpdir)
    except Exception:
        pass

    return vals

# =========================================================
# Main
# =========================================================
if __name__ == '__main__':
    argv = sys.argv[1:]

    # NEW options for tiled mode
    tile_rows = tile_cols = stride = None
    save_map = None
    tmpdir = None
    batch_pred = 1

    try:
        opts, args = getopt.getopt(
            argv,
            "h:c:i:w:1:2:3:4:",
            ["tile_rows=", "tile_cols=", "stride=", "save_map=", "tmpdir=", "batch_pred="]
        )
    except getopt.GetoptError:
        print('python sedinet_predict1image.py -c config.json -i path/to/image.tif '
              '{-w weights.hdf5 | -1 w1.hdf5 -2 w2.hdf5 ...} '
              '[--tile_rows 256 --tile_cols 256 --stride 256 --save_map out.tif --tmpdir tmp_tiles --batch_pred 1]')
        sys.exit(2)

    for opt, arg in opts:
        if opt == '-h':
            print('Examples:\n'
                  '  Single image:\n'
                  '    python sedinet_predict1image.py -c config.json -i images/img.JPG -w weights.hdf5\n'
                  '  Tiled GeoTIFF:\n'
                  '    python sedinet_predict1image.py -c config.json -i big.tif -w weights.hdf5 '
                  '--tile_rows 256 --tile_cols 256 --stride 256 --save_map dm_pred.tif --tmpdir tmp_tiles')
            sys.exit()
        elif opt == "-c":
            configfile = arg
        elif opt == "-w":
            weights_path = arg
        elif opt == "-i":
            image = arg
        elif opt == "-1":
            weights_path1 = arg
        elif opt == "-2":
            weights_path2 = arg
        elif opt == "-3":
            weights_path3 = arg
        elif opt == "-4":
            weights_path4 = arg
        elif opt == "--tile_rows":
            tile_rows = int(arg)
        elif opt == "--tile_cols":
            tile_cols = int(arg)
        elif opt == "--stride":
            stride = int(arg)
        elif opt == "--save_map":
            save_map = arg
        elif opt == "--tmpdir":
            tmpdir = arg
        elif opt == "--batch_pred":
            batch_pred = int(arg)

    # inputs present?
    if 'image' not in locals():
        print("image path required (-i) ... exiting")
        sys.exit(2)

    # check weights
    if 'weights_path1' not in locals():
        if not (os.path.isfile(os.getcwd()+os.sep+weights_path) or os.path.isfile(weights_path)):
            print("Weights path does not exist ... exiting")
            sys.exit(2)
    else:
        weights_path = []
        for key in ['weights_path1', 'weights_path2', 'weights_path3', 'weights_path4']:
            if key in locals():
                w = locals()[key]
                if os.path.isfile(os.getcwd()+os.sep+w) or os.path.isfile(w):
                    weights_path.append(w)
                else:
                    print(f"Weights path {key[-1]} does not exist ... exiting")
                    sys.exit(2)

    # load config
    try:
        with open(os.getcwd()+os.sep+configfile) as f:
            config = json.load(f)
    except Exception:
        with open(configfile) as f:
            config = json.load(f)

    BATCH_SIZE = config.get('BATCH_SIZE', 1)
    name       = config["name"]

    # -------- Single-image mode (original behavior) --------
    if tile_rows is None or tile_cols is None or stride is None:
        result = _predict_single_image(image, config, weights_path, name, BATCH_SIZE)

        print("=====================================")
        print("==============RESULTS ==============")
        vars = _vars_from_config(config)
        if isinstance(result, (list, tuple, np.ndarray)):
            for v, val in zip(vars, result):
                print(f"{v}: {float(val):.3f}")
        else:
            v = vars[0] if vars else 'P50'
            print(f"{v}: {float(result):.3f}")
        sys.exit(0)

    # -------- Tiled GeoTIFF mode --------
    if tmpdir is None:
        tmpdir = tempfile.mkdtemp(prefix="sedinet_tiles_")

    print(f"[TILED] {os.path.basename(image)} — tiles {tile_rows}x{tile_cols}, stride {stride}px")
    vals = _tiled_predict_geotiff(
        tif_path=image,
        config=config,
        weights_path=weights_path,
        name=name,
        tile_rows=tile_rows,
        tile_cols=tile_cols,
        stride_px=stride,
        tmpdir=tmpdir,
        batch_pred=batch_pred,
        save_map_path=save_map
    )
    print(f"[TILED] done. Grid shape: {vals.shape}. Saved map: {save_map if save_map else '(not saved)'}")


# ## Written by Daniel Buscombe,
# ## MARDA Science
# ## daniel@mardascience.com

# ##> Release v1.3 (July 2020)

# ###===================================================
# # import libraries
# import sys, getopt, json, os
# from numpy import any as npany

# USE_GPU = True

# if USE_GPU == True:
   # ##use the first available GPU
   # os.environ['CUDA_VISIBLE_DEVICES'] = '0' #'1'
# else:
   # ## to use the CPU (not recommended):
   # os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# from sedinet_eval import *

# #==============================================================
# if __name__ == '__main__':

    # argv = sys.argv[1:]
    # try:
        # opts, args = getopt.getopt(argv,"h:c:i:w:1:2:3:4:")
    # except getopt.GetoptError:
        # print('python sedinet_predict1image.py -c configfile.json -i path/to/image.ext {-w weightsfile.hdf5} OR {-1 weightsfile_batch1.hdf5 -2 weightsfile_batch2.hdf5 -3 weightsfile_batch3.hdf5 -4 weightsfile_batch4.hdf5}')
        # sys.exit(2)
    # for opt, arg in opts:
        # if opt == '-h':
            # print('Example usage (single batch / weights file): python sedinet_predict1image.py \
                   # -c config/config_mattole.json \
                   # -i images/mattole_images/all/DSCN3521c.JPG \
                   # -w mattole/res/mattole_simo_batch7_im512_512_2vars_pinball_aug.hdf5')
            # print('Example usage (multiple batches / weights files): python sedinet_predict1image.py -c config/config_9percentiles.json \
                   # -1 grain_size_global/res/global_9prcs_simo_batch7_im768_9vars_pinball_noaug.hdf5 \
                   # -2 grain_size_global/res/global_9prcs_simo_batch12_im768_9vars_pinball_noaug.hdf5 \
                   # -3 grain_size_global/res/global_9prcs_simo_batch14_im768_9vars_pinball_noaug.hdf5')
            # sys.exit()
        # elif opt in ("-c"):
            # configfile = arg
        # elif opt in ("-w"):
            # weights_path = arg
        # elif opt in ("-i"):
            # image = arg
        # elif opt in ("-1"):
            # weights_path1 = arg
        # elif opt in ("-2"):
            # weights_path2 = arg
        # elif opt in ("-3"):
            # weights_path3 = arg
        # elif opt in ("-4"):
            # weights_path4 = arg

    # if 'image' not in locals():
        # if not os.path.isfile(image):
           # print("image folder path does not exist ... exiting")
           # sys.exit()

    # if 'weights_path1' not in locals():
        # if not os.path.isfile(os.getcwd()+os.sep+weights_path):
            # if not os.path.isfile(weights_path):
               # print("Weights path does not exist ... exiting")
               # sys.exit()
    # else:
        # weights_path = []
        # if not os.path.isfile(os.getcwd()+os.sep+weights_path1):
            # if not os.path.isfile(weights_path1):
               # print("Weights path 1 does not exist ... exiting")
               # sys.exit()
            # else:
               # weights_path.append(weights_path1)
        # else:
           # weights_path.append(weights_path1)

    # if 'weights_path2' in locals():
        # if not os.path.isfile(os.getcwd()+os.sep+weights_path2):
            # if not os.path.isfile(weights_path2):
               # print("Weights path 2 does not exist ... exiting")
               # sys.exit()
            # else:
               # weights_path.append(weights_path2)
        # else:
           # weights_path.append(weights_path2)

    # if 'weights_path3' in locals():
        # if not os.path.isfile(os.getcwd()+os.sep+weights_path3):
            # if not os.path.isfile(weights_path3):
               # print("Weights path 3 does not exist ... exiting")
               # sys.exit()
            # else:
               # weights_path.append(weights_path3)
        # else:
           # weights_path.append(weights_path3)

    # if 'weights_path4' in locals():
        # if not os.path.isfile(os.getcwd()+os.sep+weights_path4):
            # if not os.path.isfile(weights_path4):
               # print("Weights path 4 does not exist ... exiting")
               # sys.exit()
            # else:
               # weights_path.append(weights_path4)
        # else:
           # weights_path.append(weights_path4)

    # try:
       # # load the user configs
       # with open(os.getcwd()+os.sep+configfile) as f:
          # config = json.load(f)
    # except:
       # # load the user configs
       # with open(configfile) as f:
          # config = json.load(f)

    # ###===================================================
    # #csvfile containing image names and class values
    # csvfile = config["csvfile"]
    # #csvfile containing image names and class values
    # res_folder = config["res_folder"]
    # #folder containing csv file and that will contain model outputs
    # name = config["name"]
    # #name prefix for output files
    # #convert imagery to greyscale or not
    # dropout = config["dropout"]
    # #dropout factor
    # scale = config["scale"] #do scaling on variable
    # greyscale = config['greyscale']

    # try:
       # numclass = config['numclass']
    # except:
       # numclass = 0

    # try:
       # greyscale = config['greyscale']
    # except:
       # greyscale = 'true'

    # #output variables
    # vars = [k for k in config.keys() if not npany([k.startswith('base'), k.startswith('MAX_LR'),
            # k.startswith('MIN_LR'), k.startswith('DO_AUG'), k.startswith('SHALLOW'),
            # k.startswith('res_folder'), k.startswith('train_csvfile'), k.startswith('csvfile'),
            # k.startswith('test_csvfile'), k.startswith('name'),
            # k.startswith('greyscale'), k.startswith('aux_in'),
            # k.startswith('dropout'), k.startswith('N'),
            # k.startswith('scale'), k.startswith('numclass')])]
    # vars = sorted(vars)

    # #this relates to 'mimo' and 'miso' modes that are planned for the future but not currently implemented
    # auxin = [k for k in config.keys() if k.startswith('aux_in')]

    # if len(auxin) > 0:
       # auxin = config[auxin[0]]   ##at least for now, just one 'auxilliary' (numerical/categorical) input in addition to imagery
       # if len(vars) ==1:
          # mode = 'miso'
       # elif len(vars) >1:
          # mode = 'mimo'
    # else:
       # if len(vars) ==1:
          # mode = 'siso'
       # elif len(vars) >1:
          # mode = 'simo'

    # print("Mode: %s" % (mode))
    # ###==================================================

    # if type(BATCH_SIZE) is list and type(weights_path) is not list:
       # print("Please specify one weights file per batch size in the list ... exiting")
       # sys.exit()

    # if (mode=='siso' or mode=='simo'):
       # if numclass>0:
          # result = estimate_categorical_1image(vars, image, res_folder, dropout,
                                   # numclass, greyscale, name, mode, weights_path)
       # else:
          # result = estimate_siso_simo_1image(vars, image, greyscale,
                             # dropout, numclass, scale, name, mode,
                             # res_folder, BATCH_SIZE, weights_path)


    # print("=====================================")
    # print("==============RESULTS ==============")
    # if len(vars)>1:
        # counter = 0
        # for var in vars:
            # print(var+": %f" % (result[counter]))
            # counter +=1
    # else:
        # print(vars[0]+": %f" % (result))
