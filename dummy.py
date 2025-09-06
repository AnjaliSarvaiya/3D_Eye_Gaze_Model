# import h5py

# file_path = "/media/ml/Data Disk/Anjali/Model-aware_3D_Eye_Gaze-main/Datasets/All/DikablisT_10_1.h5"
# with h5py.File(file_path, 'r') as f:
#     print("Top-level groups:", list(f.keys()))
#     grp = list(f.keys())[0]
#     print(f"Inside {grp}:", list(f[grp].keys()))
#     sample = list(f[grp].keys())[0]
#     print(f"Inside {grp}/{sample}:", list(f[grp][sample].keys()))

# import h5py
# import os

# # Update this to your dataset path
# dataset_dir = "/media/ml/Data Disk/Anjali/Model-aware_3D_Eye_Gaze-main/Datasets/All"

# required_keys = ["Images", "pupil_loc", "Gaze_vector"]

# def inspect_file(file_path):
#     with h5py.File(file_path, "r") as f:
#         print(f"\nInspecting: {os.path.basename(file_path)}")
#         missing = []
#         for key in required_keys:
#             if key not in f:
#                 missing.append(key)
#             else:
#                 data = f[key]
#                 print(f"  {key}: shape={data.shape}, dtype={data.dtype}")
#         if missing:
#             print(f"  ⚠ Missing keys: {missing}")

# # Iterate through all h5 files in dataset directory
# for file in os.listdir(dataset_dir):
#     if file.endswith(".h5"):
#         inspect_file(os.path.join(dataset_dir, file))

# import h5py, os
# import numpy as np

# dataset_dir = "/media/ml/Data Disk/Anjali/Model-aware_3D_Eye_Gaze-main/Datasets/All"

# for file in os.listdir(dataset_dir):
#     if file.endswith(".h5"):
#         path = os.path.join(dataset_dir, file)
#         with h5py.File(path, "r") as f:
#             pupil = np.array(f["pupil_loc"])
#             invalid = np.sum(np.isnan(pupil)) + np.sum(pupil < 0)
#             print(f"{file}: total={len(pupil)}, invalid={invalid}")
# dataset_dir = "/media/ml/Data Disk/Anjali/Model-aware_3D_Eye_Gaze-main/Datasets/All"
# masterkey_dir = "/media/ml/Data Disk/Anjali/Model-aware_3D_Eye_Gaze-main/Datasets/MasterKey"
# # import h5py
# import os
# import random

# # Paths
# dataset_dir = "/media/ml/Data Disk/Anjali/Model-aware_3D_Eye_Gaze-main/Datasets/All"
# masterkey_dir = "/media/ml/Data Disk/Anjali/Model-aware_3D_Eye_Gaze-main/Datasets/MasterKey"
# os.makedirs(masterkey_dir, exist_ok=True)

# train_list, val_list, test_list = [], [], []

# # Iterate through each .h5 dataset
# for fname in sorted(os.listdir(dataset_dir)):
#     if fname.endswith(".h5"):
#         fpath = os.path.join(dataset_dir, fname)
#         with h5py.File(fpath, "r") as f:
#             n_frames = f["Images"].shape[0]
#         indices = list(range(n_frames))
#         random.shuffle(indices)
        
#         # Split 80% train, 10% val, 10% test
#         n_train = int(0.8 * n_frames)
#         n_val = int(0.1 * n_frames)
        
#         for idx in indices[:n_train]:
#             train_list.append(f"{fname} {idx}")
#         for idx in indices[n_train:n_train+n_val]:
#             val_list.append(f"{fname} {idx}")
#         for idx in indices[n_train+n_val:]:
#             test_list.append(f"{fname} {idx}")

# # Write lists
# with open(os.path.join(masterkey_dir, "train.txt"), "w") as f:
#     f.write("\n".join(train_list))
# with open(os.path.join(masterkey_dir, "val.txt"), "w") as f:
#     f.write("\n".join(val_list))
# with open(os.path.join(masterkey_dir, "test.txt"), "w") as f:
#     f.write("\n".join(test_list))

# print("MasterKey files generated successfully!")
# audit_teyed.py
# import os, sys, argparse, json, textwrap
# import numpy as np
# import h5py
# import scipy.io as scio

# def h5_summary(path):
#     out = {"file": path, "ok": True, "error": None, "shapes": {}, "lengths": {}}
#     must = ["Images"]
#     nice = ["pupil_loc", "Masks_noSkin", "Eyeball", "Gaze_vector",
#             "pupil_lm_2D", "pupil_lm_3D", "iris_lm_2D", "iris_lm_3D"]
#     try:
#         with h5py.File(path, "r") as f:
#             def shape_of(key):
#                 try:
#                     d = f[key]
#                     return d.shape
#                 except Exception:
#                     return None

#             # top-level datasets
#             for k in must + nice:
#                 if k in f:
#                     out["shapes"][k] = shape_of(k)

#             # fits group
#             if "Fits" in f:
#                 out["shapes"]["Fits/pupil"] = shape_of("Fits/pupil")
#                 out["shapes"]["Fits/iris"]  = shape_of("Fits/iris")

#             # lengths sanity
#             if "Images" in f:
#                 n = f["Images"].shape[0]
#                 out["lengths"]["Images"] = n
#                 for k in ["pupil_loc","Masks_noSkin","Eyeball","Gaze_vector",
#                           "pupil_lm_2D","pupil_lm_3D","iris_lm_2D","iris_lm_3D"]:
#                     if k in f and f[k].shape[0] != n:
#                         out["lengths"][k] = f[k].shape[0]

#                 if "Fits" in f:
#                     for k in ["pupil","iris"]:
#                         kk = f"Fits/{k}"
#                         if kk in f and f[kk].shape[0] != n:
#                             out["lengths"][kk] = f[kk].shape[0]

#             # a few content checks
#             if "pupil_loc" in f:
#                 pl = f["pupil_loc"][...]
#                 out["pupil_loc_stats"] = {
#                     "neg_rows": int(np.any(pl<0, axis=1).sum()),
#                     "near_borders_10pct": int(np.any((pl<0.10)|(pl>0.90), axis=1).sum())
#                 }
#             if "Masks_noSkin" in f:
#                 ms = f["Masks_noSkin"][0] if f["Masks_noSkin"].shape[0]>0 else None
#                 out["mask_dtype"] = str(ms.dtype) if ms is not None else None
#     except Exception as e:
#         out["ok"] = False
#         out["error"] = repr(e)
#     return out

# def masterkey_summary(path):
#     out = {"dir": path, "files": [], "ok": True}
#     for fn in sorted(os.listdir(path)):
#         if not fn.lower().endswith(".mat"):
#             continue
#         p = os.path.join(path, fn)
#         try:
#             m = scio.loadmat(p, squeeze_me=True, struct_as_record=False)
#             # required fields
#             dataset = str(m.get("dataset","?"))
#             subset  = m.get("subset","?")
#             archive = m.get("archive", np.array([], dtype=str))
#             pupil_loc = m.get("pupil_loc", np.empty((0,2)))
#             fits = m.get("Fits", None)
#             res = m.get("resolution", np.empty((0,2)))

#             iris_ok = 0
#             if fits is not None and hasattr(fits, "iris"):
#                 iris = fits.iris if fits.iris is not None else np.empty((0,5))
#                 iris_ok = iris.shape[0]

#             entry = {
#                 "file": fn,
#                 "dataset": dataset,
#                 "subset": str(subset) if not isinstance(subset, np.ndarray) else f"array({subset.shape})",
#                 "archive_len": int(np.size(archive)),
#                 "pupil_loc_shape": tuple(pupil_loc.shape) if hasattr(pupil_loc, "shape") else "list",
#                 "iris_fits_len": int(iris_ok),
#                 "resolution_shape": tuple(res.shape) if hasattr(res, "shape") else "?"
#             }
#             out["files"].append(entry)
#         except Exception as e:
#             out["files"].append({"file": fn, "error": repr(e)})
#     return out

# def main():
#     ap = argparse.ArgumentParser()
#     ap.add_argument("--path_data", required=True, help="Root with H5 files (e.g., .../Datasets/All)")
#     ap.add_argument("--path_masterkey", required=True, help="MasterKey dir with .mat files")
#     args = ap.parse_args()

#     print("=== MASTERKEY SUMMARY ===")
#     mk = masterkey_summary(args.path_masterkey)
#     print(json.dumps(mk, indent=2))

#     print("\n=== H5 SUMMARY (first 10 files) ===")
#     h5s = [os.path.join(args.path_data, f) for f in sorted(os.listdir(args.path_data)) if f.endswith(".h5")]
#     for p in h5s[:10]:
#         rep = h5_summary(p)
#         print(json.dumps(rep, indent=2))

#     # cross-check: all H5 share consistent first dimension?
#     lens = []
#     for p in h5s[:10]:
#         try:
#             with h5py.File(p, "r") as f:
#                 lens.append(f["Images"].shape[0])
#         except Exception:
#             pass
#     if lens:
#         print("\nH5 Images lengths (first 10):", lens, "unique:", sorted(set(lens)))

# if __name__ == "__main__":
#     main()
# import h5py, numpy as np

# p = "/media/ml/03C90A2604F5F8BA/Anjali/Model-aware_3D_Eye_Gaze-main/Results/pretrained_sem/results/test_results.h5"
# with h5py.File(p, "r") as f:
#     # pick any sample path you used earlier
#     g = f["DikablisR_1_1"]["1258"]
#     img = np.asarray(g["image"])
#     mgt = np.asarray(g["mask_gt"]) if "mask_gt" in g else None

#     print("image unique:", np.unique(img))
#     if mgt is not None:
#         print("mask_gt unique:", np.unique(mgt))
#         print("image==mask_gt ? ", np.array_equal(img, mgt))


# dummy_h5py.py
# import h5py, numpy as np
# p = "/media/ml/03C90A2604F5F8BA/Anjali/Model-aware_3D_Eye_Gaze-main/Results/pretrained_sem/results/test_results.h5"
# with h5py.File(p,"r") as f:
#     arch = next(iter(f.keys()))
#     frame = next(iter(f[arch].keys()))
#     arr = f[arch][frame]["image"][...]
#     print(arr.dtype, np.unique(arr))


# import h5py, itertools
# p = "/media/ml/03C90A2604F5F8BA/Anjali/Model-aware_3D_Eye_Gaze-main/Results/pretrained_sem/results/test_results.h5"
# with h5py.File(p, "r") as f:
#     for arch in itertools.islice(f.keys(), 5):
#         g_arch = f[arch]
#         print(f"[ARCH] {arch}")
#         for frame in itertools.islice(g_arch.keys(), 5):
#             g = g_arch[frame]
#             print(f"  [FRAME] {frame}")
#             for k in g.keys():
#                 d = g[k]
#                 shape = getattr(d, "shape", None)
#                 dtype = getattr(d, "dtype", None)
#                 print(f"    - {k}: shape={shape}, dtype={dtype}")

# import h5py, numpy as np
# p = "/media/ml/03C90A2604F5F8BA/Anjali/Model-aware_3D_Eye_Gaze-main/Results/pretrained_sem/results/test_results.h5"
# with h5py.File(p,"r") as f:
#     arch = next(iter(f.keys()))
#     frame = next(iter(f[arch].keys()))
#     arr = f[arch][frame]["image_raw_u8"][...]
#     print(arr.dtype, np.unique(arr))
#!/usr/bin/env python3
# import argparse, pickle, inspect, sys
# import numpy as np

# def _fmt(x):
#     try:
#         if hasattr(x, "shape"):  return f"{type(x).__name__} shape={tuple(x.shape)} dtype={getattr(x,'dtype',None)}"
#         if isinstance(x, (list, tuple)): return f"{type(x).__name__} len={len(x)}"
#         if isinstance(x, dict):  return f"dict keys={list(x.keys())[:10]}{'...' if len(x)>10 else ''}"
#         return f"{type(x).__name__}: {str(x)[:80]}"
#     except Exception as e:
#         return f"{type(x).__name__} (repr failed: {e})"

# def describe_ds(ds, name):
#     print(f"\n=== {name} ===")
#     print("Type:", type(ds))
#     # length
#     try: print("len(ds):", len(ds))
#     except Exception as e: print("len(ds): <error>", e)

#     # common attrs we see in your code
#     for attr in ["imList", "arch", "augFlag", "equi_var", "scale", "frames", "path2data", "path2h5"]:
#         if hasattr(ds, attr):
#             val = getattr(ds, attr)
#             if isinstance(val, np.ndarray):
#                 print(f"{attr}: ndarray shape={val.shape} dtype={val.dtype}")
#             else:
#                 print(f"{attr}: {_fmt(val)}")

#     # imList quick summary
#     if hasattr(ds, "imList") and isinstance(ds.imList, np.ndarray):
#         il = ds.imList
#         print(f"imList dims: {il.shape} (usually [num_sequences, frames, ...])")
#         if il.ndim >= 2:
#             unique_vids = np.unique(il[:, :, 1]) if il.shape[-1] > 1 else None
#             if unique_vids is not None:
#                 print(f"unique video IDs (sample): {unique_vids[:10]}{' ...' if unique_vids.size>10 else ''}")

#     # one sample peek (can be heavy; gated by --peek)
#     return

# def peek_sample(ds, idx=0):
#     print("\n--- sample peek ---")
#     try:
#         item = ds[idx]
#         if isinstance(item, dict):
#             for k, v in item.items():
#                 print(f"{k:>20}: {_fmt(v)}")
#         elif isinstance(item, tuple):
#             for i, v in enumerate(item):
#                 print(f"field[{i}]: {_fmt(v)}")
#         else:
#             print("item:", _fmt(item))
#     except Exception as e:
#         print("Could not fetch sample:", e)

# def main():
#     ap = argparse.ArgumentParser()
#     ap.add_argument("--pkl", required=True)
#     ap.add_argument("--peek", type=int, default=0, help="peek one dataset sample per split")
#     args = ap.parse_args()

#     print(f"Loading: {args.pkl}")
#     with open(args.pkl, "rb") as f:
#         obj = pickle.load(f)

#     print("\nTop-level object:", type(obj))
#     if isinstance(obj, (list, tuple)) and len(obj) == 3:
#         train_obj, valid_obj, test_obj = obj
#         describe_ds(train_obj, "train_obj")
#         if args.peek: peek_sample(train_obj, 0)
#         describe_ds(valid_obj, "valid_obj")
#         if args.peek: peek_sample(valid_obj, 0)
#         describe_ds(test_obj, "test_obj")
#         if args.peek: peek_sample(test_obj, 0)
#     elif isinstance(obj, dict):
#         print("dict keys:", list(obj.keys()))
#         for k, v in obj.items():
#             print(f"\n[{k}] -> {_fmt(v)}")
#     else:
#         print("Object summary:", _fmt(obj))

# if __name__ == "__main__":
#     # ⚠️ Only unpickle files you trust.
#     main()
# # python - <<'PY'
# import pickle
# p = "cur_objs/one_vs_one/cond_TEyeD.pkl"
# train_obj, valid_obj, test_obj = pickle.load(open(p, "rb"))

# def report(name, obj):
#     nseq, nframes = obj.imList.shape[:2]
#     print(f"{name}: sequences={nseq}, frames/seq={nframes}, images={nseq*nframes}")

# report("train", train_obj)
# report("valid", valid_obj)
# report("test",  test_obj)
# PY
# python - <<'PY'
# import pickle, numpy as np
# p = "cur_objs/one_vs_one/cond_TEyeD.pkl"
# train, valid, test = pickle.load(open(p, "rb"))

# def sig(x):
#     # (arch_idx, frame_id) signature per frame
#     return x.imList[..., :2].reshape(-1, 2)

# def report(a, b, name_a, name_b):
#     same_shape = a.shape == b.shape
#     same_all = same_shape and np.array_equal(a, b)
#     inter = {tuple(t) for t in a.tolist()} & {tuple(t) for t in b.tolist()}
#     print(f"{name_a} vs {name_b}:")
#     print(f"  exact same arrays? {same_all}")
#     print(f"  overlap frames: {len(inter)} / {len(a)} ({len(inter)/len(a)*100:.1f}%)\n")

# s_tr, s_va, s_te = sig(train), sig(valid), sig(test)
# print("Sizes (images):", len(s_tr), len(s_va), len(s_te))
# report(s_tr, s_va, "train", "valid")
# report(s_tr, s_te, "train", "test")
# report(s_va, s_te, "valid", "test")
# # sPY

# make_disjoint_splits.py
# make_disjoint_splits.py
import pickle, numpy as np, copy, os

SRC = "cur_objs/one_vs_one/cond_TEyeD.pkl"
DST = "cur_objs/one_vs_one/cond_TEyeD_disjoint.pkl"
RATIOS = (0.70, 0.15, 0.15)  # train/valid/test
SEED = 42

def norm_imlist(im):
    im = np.asarray(im)
    if im.ndim == 3:          # (N_seq, F, 3)
        return im
    if im.ndim == 2 and im.shape[1] == 3:
        return im[:, None, :] # (N_seq, 1, 3)
    if im.ndim == 2 and im.shape[1] % 3 == 0:
        F = im.shape[1] // 3
        return im.reshape(im.shape[0], F, 3)
    raise ValueError(f"Unexpected imList shape: {im.shape}")

def seq_table(ds):
    im = norm_imlist(ds.imList)
    first = im[:, 0, :]  # (N_seq, 3) -> [arch_id, video_id, frame_id]
    return im, first

def largest_remainder_counts(n, ratios):
    q = np.array(ratios) * n
    base = np.floor(q).astype(int)
    remainder = n - base.sum()
    # distribute leftover to largest fractional parts
    frac_order = np.argsort(-(q - base))
    base[frac_order[:remainder]] += 1
    return tuple(base.tolist())  # (n_tr, n_va, n_te)

def filtered_copy(ds, keep_pairs):
    """keep_pairs is a set of (arch_id, video_id) to keep."""
    ds2 = copy.deepcopy(ds)
    im, first = seq_table(ds2)
    keep_idx = [i for i in range(first.shape[0])
                if (int(first[i,0]), int(first[i,1])) in keep_pairs]
    ds2.imList = ds2.imList[keep_idx]
    return ds2

# ---- load
train, valid, test = pickle.load(open(SRC, "rb"))
_, first = seq_table(train)

# unique (arch, video) pairs globally
pairs = np.unique(first[:, :2], axis=0).astype(int)  # shape (M, 2)
rng = np.random.default_rng(SEED)
rng.shuffle(pairs)

n = len(pairs)
n_tr, n_va, n_te = largest_remainder_counts(n, RATIOS)
tr_pairs = set(map(tuple, pairs[:n_tr]))
va_pairs = set(map(tuple, pairs[n_tr:n_tr+n_va]))
te_pairs = set(map(tuple, pairs[n_tr+n_va:]))

train_new = filtered_copy(train, tr_pairs)
valid_new = filtered_copy(valid, va_pairs)
test_new  = filtered_copy(test,  te_pairs)

os.makedirs(os.path.dirname(DST), exist_ok=True)
pickle.dump((train_new, valid_new, test_new), open(DST, "wb"))
print("Saved:", DST)

def count_images(ds):
    im = norm_imlist(ds.imList); return im.shape[0]*im.shape[1]
for name, ds in [("train", train_new), ("valid", valid_new), ("test", test_new)]:
    im = norm_imlist(ds.imList)
    print(f"{name}: sequences={im.shape[0]}, frames/seq={im.shape[1]}, images={count_images(ds)}")


