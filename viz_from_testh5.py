# #!/usr/bin/env python3
# import os, math, h5py, numpy as np
# import matplotlib.pyplot as plt

# def to_uint8(gray):
#     g = np.asarray(gray)
#     if g.ndim == 3 and g.shape[0] == 1: g = g[0]
#     g = g.astype(np.float32)
#     g -= np.nanmin(g)
#     mx = np.nanmax(g)
#     if mx > 0: g /= mx
#     return (g * 255.0).clip(0,255).astype(np.uint8)

# def mask_label_map(mask):
#     if mask is None: return (None, None)
#     u, cnt = np.unique(mask, return_counts=True)
#     uc = dict(zip(u.tolist(), cnt.tolist()))
#     uc.pop(0, None)
#     if not uc: return (None, None)
#     order = sorted(uc.items(), key=lambda x: x[1])
#     pupil_id = order[0][0]
#     iris_id  = order[-1][0] if len(order) > 1 else None
#     return iris_id, pupil_id

# def overlay_mask(gray, mask):
#     if mask is None:
#         return np.stack([gray,gray,gray], -1)
#     rgb = np.stack([gray,gray,gray], -1).astype(np.float32)
#     iris_id, pupil_id = mask_label_map(mask)
#     alpha = 0.45
#     if iris_id is not None:
#         m = (mask == iris_id); rgb[m] = (1-alpha)*rgb[m] + alpha*np.array([128,0,128], np.float32)
#     if pupil_id is not None:
#         m = (mask == pupil_id); rgb[m] = (1-alpha)*rgb[m] + alpha*np.array([255,215,0], np.float32)
#     return rgb.clip(0,255).astype(np.uint8)

# def norm2d(vx, vy, eps=1e-6):
#     n = math.hypot(vx, vy)
#     if n < eps: return 0.0, 0.0
#     return vx/n, vy/n

# def draw_tile(ax, img_rgb, origin, g_pred, g_gt, arrow_len=60):
#     ax.imshow(img_rgb)
#     ax.axis("off")
#     ox, oy = origin
#     # red origin marker "A"
#     ax.plot(ox, oy, marker='o', markersize=4, linewidth=0, color='red')
#     ax.text(ox+3, oy-3, "A", color='red', fontsize=8, weight='bold')
#     # normalize to unit 2D and draw (note: y axis points downward -> use -vy)
#     px, py = norm2d(g_pred[0], g_pred[1])
#     gx, gy = norm2d(g_gt[0],   g_gt[1])
#     ax.arrow(ox, oy, px*arrow_len, -py*arrow_len, width=1.5, head_width=8, head_length=10,
#              length_includes_head=True, color='lime')  # pred
#     ax.arrow(ox, oy, gx*arrow_len, -gy*arrow_len, width=1.5, head_width=8, head_length=10,
#              length_includes_head=True, color='red')   # GT

# def save_grid(tiles, cols, out_png, titles=None):
#     cols = max(1, cols)
#     rows = (len(tiles) + cols - 1) // cols
#     fig, axes = plt.subplots(rows, cols, figsize=(cols*3.6, rows*3.0))
#     axes = np.array(axes).reshape(rows, cols)
#     for i, ax in enumerate(axes.ravel()):
#         if i < len(tiles):
#             img_rgb, origin, g_pred, g_gt, ttl = tiles[i]
#             draw_tile(ax, img_rgb, origin, g_pred, g_gt)
#             if ttl: ax.set_title(ttl, fontsize=9)
#         else:
#             ax.axis("off")
#     plt.tight_layout(pad=0.05)
#     fig.savefig(out_png, dpi=220, bbox_inches="tight", pad_inches=0)
#     plt.close(fig)

# def collect_samples(test_h5_path):
#     items = []
#     with h5py.File(test_h5_path, "r") as f:
#         for arch in sorted(f.keys()):
#             g_arch = f[arch]
#             # sort frames numerically if possible
#             def _keynum(k):
#                 try: return int(k)
#                 except: return k
#             for frame_id in sorted(g_arch.keys(), key=_keynum):
#                 g = g_arch[frame_id]
#                 # required fields
#                 if not all(k in g for k in ("image", "gaze_vector_3D", "gaze_vector_gt")):
#                     continue
#                 item = {
#                     "arch": arch,
#                     "fid": int(g["frame_id"][()]) if "frame_id" in g else frame_id,
#                     "image": np.asarray(g["image"]),
#                     "g_pred": np.asarray(g["gaze_vector_3D"]),
#                     "g_gt":   np.asarray(g["gaze_vector_gt"]),
#                     "mask_gt": np.asarray(g["mask_gt"]) if "mask_gt" in g else None,
#                 }
#                 # preferred origin (GT pupil center)
#                 if "pupil_center_gt" in g:
#                     item["origin"] = tuple(np.asarray(g["pupil_center_gt"]).tolist())
#                 else:
#                     # fallback: recompute from eyeball+gaze
#                     if "eyeball_gt" in g:
#                         E = np.asarray(g["eyeball_gt"])  # [r, cx, cy, ...]
#                         r, cx, cy = float(E[0]), float(E[1]), float(E[2])
#                         vx, vy = norm2d(item["g_gt"][0], item["g_gt"][1])
#                         item["origin"] = (cx + r*vx, cy + r*vy)
#                     else:
#                         # fallback to image center
#                         H, W = item["image"].shape[:2]
#                         item["origin"] = (W/2.0, H/2.0)
#                 items.append(item)
#     return items

# def make_viz(test_h5, out_dir=None, limit=24, cols=4, arrow_len=60):
#     out_dir = out_dir or os.path.join(os.path.dirname(test_h5), "viz_from_testh5")
#     os.makedirs(out_dir, exist_ok=True)
#     items = collect_samples(test_h5)[:limit]
#     if not items:
#         print("No visualizable samples found."); return
#     tiles, titles = [], []
#     for it in items:
#         gray = to_uint8(it["image"])
#         over = overlay_mask(gray, it["mask_gt"])
#         tiles.append((over, it["origin"], it["g_pred"], it["g_gt"],
#                       f"{it['arch']} (frame {it['fid']})"))
#     out_png = os.path.join(out_dir, f"qual_grid_{len(tiles)}.png")
#     save_grid(tiles, cols=cols, out_png=out_png, titles=titles)
#     print(f"[OK] wrote {out_png}")

# if __name__ == "__main__":
#     import argparse
#     ap = argparse.ArgumentParser()
#     ap.add_argument("--test_h5", required=True)
#     ap.add_argument("--out_dir", default=None)
#     ap.add_argument("--limit", type=int, default=24)
#     ap.add_argument("--cols", type=int, default=4)
#     ap.add_argument("--arrow_len", type=float, default=60)
#     args = ap.parse_args()
#     make_viz(args.test_h5, args.out_dir, args.limit, args.cols, args.arrow_len)




##Working with original eye
#!/usr/bin/env python3
# import os, math, glob, h5py, numpy as np
# import matplotlib.pyplot as plt

# # ---------------- utils ----------------
# CAND_IMAGE_KEYS = ["Images", "images", "X"]
# CAND_MASK_KEYS  = ["Masks_noSkin", "Masks", "labels", "Segmentation", "semantics"]

# def to_uint8(gray):
#     g = np.asarray(gray)
#     if g.ndim == 3 and g.shape[0] == 1:  # (1,H,W)
#         g = g[0]
#     g = g.astype(np.float32)
#     g -= np.nanmin(g)
#     mx = np.nanmax(g)
#     if mx > 0:
#         g /= mx
#     return (g * 255.0).clip(0,255).astype(np.uint8)

# def mask_label_map(mask):
#     if mask is None: return (None, None)
#     u, cnt = np.unique(mask, return_counts=True)
#     uc = dict(zip(u.tolist(), cnt.tolist()))
#     uc.pop(0, None)  # drop background
#     if not uc: return (None, None)
#     order = sorted(uc.items(), key=lambda x: x[1])  # small..large
#     pupil_id = order[0][0]
#     iris_id  = order[-1][0] if len(order) > 1 else None
#     return iris_id, pupil_id

# def overlay_mask(gray, mask, alpha=0.45):
#     rgb = np.stack([gray,gray,gray], -1).astype(np.float32)
#     if mask is None:
#         return rgb.clip(0,255).astype(np.uint8)
#     iris_id, pupil_id = mask_label_map(mask)
#     if iris_id is not None:
#         m = (mask == iris_id)
#         rgb[m] = (1-alpha)*rgb[m] + alpha*np.array([128,0,128], np.float32)  # purple
#     if pupil_id is not None:
#         m = (mask == pupil_id)
#         rgb[m] = (1-alpha)*rgb[m] + alpha*np.array([255,215,0], np.float32)  # gold
#     return rgb.clip(0,255).astype(np.uint8)

# def norm2d(vx, vy, eps=1e-6):
#     n = math.hypot(vx, vy)
#     if n < eps: return 0.0, 0.0
#     return vx/n, vy/n

# def draw_tile(ax, img_rgb, origin, g_pred, g_gt, arrow_len=60):
#     ax.imshow(img_rgb)
#     ax.axis("off")
#     ox, oy = origin
#     # red origin marker "A"
#     ax.plot(ox, oy, marker='o', markersize=4, linewidth=0, color='red')
#     ax.text(ox+3, oy-3, "A", color='red', fontsize=8, weight='bold')
#     # normalize to unit 2D; image y-axis goes DOWN -> use -vy
#     px, py = norm2d(g_pred[0], g_pred[1])
#     gx, gy = norm2d(g_gt[0],   g_gt[1])
#     ax.arrow(ox, oy, px*arrow_len, -py*arrow_len, width=1.5, head_width=8, head_length=10,
#              length_includes_head=True, color='lime')  # predicted
#     ax.arrow(ox, oy, gx*arrow_len, -gy*arrow_len, width=1.5, head_width=8, head_length=10,
#              length_includes_head=True, color='red')   # GT

# def save_grid(tiles, cols, out_png, arrow_len=60):
#     cols = max(1, cols)
#     rows = (len(tiles) + cols - 1) // cols
#     fig, axes = plt.subplots(rows, cols, figsize=(cols*3.6, rows*3.0))
#     axes = np.array(axes).reshape(rows, cols)
#     for i, ax in enumerate(axes.ravel()):
#         if i < len(tiles):
#             img_rgb, origin, g_pred, g_gt, ttl = tiles[i]
#             draw_tile(ax, img_rgb, origin, g_pred, g_gt, arrow_len=arrow_len)
#             if ttl: ax.set_title(ttl, fontsize=9)
#         else:
#             ax.axis("off")
#     plt.tight_layout(pad=0.05)
#     fig.savefig(out_png, dpi=220, bbox_inches="tight", pad_inches=0)
#     plt.close(fig)

# # --------------- read from test_results.h5 ---------------
# def collect_samples(test_h5_path):
#     items = []
#     with h5py.File(test_h5_path, "r") as f:
#         for arch in sorted(f.keys()):
#             g_arch = f[arch]
#             def _keynum(k):
#                 try: return int(k)
#                 except: return k
#             for frame_id in sorted(g_arch.keys(), key=_keynum):
#                 g = g_arch[frame_id]
#                 need = ("image","gaze_vector_3D","gaze_vector_gt")
#                 if not all(k in g for k in need): continue
#                 item = {
#                     "arch": arch,
#                     "fid": int(g["frame_id"][()]) if "frame_id" in g else int(frame_id),
#                     "image": np.asarray(g["image"]),
#                     "g_pred": np.asarray(g["gaze_vector_3D"]),
#                     "g_gt":   np.asarray(g["gaze_vector_gt"]),
#                     "mask_gt":   np.asarray(g["mask_gt"]) if "mask_gt" in g else None,
#                     "mask_pred": np.asarray(g["mask_pred_rend"]) if "mask_pred_rend" in g else None,
#                     "eyeball_gt": np.asarray(g["eyeball_gt"]) if "eyeball_gt" in g else None,
#                     "pupil_center_gt": np.asarray(g["pupil_center_gt"]) if "pupil_center_gt" in g else None,
#                 }
#                 # origin point A
#                 if item["pupil_center_gt"] is not None:
#                     pc = item["pupil_center_gt"].astype(np.float32)
#                     item["origin"] = (float(pc[0]), float(pc[1]))
#                 elif item["eyeball_gt"] is not None:
#                     r, cx, cy = map(float, item["eyeball_gt"][:3])
#                     vx, vy = norm2d(item["g_gt"][0], item["g_gt"][1])
#                     item["origin"] = (cx + r*vx, cy + r*vy)
#                 else:
#                     H, W = item["image"].shape[:2]
#                     item["origin"] = (W/2.0, H/2.0)
#                 items.append(item)
#     return items

# # --------------- fetch ORIGINAL image/mask from dataset ---------------
# def find_image_key(h5):
#     for k in CAND_IMAGE_KEYS:
#         if k in h5: return k
#     # fallback: first (N,H,W) dataset
#     for k,v in h5.items():
#         if isinstance(v, h5py.Dataset) and v.ndim >= 3:
#             return k
#     return None

# def find_mask_key(h5):
#     for k in CAND_MASK_KEYS:
#         if k in h5: return k
#     return None

# def open_arch_file(h5_root, arch):
#     # 1) exact match
#     p = os.path.join(h5_root, f"{arch}.h5")
#     if os.path.exists(p): return h5py.File(p, "r")
#     # 2) any file containing arch
#     cand = glob.glob(os.path.join(h5_root, "**/*.h5"), recursive=True)
#     cand.sort(key=lambda x: (arch.lower() not in os.path.basename(x).lower(), len(x)))
#     for c in cand:
#         if arch.lower() in os.path.basename(c).lower():
#             try:
#                 return h5py.File(c, "r")
#             except:
#                 pass
#     return None

# def get_original_frame(h5_root, arch, fid):
#     if not h5_root: return None, None
#     f = open_arch_file(h5_root, arch)
#     if f is None: return None, None
#     try:
#         ik = find_image_key(f)
#         if ik is None: return None, None
#         imgs = f[ik]
#         idx = int(fid)
#         if not (0 <= idx < imgs.shape[0]):  # simple guard
#             return None, None
#         gray = np.asarray(imgs[idx])
#         mk = find_mask_key(f)
#         mask = np.asarray(f[mk][idx]) if (mk and idx < f[mk].shape[0]) else None
#         return gray, mask
#     except Exception:
#         return None, None
#     finally:
#         try: f.close()
#         except: pass

# # --------------- main viz ---------------
# def make_viz(test_h5, out_dir=None, limit=24, cols=4, arrow_len=60,
#              overlay="gt", alpha=0.45, h5_root=None):
#     """
#     overlay: 'none' | 'gt' | 'pred'
#     If h5_root is provided, use ORIGINAL dataset image; else use saved image.
#     """
#     out_dir = out_dir or os.path.join(os.path.dirname(test_h5), "viz_from_testh5")
#     os.makedirs(out_dir, exist_ok=True)

#     items = collect_samples(test_h5)[:limit]
#     if not items:
#         print("No visualizable samples found."); return

#     tiles = []
#     for it in items:
#         # Prefer ORIGINAL image from dataset if available
#         gray_ds, mask_ds = get_original_frame(h5_root, it["arch"], it["fid"]) if h5_root else (None, None)
#         if gray_ds is not None:
#             gray = to_uint8(gray_ds)
#             # prefer GT mask saved in test_results; fall back to dataset mask
#             mask_for_overlay = it["mask_gt"] if overlay == "gt" else (it["mask_pred"] if overlay == "pred" else None)
#             if mask_for_overlay is None:
#                 mask_for_overlay = mask_ds if overlay in ("gt","pred") else None
#         else:
#             # fallback to saved (possibly quantized) image
#             gray = to_uint8(it["image"])
#             mask_for_overlay = it["mask_gt"] if overlay == "gt" else (it["mask_pred"] if overlay == "pred" else None)

#         if overlay in ("gt","pred"):
#             img_rgb = overlay_mask(gray, mask_for_overlay, alpha=alpha)
#         else:
#             img_rgb = np.stack([gray,gray,gray], -1)

#         tiles.append((img_rgb, it["origin"], it["g_pred"], it["g_gt"],
#                       f"{it['arch']} (frame {it['fid']})"))

#     suffix = overlay if overlay in ("gt","pred") else "plain"
#     out_png = os.path.join(out_dir, f"qual_grid_{len(tiles)}_{suffix}.png")
#     save_grid(tiles, cols=cols, out_png=out_png, arrow_len=arrow_len)
#     print(f"[OK] wrote {out_png}")

# # --------------- CLI ---------------
# if __name__ == "__main__":
#     import argparse
#     ap = argparse.ArgumentParser()
#     ap.add_argument("--test_h5", required=True)
#     ap.add_argument("--out_dir", default=None)
#     ap.add_argument("--limit", type=int, default=24)
#     ap.add_argument("--cols", type=int, default=4)
#     ap.add_argument("--arrow_len", type=float, default=60)
#     ap.add_argument("--overlay", choices=["none","gt","pred"], default="gt",
#                     help="mask overlay source (default: gt)")
#     ap.add_argument("--alpha", type=float, default=0.45, help="overlay opacity")
#     ap.add_argument("--h5_root", default=None, help="path to original dataset H5s")
#     args = ap.parse_args()
#     make_viz(args.test_h5, args.out_dir, args.limit, args.cols,
#              args.arrow_len, args.overlay, args.alpha, args.h5_root)



##working with original eye and pupil center gaze vector
#!/usr/bin/env python3
# import os, math, glob, h5py, numpy as np
# import matplotlib.pyplot as plt

# # ---------------- utils ----------------
# CAND_IMAGE_KEYS = ["Images", "images", "X"]
# CAND_MASK_KEYS  = ["Masks_noSkin", "Masks", "labels", "Segmentation", "semantics"]

# def to_uint8(gray):
#     g = np.asarray(gray)
#     if g.ndim == 3 and g.shape[0] == 1:  # (1,H,W)
#         g = g[0]
#     g = g.astype(np.float32)
#     g -= np.nanmin(g)
#     mx = np.nanmax(g)
#     if mx > 0:
#         g /= mx
#     return (g * 255.0).clip(0,255).astype(np.uint8)

# def mask_label_map(mask):
#     if mask is None: return (None, None)
#     u, cnt = np.unique(mask, return_counts=True)
#     uc = dict(zip(u.tolist(), cnt.tolist()))
#     uc.pop(0, None)  # drop background
#     if not uc: return (None, None)
#     order = sorted(uc.items(), key=lambda x: x[1])  # small..large
#     pupil_id = order[0][0]
#     iris_id  = order[-1][0] if len(order) > 1 else None
#     return iris_id, pupil_id

# def overlay_mask(gray, mask, alpha=0.45):
#     rgb = np.stack([gray,gray,gray], -1).astype(np.float32)
#     if mask is None:
#         return rgb.clip(0,255).astype(np.uint8)
#     iris_id, pupil_id = mask_label_map(mask)
#     if iris_id is not None:
#         m = (mask == iris_id)
#         rgb[m] = (1-alpha)*rgb[m] + alpha*np.array([128,0,128], np.float32)  # purple
#     if pupil_id is not None:
#         m = (mask == pupil_id)
#         rgb[m] = (1-alpha)*rgb[m] + alpha*np.array([255,215,0], np.float32)  # gold
#     return rgb.clip(0,255).astype(np.uint8)

# def norm2d(vx, vy, eps=1e-6):
#     n = math.hypot(vx, vy)
#     if n < eps: return 0.0, 0.0
#     return vx/n, vy/n

# def draw_tile(ax, img_rgb, origin_gt, origin_pred, g_pred, g_gt,
#               arrow_len=60, origin_mode="gt"):
#     ax.imshow(img_rgb)
#     ax.axis("off")

#     # Choose start points for arrows
#     if origin_mode == "gt":
#         op = og = origin_gt
#     elif origin_mode == "pred":
#         op = og = origin_pred if origin_pred is not None else origin_gt
#     else:  # "separate"
#         og = origin_gt
#         op = origin_pred if origin_pred is not None else origin_gt

#     # markers
#     if origin_mode in ("gt", "pred"):
#         ox, oy = og
#         ax.plot(ox, oy, marker='o', markersize=4, linewidth=0, color='red')
#         ax.text(ox+3, oy-3, "A", color='red', fontsize=8, weight='bold')
#     else:
#         # show both
#         gx, gy = og
#         px, py = op
#         ax.plot(gx, gy, marker='o', markersize=4, linewidth=0, color='red')
#         ax.text(gx+3, gy-3, "A", color='red', fontsize=8, weight='bold')   # GT center
#         ax.plot(px, py, marker='o', markersize=4, linewidth=0, color='lime')
#         ax.text(px+3, py-3, "P", color='lime', fontsize=8, weight='bold')  # Pred center

#     # normalize to unit 2D; image y-axis goes DOWN -> use -vy
#     px, py = norm2d(g_pred[0], g_pred[1])
#     gx, gy = norm2d(g_gt[0],   g_gt[1])

#     # arrows
#     # predicted
#     ax.arrow(op[0], op[1], px*arrow_len, -py*arrow_len,
#              width=1.5, head_width=8, head_length=10,
#              length_includes_head=True, color='lime')
#     # GT
#     ax.arrow(og[0], og[1], gx*arrow_len, -gy*arrow_len,
#              width=1.5, head_width=8, head_length=10,
#              length_includes_head=True, color='red')

# def save_grid(tiles, cols, out_png, arrow_len=60, origin_mode="gt"):
#     cols = max(1, cols)
#     rows = (len(tiles) + cols - 1) // cols
#     fig, axes = plt.subplots(rows, cols, figsize=(cols*3.6, rows*3.0))
#     axes = np.array(axes).reshape(rows, cols)
#     for i, ax in enumerate(axes.ravel()):
#         if i < len(tiles):
#             img_rgb, origin_gt, origin_pred, g_pred, g_gt, ttl = tiles[i]
#             draw_tile(ax, img_rgb, origin_gt, origin_pred, g_pred, g_gt,
#                       arrow_len=arrow_len, origin_mode=origin_mode)
#             if ttl: ax.set_title(ttl, fontsize=9)
#         else:
#             ax.axis("off")
#     plt.tight_layout(pad=0.05)
#     fig.savefig(out_png, dpi=220, bbox_inches="tight", pad_inches=0)
#     plt.close(fig)

# # --------------- read from test_results.h5 ---------------
# def collect_samples(test_h5_path):
#     items = []
#     with h5py.File(test_h5_path, "r") as f:
#         for arch in sorted(f.keys()):
#             g_arch = f[arch]
#             def _keynum(k):
#                 try: return int(k)
#                 except: return k
#             for frame_id in sorted(g_arch.keys(), key=_keynum):
#                 g = g_arch[frame_id]
#                 need = ("image","gaze_vector_3D","gaze_vector_gt")
#                 if not all(k in g for k in need): continue

#                 # centers (prefer explicit UV fields if present)
#                 pc_gt = None
#                 pc_pred = None
#                 if "pupil_c_UV" in g:         pc_gt   = np.asarray(g["pupil_c_UV"])
#                 if "pupil_center_gt" in g:    pc_gt   = np.asarray(g["pupil_center_gt"]) if pc_gt is None else pc_gt
#                 if "pupil_c_UV_pred" in g:    pc_pred = np.asarray(g["pupil_c_UV_pred"])

#                 item = {
#                     "arch": arch,
#                     "fid": int(g["frame_id"][()]) if "frame_id" in g else int(frame_id),
#                     "image": np.asarray(g["image"]),
#                     "g_pred": np.asarray(g["gaze_vector_3D"]),
#                     "g_gt":   np.asarray(g["gaze_vector_gt"]),
#                     "mask_gt":   np.asarray(g["mask_gt"]) if "mask_gt" in g else None,
#                     "mask_pred": np.asarray(g["mask_pred_rend"]) if "mask_pred_rend" in g else None,
#                     "eyeball_gt": np.asarray(g["eyeball_gt"]) if "eyeball_gt" in g else None,
#                     "pc_gt": pc_gt,
#                     "pc_pred": pc_pred,
#                 }

#                 # if no GT center saved, recompute from eyeball + gt gaze
#                 if item["pc_gt"] is None:
#                     if item["eyeball_gt"] is not None:
#                         E = np.asarray(item["eyeball_gt"])
#                         r, cx, cy = float(E[0]), float(E[1]), float(E[2])
#                         vx, vy = norm2d(item["g_gt"][0], item["g_gt"][1])
#                         item["pc_gt"] = np.array([cx + r*vx, cy + r*vy], dtype=np.float32)
#                     else:
#                         H, W = item["image"].shape[:2]
#                         item["pc_gt"] = np.array([W/2.0, H/2.0], dtype=np.float32)

#                 items.append(item)
#     return items

# # --------------- fetch ORIGINAL image/mask from dataset ---------------
# def find_image_key(h5):
#     for k in CAND_IMAGE_KEYS:
#         if k in h5: return k
#     # fallback: first (N,H,W) dataset
#     for k,v in h5.items():
#         if isinstance(v, h5py.Dataset) and v.ndim >= 3:
#             return k
#     return None

# def find_mask_key(h5):
#     for k in CAND_MASK_KEYS:
#         if k in h5: return k
#     return None

# def open_arch_file(h5_root, arch):
#     p = os.path.join(h5_root, f"{arch}.h5")
#     if os.path.exists(p): return h5py.File(p, "r")
#     cand = glob.glob(os.path.join(h5_root, "**/*.h5"), recursive=True)
#     cand.sort(key=lambda x: (arch.lower() not in os.path.basename(x).lower(), len(x)))
#     for c in cand:
#         if arch.lower() in os.path.basename(c).lower():
#             try:
#                 return h5py.File(c, "r")
#             except:
#                 pass
#     return None

# def get_original_frame(h5_root, arch, fid):
#     if not h5_root: return None, None
#     f = open_arch_file(h5_root, arch)
#     if f is None: return None, None
#     try:
#         ik = find_image_key(f)
#         if ik is None: return None, None
#         imgs = f[ik]
#         idx = int(fid)
#         if not (0 <= idx < imgs.shape[0]):
#             return None, None
#         gray = np.asarray(imgs[idx])
#         mk = find_mask_key(f)
#         mask = np.asarray(f[mk][idx]) if (mk and idx < f[mk].shape[0]) else None
#         return gray, mask
#     except Exception:
#         return None, None
#     finally:
#         try: f.close()
#         except: pass

# # --------------- main viz ---------------
# def make_viz(test_h5, out_dir=None, limit=24, cols=4, arrow_len=60,
#              overlay="gt", alpha=0.45, h5_root=None, origin="gt"):
#     """
#     overlay: 'none' | 'gt' | 'pred'
#     origin : 'gt' | 'pred' | 'separate'
#     If h5_root is provided, use ORIGINAL dataset image; else use saved image.
#     """
#     out_dir = out_dir or os.path.join(os.path.dirname(test_h5), "viz_from_testh5")
#     os.makedirs(out_dir, exist_ok=True)

#     items = collect_samples(test_h5)[:limit]
#     if not items:
#         print("No visualizable samples found."); return

#     tiles = []
#     for it in items:
#         # Prefer ORIGINAL image from dataset if available
#         gray_ds, mask_ds = get_original_frame(h5_root, it["arch"], it["fid"]) if h5_root else (None, None)
#         if gray_ds is not None:
#             gray = to_uint8(gray_ds)
#             mask_for_overlay = it["mask_gt"] if overlay == "gt" else (it["mask_pred"] if overlay == "pred" else None)
#             if mask_for_overlay is None:
#                 mask_for_overlay = mask_ds if overlay in ("gt","pred") else None
#         else:
#             # fallback to saved (possibly quantized) image
#             gray = to_uint8(it["image"])
#             mask_for_overlay = it["mask_gt"] if overlay == "gt" else (it["mask_pred"] if overlay == "pred" else None)

#         if overlay in ("gt","pred"):
#             img_rgb = overlay_mask(gray, mask_for_overlay, alpha=alpha)
#         else:
#             img_rgb = np.stack([gray,gray,gray], -1)

#         tiles.append((img_rgb, tuple(it["pc_gt"]), None if it["pc_pred"] is None else tuple(it["pc_pred"]),
#                       it["g_pred"], it["g_gt"], f"{it['arch']} (frame {it['fid']})"))

#     suffix = f"{overlay}_{origin}" if overlay in ("gt","pred") else f"plain_{origin}"
#     out_png = os.path.join(out_dir, f"qual_grid_{len(tiles)}_{suffix}.png")
#     save_grid(tiles, cols=cols, out_png=out_png, arrow_len=arrow_len, origin_mode=origin)
#     print(f"[OK] wrote {out_png}")

# # --------------- CLI ---------------
# if __name__ == "__main__":
#     import argparse
#     ap = argparse.ArgumentParser()
#     ap.add_argument("--test_h5", required=True)
#     ap.add_argument("--out_dir", default=None)
#     ap.add_argument("--limit", type=int, default=24)
#     ap.add_argument("--cols", type=int, default=4)
#     ap.add_argument("--arrow_len", type=float, default=60)
#     ap.add_argument("--overlay", choices=["none","gt","pred"], default="gt",
#                     help="mask overlay source (default: gt)")
#     ap.add_argument("--alpha", type=float, default=0.45, help="overlay opacity")
#     ap.add_argument("--h5_root", default=None, help="path to original dataset H5s")
#     ap.add_argument("--origin", choices=["gt","pred","separate"], default="gt",
#                     help="where arrows start from (default: gt pupil center)")
#     args = ap.parse_args()
#     make_viz(args.test_h5, args.out_dir, args.limit, args.cols,
#              args.arrow_len, args.overlay, args.alpha, args.h5_root, args.origin)


#!/usr/bin/env python3
#!/usr/bin/env python3
#!/usr/bin/env python3

#!/usr/bin/env python3
import os, math, h5py, numpy as np
import matplotlib.pyplot as plt

# ---------------- utils ----------------
def to_uint8(gray):
    g = np.asarray(gray)
    if g.ndim == 3 and g.shape[0] == 1:  # (1,H,W) -> (H,W)
        g = g[0]
    if g.dtype == np.uint8:
        return g
    g = g.astype(np.float32)
    g -= np.nanmin(g)
    mx = np.nanmax(g)
    if mx > 0:
        g /= mx
    return (g * 255.0).clip(0, 255).astype(np.uint8)

def mask_label_map(mask):
    if mask is None: return (None, None)
    u, cnt = np.unique(mask, return_counts=True)
    uc = dict(zip(u.tolist(), cnt.tolist()))
    uc.pop(0, None)  # drop background if present
    if not uc: return (None, None)
    order = sorted(uc.items(), key=lambda x: x[1])  # small..large
    pupil_id = order[0][0]
    iris_id  = order[-1][0] if len(order) > 1 else None
    return iris_id, pupil_id

def overlay_mask(gray, mask, alpha=0.45):
    rgb = np.stack([gray, gray, gray], -1).astype(np.float32)
    if mask is None:
        return rgb.clip(0,255).astype(np.uint8)
    iris_id, pupil_id = mask_label_map(mask)
    if iris_id is not None:
        m = (mask == iris_id)
        rgb[m] = (1-alpha)*rgb[m] + alpha*np.array([128, 0, 128], np.float32)  # purple
    if pupil_id is not None:
        m = (mask == pupil_id)
        rgb[m] = (1-alpha)*rgb[m] + alpha*np.array([255, 215, 0], np.float32)  # gold
    return rgb.clip(0,255).astype(np.uint8)

def norm2d(vx, vy, eps=1e-6):
    n = math.hypot(vx, vy)
    if n < eps: return 0.0, 0.0
    return vx/n, vy/n

def draw_tile(ax, img_rgb, origin_gt, origin_pred, g_pred, g_gt,
              arrow_len=60, origin_mode="gt"):
    ax.imshow(img_rgb)
    ax.axis("off")

    # Choose start points for arrows
    if origin_mode == "gt":
        op = og = origin_gt
    elif origin_mode == "pred":
        op = og = origin_pred if origin_pred is not None else origin_gt
    else:  # "separate"
        og = origin_gt
        op = origin_pred if origin_pred is not None else origin_gt

    # markers
    if origin_mode in ("gt", "pred"):
        ox, oy = og
        ax.plot(ox, oy, marker='o', markersize=4, linewidth=0, color='red')
        ax.text(ox+3, oy-3, "A", color='red', fontsize=8, weight='bold')
    else:
        gx, gy = og
        px, py = op
        ax.plot(gx, gy, marker='o', markersize=4, linewidth=0, color='red')
        ax.text(gx+3, gy-3, "A", color='red', fontsize=8, weight='bold')   # GT center
        ax.plot(px, py, marker='o', markersize=4, linewidth=0, color='lime')
        ax.text(px+3, py-3, "P", color='lime', fontsize=8, weight='bold')  # Pred center

    # normalize to unit 2D; image y-axis goes DOWN -> use -vy for display
    px, py = norm2d(g_pred[0], g_pred[1])
    gx, gy = norm2d(g_gt[0],   g_gt[1])

    # arrows
    ax.arrow(op[0], op[1], px*arrow_len, -py*arrow_len,
             width=1.5, head_width=8, head_length=10,
             length_includes_head=True, color='lime')
    ax.arrow(og[0], og[1], gx*arrow_len, -gy*arrow_len,
             width=1.5, head_width=8, head_length=10,
             length_includes_head=True, color='red')

def save_grid(tiles, cols, out_png, arrow_len=60, origin_mode="gt"):
    cols = max(1, cols)
    rows = (len(tiles) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols*3.6, rows*3.0))
    axes = np.array(axes).reshape(rows, cols)
    for i, ax in enumerate(axes.ravel()):
        if i < len(tiles):
            img_rgb, origin_gt, origin_pred, g_pred, g_gt, ttl = tiles[i]
            draw_tile(ax, img_rgb, origin_gt, origin_pred, g_pred, g_gt,
                      arrow_len=arrow_len, origin_mode=origin_mode)
            if ttl:
                ax.set_title(ttl, fontsize=9)
        else:
            ax.axis("off")
    plt.tight_layout(pad=0.05)
    fig.savefig(out_png, dpi=220, bbox_inches="tight", pad_inches=0)
    plt.close(fig)

# --------------- read from test_results.h5 ---------------
def collect_samples(test_h5_path):
    """
    Collects samples, preferring `image_raw_u8` for visualization.
    Falls back to `image` if `image_raw_u8` is absent.
    """
    items = []
    with h5py.File(test_h5_path, "r") as f:
        for arch in sorted(f.keys()):
            g_arch = f[arch]
            # sort frame groups numerically if possible
            def _keynum(k):
                try: return int(k)
                except: return k
            for frame_key in sorted(g_arch.keys(), key=_keynum):
                g = g_arch[frame_key]

                # Need gaze vectors to draw arrows
                need = ("gaze_vector_3D", "gaze_vector_gt")
                if not all(k in g for k in need):
                    continue

                # Prefer RAW image saved by your test script
                if "image_raw_u8" in g:
                    img = np.asarray(g["image_raw_u8"])
                elif "image" in g:
                    img = np.asarray(g["image"])
                else:
                    continue  # nothing to visualize

                # centers (prefer explicit UV fields if present)
                pc_gt = None
                pc_pred = None
                if "pupil_c_UV" in g:         pc_gt   = np.asarray(g["pupil_c_UV"])
                if "pupil_center_gt" in g:    pc_gt   = np.asarray(g["pupil_center_gt"]) if pc_gt is None else pc_gt
                if "pupil_c_UV_pred" in g:    pc_pred = np.asarray(g["pupil_c_UV_pred"])

                # basic payload
                item = {
                    "arch": arch,
                    "fid": int(g["frame_id"][()]) if "frame_id" in g else int(frame_key) if str(frame_key).isdigit() else frame_key,
                    "img": img,
                    "g_pred": np.asarray(g["gaze_vector_3D"]),
                    "g_gt":   np.asarray(g["gaze_vector_gt"]),
                    "mask_gt":   np.asarray(g["mask_gt"]) if "mask_gt" in g else None,
                    "mask_pred": np.asarray(g["mask_pred_rend"]) if "mask_pred_rend" in g else None,
                    "eyeball_gt": np.asarray(g["eyeball_gt"]) if "eyeball_gt" in g else None,
                    "pc_gt": pc_gt,
                    "pc_pred": pc_pred,
                }

                # if no GT center saved, recompute from eyeball + gt gaze when possible
                if item["pc_gt"] is None:
                    if item["eyeball_gt"] is not None:
                        E = np.asarray(item["eyeball_gt"])
                        r, cx, cy = float(E[0]), float(E[1]), float(E[2])
                        vx, vy = norm2d(item["g_gt"][0], item["g_gt"][1])
                        item["pc_gt"] = np.array([cx + r*vx, cy + r*vy], dtype=np.float32)
                    else:
                        H, W = item["img"].shape[:2]
                        item["pc_gt"] = np.array([W/2.0, H/2.0], dtype=np.float32)

                items.append(item)
    return items

# --------------- main viz ---------------
def make_viz(test_h5, out_dir=None, limit=24, cols=4, arrow_len=60,
             overlay="gt", alpha=0.45, origin="gt"):
    """
    overlay: 'none' | 'gt' | 'pred'  (mask overlay source)
    origin : 'gt' | 'pred' | 'separate' (arrow starting point)
    Uses image_raw_u8 if present; else uses image.
    """
    out_dir = out_dir or os.path.join(os.path.dirname(test_h5), "viz_from_testh5_raw")
    os.makedirs(out_dir, exist_ok=True)

    items = collect_samples(test_h5)
    if not items:
        print("No visualizable samples found."); return

    items = items[:limit]

    tiles = []
    for it in items:
        gray = to_uint8(it["img"])
        if overlay == "gt":
            mask_for_overlay = it["mask_gt"]
        elif overlay == "pred":
            mask_for_overlay = it["mask_pred"]
        else:
            mask_for_overlay = None

        if overlay in ("gt", "pred"):
            img_rgb = overlay_mask(gray, mask_for_overlay, alpha=alpha)
        else:
            img_rgb = np.stack([gray, gray, gray], -1)

        tiles.append((
            img_rgb,
            tuple(it["pc_gt"]),
            None if it["pc_pred"] is None else tuple(it["pc_pred"]),
            it["g_pred"], it["g_gt"],
            f"{it['arch']} (frame {it['fid']})"
        ))

    suffix = f"{overlay}_{origin}" if overlay in ("gt","pred") else f"plain_{origin}"
    out_png = os.path.join(out_dir, f"qual_grid_{len(tiles)}_{suffix}.png")
    save_grid(tiles, cols=cols, out_png=out_png, arrow_len=arrow_len, origin_mode=origin)
    print(f"[OK] wrote {out_png}")

# --------------- CLI ---------------
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--test_h5", required=True)
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--limit", type=int, default=24)
    ap.add_argument("--cols", type=int, default=4)
    ap.add_argument("--arrow_len", type=float, default=60)
    ap.add_argument("--overlay", choices=["none","gt","pred"], default="gt",
                    help="mask overlay source (default: gt)")
    ap.add_argument("--alpha", type=float, default=0.45, help="overlay opacity")
    ap.add_argument("--origin", choices=["gt","pred","separate"], default="gt",
                    help="where arrows start from (default: gt pupil center)")
    args = ap.parse_args()
    make_viz(args.test_h5, args.out_dir, args.limit, args.cols,
             args.arrow_len, args.overlay, args.alpha, args.origin)








