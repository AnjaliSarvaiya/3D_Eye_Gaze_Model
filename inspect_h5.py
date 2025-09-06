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
import h5py, numpy as np
p="/media/ml/03C90A2604F5F8BA/Anjali/Model-aware_3D_Eye_Gaze-main/Results/pretrained_sem/results/test_results.h5"
with h5py.File(p,"r") as f:
    g=f["DikablisR_1_1"]["1258"]
    img=np.asarray(g["image"])
    print(img.dtype, img.min(), img.max(), np.unique(img)[:10], len(np.unique(img)))
