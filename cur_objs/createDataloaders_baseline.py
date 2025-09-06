#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
This started as a copy of https://bitbucket.org/RSKothari/multiset_gaze/src/master/ 
with additional changes and modifications to adjust it to our implementation. 

Copyright (c) 2021 Rakshit Kothari, Aayush Chaudhary, Reynold Bailey, Jeff Pelz, 
and Gabriel Diaz

This file generates objects with train and testing split information for each
dataset. Each dataset has a predefined train and test partition. For more info
on the partitions, please see the file <datasetSelections.py>
'''

# import os
# import sys
# import pickle
# import numpy as np

# sys.path.append('..')
# import helperfunctions.CurriculumLib as CurLib
# from helperfunctions.CurriculumLib import DataLoader_riteyes

# path2data = '/media/ml/Data Disk/Anjali/Model-aware_3D_Eye_Gaze-main/Datasets'
# path2h5 = os.path.join(path2data, 'All')
# keepOld = False

# DS_sel = pickle.load(open('dataset_selections.pkl', 'rb'))
# print(DS_sel['train'].get('TEyeD', []))
# AllDS = CurLib.readArchives(os.path.join(path2data, 'MasterKey'))
# print("Type of AllDS:", type(AllDS))
# print("First entry:", AllDS[0])
# #list_ds = ['OpenEDS','sequence', 'S']
# list_ds = ['TEyeD']

# args={}
# args['train_data_percentage'] = 1.0
# args['net_ellseg_head'] =False
# args['loss_w_rend_pred_2_gt_edge'] = 0.1
# args['loss_w_rend_gt_2_pred'] = 0.1
# args['loss_w_rend_pred_2_gt'] = 0.0
# args['net_ellseg_head'] = 0.0

# # Generate objects per dataset
# for setSel in list_ds:
#     # Train object
#     AllDS_cond = CurLib.selSubset(AllDS, DS_sel['train'][setSel])
#     dataDiv_obj = CurLib.generate_fileList(AllDS_cond, mode='none', notest=True)
#     trainObj = DataLoader_riteyes(dataDiv_obj, path2h5, 'train', True, (480, 640), 
#                                   scale=0.5, num_frames=4, args=args)
#     validObj = DataLoader_riteyes(dataDiv_obj, path2h5, 'valid', False, (480, 640), 
#                                   scale=0.5, num_frames=4,args=args)
#     # Test object
#     AllDS_cond = CurLib.selSubset(AllDS, DS_sel['test'][setSel])
#     dataDiv_obj = CurLib.generate_fileList(AllDS_cond, mode='none', notest=True)
#     testObj = DataLoader_riteyes(dataDiv_obj, path2h5, 'test', False, (480, 640), 
#                                  scale=0.5, num_frames=4, args=args)

#     if setSel == 'S':
#         path2save = os.path.join(os.getcwd(), 'one_vs_one', 'cond_'+'OpenEDS_S'+'.pkl')
#     else:
#         path2save = os.path.join(os.getcwd(), 'one_vs_one', 'cond_'+setSel+'.pkl')
#     if os.path.exists(path2save) and keepOld:
#         print('Preserving old selections ...')

#         # This ensure that the original selection remains the same
#         trainObj_orig, validObj_orig, testObj_orig = pickle.load(open(path2save, 'rb'))
#         trainObj.imList = trainObj_orig.imList
#         validObj.imList = validObj_orig.imList
#         testObj.imList = testObj_orig.imList
#         pickle.dump((trainObj, validObj, testObj), open(path2save, 'wb'))
#     else:
#         print('Save data')
#         pickle.dump((trainObj, validObj, testObj), open(path2save, 'wb'))
# import os
# import sys
# import pickle
# import numpy as np

# sys.path.append('..')
# import helperfunctions.CurriculumLib as CurLib
# from helperfunctions.CurriculumLib import DataLoader_riteyes

# path2data = '/media/ml/03C90A2604F5F8BA/Anjali/Model-aware_3D_Eye_Gaze-main/Datasets'
# path2h5 = os.path.join(path2data, 'All')
# keepOld = False

# # Read master archive data
# AllDS = CurLib.readArchives(os.path.join(path2data, 'MasterKey'))
# print("Loaded AllDS with keys:", list(AllDS.keys()))
# print("Unique values in AllDS['archive']:", np.unique(AllDS['archive']))

# # Recordings to keep
# recordings = ['DikablisR_1_1', 'DikablisR_1_2', 'DikablisSA_1_1', 'DikablisSA_1_2',
#  'DikablisSA_2_1', 'DikablisSA_2_2', 'DikablisSS_1_1', 'DikablisSS_2_1',
#  'DikablisT_10_1', 'DikablisT_10_2']

# # Create mask for selected recordings
# mask = np.isin(AllDS['archive'], recordings)
# AllDS_filtered = {k: v[mask] for k, v in AllDS.items()}
# print(f"Filtered entries: {len(AllDS_filtered['archive'])}")
# print(f"Archives in filtered data: {np.unique(AllDS_filtered['archive'])}")

# # Early exit if no data found
# if len(AllDS_filtered['archive']) == 0:
#     print("❌ No valid entries found for specified recordings. Exiting.")
#     sys.exit(1)

# # Set dataloader args
# args = {
#     'train_data_percentage': 1.0,
#     'net_ellseg_head': False,
#     'loss_w_rend_pred_2_gt_edge': 0.1,
#     'loss_w_rend_gt_2_pred': 0.1,
#     'loss_w_rend_pred_2_gt': 0.0,
#     'net_ellseg_head': 0.0
# }

# # Split into train / valid / test
# dataDiv_obj = CurLib.generate_fileList(AllDS_filtered, mode='none', notest=True)
# trainObj = DataLoader_riteyes(dataDiv_obj, path2h5, 'train', True, (480, 640), scale=0.5, num_frames=4, args=args)
# validObj = DataLoader_riteyes(dataDiv_obj, path2h5, 'valid', False, (480, 640), scale=0.5, num_frames=4, args=args)
# testObj = DataLoader_riteyes(dataDiv_obj, path2h5, 'test', False, (480, 640), scale=0.5, num_frames=4, args=args)

# # Save
# os.makedirs('one_vs_one', exist_ok=True)
# path2save = os.path.join(os.getcwd(), 'one_vs_one', 'cond_DikablisR.pkl')

# if os.path.exists(path2save) and keepOld:
#     print("Preserving old selections ...")
#     trainObj_orig, validObj_orig, testObj_orig = pickle.load(open(path2save, 'rb'))
#     trainObj.imList = trainObj_orig.imList
#     validObj.imList = validObj_orig.imList
#     testObj.imList = testObj_orig.imList

# print("✅ Saving dataloaders to:", path2save)
# pickle.dump((trainObj, validObj, testObj), open(path2save, 'wb'))
#!/usr/bin/env python3
import os, sys, pickle, copy, argparse
import numpy as np

sys.path.append('..')
import helperfunctions.CurriculumLib as CurLib
from helperfunctions.CurriculumLib import DataLoader_riteyes

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path2data", default="/media/ml/03C90A2604F5F8BA/Anjali/Model-aware_3D_Eye_Gaze-main/Datasets")
    ap.add_argument("--recordings", nargs="+", default=[
        "DikablisR_1_1","DikablisR_1_2","DikablisSA_1_1","DikablisSA_1_2",
        "DikablisSA_2_1","DikablisSA_2_2","DikablisSS_1_1","DikablisSS_2_1",
        "DikablisT_10_1","DikablisT_10_2"
    ])
    ap.add_argument("--ratios", type=str, default="0.7,0.15,0.15",
                    help="train,valid,test ratios (sum≈1)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--scale", type=float, default=0.5)
    ap.add_argument("--frames", type=int, default=4)
    ap.add_argument("--out", default="one_vs_one/cond_DikablisR_disjoint.pkl")
    return ap.parse_args()

def filtered_copy(ds, keep_vids_by_arch):
    ds2 = copy.deepcopy(ds)
    keep = []
    for seq in ds.imList:              # seq shape: (F, 3) => [arch_id, video_id, frame_id]
        a = int(seq[0,0]); v = int(seq[0,1])
        if v in keep_vids_by_arch.get(a, set()):
            keep.append(seq)
    ds2.imList = np.asarray(keep, dtype=ds.imList.dtype)
    return ds2

def main():
    args = parse_args()
    r_tr, r_va, r_te = [float(x) for x in args.ratios.split(",")]
    if not (0 < r_tr <= 1 and 0 <= r_va <= 1 and 0 <= r_te <= 1):
        raise ValueError("Ratios must be in (0,1].")
    if abs((r_tr + r_va + r_te) - 1.0) > 1e-6:
        # Soft-normalize if user passed e.g. 7,1.5,1.5
        s = (r_tr + r_va + r_te)
        r_tr, r_va, r_te = r_tr/s, r_va/s, r_te/s

    path2h5 = os.path.join(args.path2data, "All")

    # Read the master table and filter by recordings you want
    AllDS = CurLib.readArchives(os.path.join(args.path2data, "MasterKey"))
    mask = np.isin(AllDS["archive"], args.recordings)
    AllDS_filtered = {k: v[mask] for k, v in AllDS.items()}
    if len(AllDS_filtered["archive"]) == 0:
        print("❌ No valid entries for specified recordings.")
        sys.exit(1)

    # Build one base object (same pool) then split by video IDs
    cfg = {
        "train_data_percentage": 1.0,
        "net_ellseg_head": False,
        "loss_w_rend_pred_2_gt_edge": 0.1,
        "loss_w_rend_gt_2_pred": 0.1,
        "loss_w_rend_pred_2_gt": 0.0,
        "net_ellseg_head": 0.0,
    }
    dataDiv_obj = CurLib.generate_fileList(AllDS_filtered, mode="none", notest=True)

    base_train = DataLoader_riteyes(dataDiv_obj, path2h5, "train", True,  (480,640),
                                    scale=args.scale, num_frames=args.frames, args=cfg)
    base_valid = DataLoader_riteyes(dataDiv_obj, path2h5, "valid", False, (480,640),
                                    scale=args.scale, num_frames=args.frames, args=cfg)
    base_test  = DataLoader_riteyes(dataDiv_obj, path2h5, "test",  False, (480,640),
                                    scale=args.scale, num_frames=args.frames, args=cfg)

    # All three currently include the same sequences; split them disjointly by video id per arch
    im = base_train.imList                          # shape: (num_sequences, F, 3)
    F  = im.shape[1]
    arch_ids = np.unique(im[:,0,0]).astype(int)
    rng = np.random.default_rng(args.seed)

    vids_by_arch = {}
    for a in arch_ids:
        vids_a = np.unique(im[im[:,0,0]==a][:,0,1]).astype(int)
        rng.shuffle(vids_a)
        n = len(vids_a)
        n_tr = int(np.floor(r_tr*n))
        n_va = int(np.floor(r_va*n))
        # put the remainder to test to guarantee coverage
        n_te = max(0, n - (n_tr + n_va))
        # Edge handling: ensure train gets at least 1 if possible
        if n > 0 and n_tr == 0: n_tr = min(1, n)
        if n - n_tr > 0 and n_va == 0 and r_va > 0:  # give valid one if ratio > 0 and we still have vids
            n_va = min(1, n - n_tr)
        n_te = max(0, n - (n_tr + n_va))

        vids_by_arch[a] = {
            "train": set(vids_a[:n_tr].tolist()),
            "valid": set(vids_a[n_tr:n_tr+n_va].tolist()),
            "test":  set(vids_a[n_tr+n_va:].tolist()),
        }

    train_new = filtered_copy(base_train, {a:s["train"] for a,s in vids_by_arch.items()})
    valid_new = filtered_copy(base_valid, {a:s["valid"] for a,s in vids_by_arch.items()})
    test_new  = filtered_copy(base_test,  {a:s["test"]  for a,s in vids_by_arch.items()})

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    pickle.dump((train_new, valid_new, test_new), open(args.out, "wb"))
    print("✅ Saved:", args.out)

    # Quick report
    def count_images(ds): 
        return ds.imList.shape[0], ds.imList.shape[1], ds.imList.shape[0]*ds.imList.shape[1]
    nseq_tr, Ftr, Nim_tr = count_images(train_new)
    nseq_va, Fva, Nim_va = count_images(valid_new)
    nseq_te, Fte, Nim_te = count_images(test_new)
    print(f"train: sequences={nseq_tr}, frames/seq={Ftr}, images={Nim_tr}")
    print(f"valid: sequences={nseq_va}, frames/seq={Fva}, images={Nim_va}")
    print(f"test : sequences={nseq_te}, frames/seq={Fte}, images={Nim_te}")

    # Sanity check: no overlap (by (arch,video))
    def av_pairs(ds):
        av = ds.imList[:,0,:2]  # (N, 2) [arch, video]
        return set(map(tuple, av.astype(int)))
    tr_av, va_av, te_av = av_pairs(train_new), av_pairs(valid_new), av_pairs(test_new)
    print("overlap train∩valid:", len(tr_av & va_av))
    print("overlap train∩test :", len(tr_av & te_av))
    print("overlap valid∩test :", len(va_av & te_av))

if __name__ == "__main__":
    main()
