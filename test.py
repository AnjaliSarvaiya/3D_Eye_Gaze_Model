#!/usr/bin/env python3

import os, json, pickle
import torch
from torch.utils.data import DataLoader

from tensorboardX import SummaryWriter  # not used, but forward signature matches
from helperfunctions.utils import make_logger, move_to_single, FRN_TLU, do_nothing, get_nparams
import helperfunctions.CurriculumLib as CurLib
from helperfunctions.CurriculumLib import DataLoader_riteyes
from SRC.models_mux import model_dict
from scripts import forward
import numpy as np
import wandb
import config_test as cfg

# import os
os.environ["WANDB_DISABLED"] = "true"


def ensure_dirs(paths):
    for p in [paths["exp"], paths["results"], paths["logs"], paths["figures"]]:
        os.makedirs(p, exist_ok=True)

def build_net(args):
    # mirror main.py’s norm selection (simplified)
    if args['use_frn_tlu']:
        norm = FRN_TLU; act = do_nothing
    elif args['use_instance_norm']:
        norm = torch.nn.InstanceNorm3d if args['model'] == 'DenseElNet' else torch.nn.InstanceNorm2d
        act  = torch.nn.functional.leaky_relu
    elif args['use_group_norm']:
        norm = 'group_norm'; act = torch.nn.functional.leaky_relu
    elif args['use_ada_instance_norm'] or args['use_ada_instance_norm_mixup']:
        norm = torch.nn.InstanceNorm3d if args['model'] == 'DenseElNet' else torch.nn.InstanceNorm2d
        act  = torch.nn.functional.leaky_relu
    else:
        norm = torch.nn.BatchNorm3d if args['model'] == 'DenseElNet' else torch.nn.BatchNorm2d
        act  = torch.nn.functional.leaky_relu

    net = model_dict[args['model']](args, norm=norm, act_func=act)
    if args['use_GPU'] and torch.cuda.is_available():
        net = net.cuda()

    # Load weights
    assert args['path_model'] and os.path.isfile(args['path_model']), f"Missing model file: {args['path_model']}"
    ckpt = torch.load(args['path_model'], map_location="cpu")
    state_dict_single = move_to_single(ckpt['state_dict'])
    net.load_state_dict(state_dict_single, strict=False)
    print(f"Loaded weights: {args['path_model']}  |  #params={get_nparams(net)}")
    return net

def build_test_loader(args, paths):
    if args['use_pkl_for_dataload']:
        if paths.get("pkl_override"):
            pkl_path = paths["pkl_override"]
        else:
            pkl_path = os.path.join(paths["repo_root"], "cur_objs", args["mode"], f"cond_{args['cur_obj']}.pkl")
        assert os.path.isfile(pkl_path), f"Pickle not found: {pkl_path}"
        train_obj, valid_obj, test_obj = pickle.load(open(pkl_path, "rb"))
        # NO 100k subselect here
    else:
        # Build from MasterKey selections (you can customize as needed)
        ds_sel_path = os.path.join(paths["repo_root"], "cur_objs", "dataset_selections.pkl")
        DS_sel = pickle.load(open(ds_sel_path, "rb"))
        AllDS = CurLib.readArchives(args['path2MasterKey'])
        sel = 'S' if args['cur_obj'] == 'OpenEDS_S' else args['cur_obj']

        AllDS_cond = CurLib.selSubset(AllDS, DS_sel['test'][sel])
        dataDiv_obj = CurLib.generate_fileList(AllDS_cond, mode='none', notest=False)
        test_obj = DataLoader_riteyes(dataDiv_obj, args['path_data'], 'test', False, (480, 640),
                                      sort='nothing', scale=False, num_frames=args['frames'], args=args)

    # Standard flags used by forward()
    test_obj.augFlag = False
    test_obj.equi_var = args['equi_var']
    test_obj.path2data = paths['path_data']
    test_obj.scale = False

    test_loader = DataLoader(test_obj,
                             shuffle=False,
                             num_workers=args['workers'],
                             drop_last=True,
                             batch_size=args['batch_size'],
                             sampler=None)
    # quick size print
    num_seq, F = test_obj.imList.shape[0], test_obj.imList.shape[1]
    print(f"[TEST SET] sequences={num_seq}, frames/seq={F}, images={num_seq*F}")
    return test_loader

def main():
    args  = cfg.get_args()
    paths = cfg.get_paths()
    ensure_dirs(paths)

    os.environ["WANDB_SILENT"] = "true"
    try:
        # disabled mode: wandb.init() is required so wandb.log() doesn't error,
        # but it won't create a real run or write online.
        wandb.init(project="offline_eval",
                   name=args['exp_name'],
                   mode="disabled",
                   dir=paths["results"])
    except Exception as e:
        print(f"[wandb] init in disabled mode failed (will proceed anyway): {e}")



    # Save a copy of the args/paths used
    with open(os.path.join(paths["results"], "config_used.json"), "w") as f:
        json.dump({"args": args, "paths": paths}, f, indent=2)

    # Logger
    logger = make_logger(os.path.join(paths['logs'], 'test_log.log'),
                         rank=args['local_rank'] if args['do_distributed'] else 0)
    logger.write("=== TEST ONLY RUN ===")

    # Net
    net = build_net(args)
    net.eval()

    # Test loader
    test_loader = build_test_loader(args, paths)

    # A throwaway optimizer object to satisfy forward() calls that zero_grad()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.0)

    # Run forward in test mode over the entire loader
    batches_per_ep = len(test_loader)
    results = forward(net,
                      spiker=[],
                      logger=logger,
                      loader=test_loader,
                      optimizer=optimizer,
                      args=args,
                      path_dict={"repo_root": paths["repo_root"],
                                 "path_data": paths["path_data"],
                                 "exp": paths["exp"],
                                 "results": paths["results"],
                                 "logs": paths["logs"],
                                 "figures": paths["figures"]},
                      writer=[],
                      rank_cond=False,
                      optimizer_disc=False,
                      batches_per_ep=batches_per_ep,
                      last_epoch_valid=True,
                      csv_save_dir=paths["exp"])

    # Save the aggregated results as well
    with open(os.path.join(paths["results"], "test_results_summary.pkl"), "wb") as f:
        pickle.dump(results, f)
    with open(os.path.join(paths["results"], "test_results_summary.json"), "w") as f:
        json.dump(results, f, indent=2, default=lambda x: float(x) if hasattr(x, "__float__") else str(x))

    print("\n✅ Done. Outputs:")
    print("  - test_results.h5 (if SAVE_TEST_MAPS=True) in", paths["results"])
    print("  - test_results_summary.pkl / .json in", paths["results"])
    print("  - logs in", paths["logs"])

if __name__ == "__main__":
    main()
