
import os
# --- make sure wandb is fully disabled before any import that might use it ---
os.environ.setdefault("WANDB_DISABLED", "true")

import sys
import json
import time
import random
import string
import logging
import pickle
import warnings
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ===== OPTIONAL: soften wandb calls so wandb.watch/log don't crash ============
try:
    import wandb  # used by scripts.py and by the training loops below
    # convert common calls to no-ops (keeps wandb.Image available if needed)
    wandb.init = lambda *a, **k: None
    wandb.log = lambda *a, **k: None
    wandb.watch = lambda *a, **k: None
    wandb.finish = lambda *a, **k: None
except Exception:
    class _WB:
        def __getattr__(self, _): return lambda *a, **k: None
    wandb = _WB()
# ==============================================================================

# repo imports
from timm.optim import Lamb as Lamb_timm
from timm.scheduler import CosineLRScheduler as CosineLRScheduler_timm
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from tensorboardX import SummaryWriter

# keep your original module paths
from SRC.models_mux import model_dict
from scripts import forward
import helperfunctions.CurriculumLib as CurLib
from helperfunctions.CurriculumLib import DataLoader_riteyes
from helperfunctions.helperfunctions import mod_scalar
from helperfunctions.utils import EarlyStopping, make_logger
from helperfunctions.utils import SpikeDetection, get_nparams
from helperfunctions.utils import move_to_single, FRN_TLU, do_nothing

# read args/paths from config.py
import config_train as cfg

warnings.filterwarnings("ignore")


# -----------------------------------------------------------------------------#
# Helpers to create a unique experiment folder tree inside Results/
# -----------------------------------------------------------------------------#
def _unique_name(base: str) -> str:
    now = datetime.now().strftime("%d_%m_%y_%H_%M_%S")
    rnd = "".join(random.choice(string.ascii_letters) for _ in range(5))
    return f"{base}_{rnd}_{now}"

def create_experiment_folder_tree(paths_from_cfg: dict, args_from_cfg: dict):
    """
    Build an experiment directory like:
      Results/<EXP_NAME>_<RAND>_<STAMP>/{results,figures,logs,src}
    unless config sets USE_UNIQUE_EXP_DIR=False, in which case:
      Results/<EXP_NAME>/{results,figures,logs,src}
    """
    path_exp_tree = paths_from_cfg["exp"] if "exp" in paths_from_cfg else cfg.PATH_EXP_TREE
    exp_name = args_from_cfg.get("exp_name", "EXP")

    use_unique = getattr(cfg, "USE_UNIQUE_EXP_DIR", True)
    leaf = _unique_name(exp_name) if use_unique else exp_name

    exp_root = os.path.join(path_exp_tree, leaf)
    results = os.path.join(exp_root, "results")
    logs    = os.path.join(exp_root, "logs")
    figs    = os.path.join(exp_root, "figures")
    src_dir = os.path.join(exp_root, "src")

    for d in (exp_root, results, logs, figs, src_dir):
        os.makedirs(d, exist_ok=True)

    out_paths = dict(paths_from_cfg)  # copy
    out_paths.update({
        "exp": exp_root,
        "results": results,
        "logs": logs,
        "figures": figs,
    })
    return out_paths, leaf


# -----------------------------------------------------------------------------#
# Core training code( reads args/paths from config)
# -----------------------------------------------------------------------------#
def train(args, path_dict, validation_mode=False, test_mode=False):

    rank_cond = (args['local_rank'] == 0) or not args['do_distributed']
    rank_cond_early_stop = rank_cond
    # deactivate tensorboard log
    rank_cond = False

    net_dict = []

    # %% Load model
    if args['use_frn_tlu']:
        net = model_dict[args['model']](args, norm=FRN_TLU, act_func=do_nothing)
    elif args['use_instance_norm']:
        norm = nn.InstanceNorm3d if args['model'] == 'DenseElNet' else nn.InstanceNorm2d
        net = model_dict[args['model']](args, norm=norm, act_func=F.leaky_relu)
    elif args['use_group_norm']:
        norm = 'group_norm'
        net = model_dict[args['model']](args, norm=norm, act_func=F.leaky_relu)
    elif args['use_ada_instance_norm']:
        norm = nn.InstanceNorm3d if args['model'] == 'DenseElNet' else nn.InstanceNorm2d
        net = model_dict[args['model']](args, norm=norm, act_func=F.leaky_relu)
    elif args['use_ada_instance_norm_mixup']:
        norm = nn.InstanceNorm3d if args['model'] == 'DenseElNet' else nn.InstanceNorm2d
        net = model_dict[args['model']](args, norm=norm, act_func=F.leaky_relu)
    else:
        norm = nn.BatchNorm3d if args['model'] == 'DenseElNet' else nn.BatchNorm2d
        net = model_dict[args['model']](args, norm=norm, act_func=F.leaky_relu)

    # %% Weight loaders
    if args['pretrained'] or args['continue_training'] or args['weights_path']:

        if args['weights_path']:
            path_pretrained = args['weights_path']
        elif args['pretrained']:
            path_pretrained = os.path.join(path_dict['repo_root'], '..', 'pretrained', 'pretrained.git_ok')
        elif args['continue_training']:
            path_pretrained = os.path.join(args['continue_training'])

        net_dict = torch.load(path_pretrained, map_location=torch.device('cpu'))
        state_dict_single = move_to_single(net_dict['state_dict'])
        net.load_state_dict(state_dict_single, strict=False)
        print(f'Pretrained model loaded from: {path_pretrained}')

    if test_mode or validation_mode:
        print(('Test' if test_mode else 'Validation') + ' mode detected. Loading model.')
        if args['path_model']:
            net_dict = torch.load(args['path_model'], map_location=torch.device('cpu'))
        else:
            net_dict = torch.load(os.path.join(path_dict['results'], 'last.pt'), map_location=torch.device('cpu'))

        state_dict_single = move_to_single(net_dict['state_dict'])
        net.load_state_dict(state_dict_single, strict=False)
        writer = []
    else:
        writer = SummaryWriter(path_dict['logs']) if rank_cond else []

    if args['use_GPU']:
        net.cuda()

    # %% DDP
    if args['do_distributed']:
        net = DDP(net, device_ids=[args['local_rank']], find_unused_parameters=True)

    # %% Logger
    logger = make_logger(os.path.join(path_dict['logs'], 'train_log.log'),
                         rank=args['local_rank'] if args['do_distributed'] else 0)
    logger.write_summary(str(net.parameters))
    logger.write('# of parameters: {}'.format(get_nparams(net)))

    if not (test_mode or validation_mode):
        logger.write('Training!')
        # wandb.watch(net)  # <- harmless no-op due to stub
    elif validation_mode:
        logger.write('Validating!')
    else:
        logger.write('Testing!')

    # %% Go
    train_validation_loops(net, net_dict, logger, args, path_dict, writer,
                           rank_cond, rank_cond_early_stop, validation_mode, test_mode)

    if writer:
        writer.close()


def train_validation_loops(net, net_dict, logger, args, path_dict, writer,
                           rank_cond, rank_cond_early_stop, validation_mode, test_mode):

    # Load from PKL if requested
    if args['use_pkl_for_dataload']:
        # allow override from config if provided
        pkl_override = path_dict.get("pkl_override") or getattr(cfg, "PKL_PATH_OVERRIDE", None)
        if pkl_override:
            path_cur_obj = pkl_override
        else:
            path_cur_obj = os.path.join(path_dict['repo_root'], 'cur_objs', args['mode'], 'cond_'+args['cur_obj']+'.pkl')
        with open(path_cur_obj, 'rb') as f:
            train_obj, valid_obj, test_obj = pickle.load(f)
    else:
        path_cur_obj = os.path.join(path_dict['repo_root'], 'cur_objs', 'dataset_selections.pkl')
        path2h5 = args['path_data']
        DS_sel = pickle.load(open(path_cur_obj, 'rb'))
        AllDS = CurLib.readArchives(args['path2MasterKey'])

        sel = 'S' if (args['cur_obj'] == 'OpenEDS_S') else args['cur_obj']

        if args['cur_obj'] == 'Ours':
            AllDS_cond = CurLib.selSubset(AllDS, DS_sel['train'][sel])
            dataDiv_obj = CurLib.generate_fileList(AllDS_cond, mode='none', notest=False)
            train_obj = DataLoader_riteyes(dataDiv_obj, path2h5, 'train', True, (480, 640),
                                           scale=False, num_frames=args['frames'], args=args)
            valid_obj = DataLoader_riteyes(dataDiv_obj, path2h5, 'valid', False, (480, 640),
                                           sort='nothing', scale=False, num_frames=args['frames'], args=args)
            AllDS_cond = CurLib.selSubset(AllDS, DS_sel['test'][sel])
            dataDiv_obj = CurLib.generate_fileList(AllDS_cond, mode='none', notest=False)
            test_obj = DataLoader_riteyes(dataDiv_obj, path2h5, 'test', False, (480, 640),
                                          sort='nothing', scale=False, num_frames=args['frames'], args=args)
        else:
            AllDS_cond = CurLib.selSubset(AllDS, DS_sel['train'][sel])
            dataDiv_obj = CurLib.generate_fileList(AllDS_cond, mode='vanilla', notest=False)
            train_obj = DataLoader_riteyes(dataDiv_obj, path2h5, 'train', True, (480, 640),
                                           scale=False, num_frames=args['frames'], args=args)
            valid_obj = DataLoader_riteyes(dataDiv_obj, path2h5, 'valid', False, (480, 640),
                                           sort='nothing', scale=False, num_frames=args['frames'], args=args)
            AllDS_cond = CurLib.selSubset(AllDS, DS_sel['test'][sel])
            dataDiv_obj = CurLib.generate_fileList(AllDS_cond, mode='none', notest=False)
            test_obj = DataLoader_riteyes(dataDiv_obj, path2h5, 'test', False, (480, 640),
                                          sort='nothing', scale=False, num_frames=args['frames'], args=args)

            # cleanup corrupt entries (same as your code)
            def clean_corrupt_entries(dataset, name="dataset"):
                valid_indices = []
                for i, entry in enumerate(dataset.imList[:, 0, 0]):
                    try:
                        _ = dataset.__getitem__(i)
                        valid_indices.append(i)
                    except Exception:
                        print(f"[FIX] Removing corrupt sample from {name}: {entry}")
                dataset.imList = dataset.imList[valid_indices]
                print(f"[CLEANUP] {name} reduced to {len(valid_indices)} valid sequences")

            clean_corrupt_entries(train_obj, "train")
            clean_corrupt_entries(valid_obj, "valid")
            clean_corrupt_entries(test_obj, "test")

    # diagnostics + subselect
    print('Starting the procedure of removing unwanted train-val video overlap...')
    train_vid_ids = list(np.unique(train_obj.imList[:, :, 1]))
    print('Sub-selecting first 100k test frames')
    test_cutoff = int(100000 / test_obj.imList.shape[1])
    test_obj.imList = test_obj.imList[:test_cutoff]

    print('\nNumber of images:')
    print(f'Train images left: {train_obj.imList.shape[0]*train_obj.imList.shape[1]}')
    print(f'Valid  images left: {valid_obj.imList.shape[0]*valid_obj.imList.shape[1]}')
    print(f'Test   images left: {test_obj.imList.shape[0]*test_obj.imList.shape[1]}\n')

    # flags
    train_obj.augFlag = args['aug_flag']
    valid_obj.augFlag = False
    test_obj.augFlag = False

    train_obj.equi_var = args['equi_var']
    valid_obj.equi_var = args['equi_var']
    test_obj.equi_var = args['equi_var']

    train_obj.path2data = path_dict['path_data']
    valid_obj.path2data = path_dict['path_data']
    test_obj.path2data  = path_dict['path_data']

    train_obj.scale = False
    valid_obj.scale = False
    test_obj.scale  = False

    # samplers
    train_sampler = DistributedSampler(train_obj, rank=args['local_rank'], shuffle=False,
                                       num_replicas=args['world_size'])
    valid_sampler = DistributedSampler(valid_obj, rank=args['local_rank'], shuffle=False,
                                       num_replicas=args['world_size'])
    test_sampler  = DistributedSampler(test_obj,  rank=args['local_rank'], shuffle=False,
                                       num_replicas=args['world_size'])

    # loaders
    logger.write('Initializing loaders')
    if validation_mode:
        valid_loader = DataLoader(valid_obj, shuffle=False, num_workers=args['workers'],
                                  drop_last=True, pin_memory=True, batch_size=args['batch_size'],
                                  sampler=valid_sampler if args['do_distributed'] else None)
    elif test_mode:
        test_loader = DataLoader(test_obj, shuffle=False, num_workers=0, drop_last=True,
                                 batch_size=args['batch_size'],
                                 sampler=test_sampler if args['do_distributed'] else None)
    else:
        train_loader = DataLoader(train_obj, shuffle=args['random_dataloader'], num_workers=args['workers'],
                                  drop_last=True, pin_memory=True, batch_size=args['batch_size'],
                                  sampler=train_sampler if args['do_distributed'] else None)
        valid_loader = DataLoader(valid_obj, shuffle=False, num_workers=args['workers'],
                                  drop_last=True, pin_memory=True, batch_size=args['batch_size'],
                                  sampler=valid_sampler if args['do_distributed'] else None)

    # early stopping
    if '3D' in args['early_stop_metric'] or '2D' in args['early_stop_metric']:
        early_stop = EarlyStopping(metric=args['early_stop_metric'], patience=args['early_stop'],
                                   verbose=True, delta=0.001, rank_cond=rank_cond_early_stop,
                                   mode='min', fName='best_model.pt', path_save=path_dict['results'])
    else:
        early_stop = EarlyStopping(metric=args['early_stop_metric'], patience=args['early_stop'],
                                   verbose=True, delta=0.001, rank_cond=rank_cond_early_stop,
                                   mode='max', fName='best_model.pt', path_save=path_dict['results'])

    # scalars
    if args['curr_learn_losses']:
        alpha_scalar = mod_scalar([0, args['epochs']], [0, 1])
        beta_scalar  = mod_scalar([10, 20], [0, 1])

    # optimizer
    param_list = [p for n, p in net.named_parameters() if 'adv' not in n]
    use_sched = False
    if 'LAMB' in args['optimizer_type']:
        optimizer = Lamb_timm(param_list, lr=args['lr'], weight_decay=args['wd'])
        scheduler = CosineLRScheduler_timm(optimizer, t_initial=args['epochs'],
                                           lr_min=args['lr']/(10.0**2), warmup_t=4,
                                           warmup_lr_init=args['lr']/(10.0**2))
        use_sched = True
    elif 'adamw_cos' in args['optimizer_type']:
        optimizer = torch.optim.AdamW(param_list, lr=args['lr'], betas=(0.9, 0.99), weight_decay=args['wd'])
        scheduler = CosineLRScheduler_timm(optimizer, t_initial=args['epochs'],
                                           lr_min=args['lr']/(10.0**2), warmup_t=4,
                                           warmup_lr_init=args['lr']/(10.0**2))
        use_sched = True
    elif 'adamw_step' in args['optimizer_type']:
        optimizer = torch.optim.AdamW(param_list, lr=args['lr'], betas=(0.9, 0.99), weight_decay=args['wd'])
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)
        use_sched = True
    elif 'adam_cos' in args['optimizer_type']:
        optimizer = torch.optim.Adam(param_list, lr=args['lr'], betas=(0.9, 0.99), weight_decay=args['wd'])
        scheduler = CosineLRScheduler_timm(optimizer, t_initial=args['epochs'],
                                           lr_min=args['lr']/(10.0**3), warmup_t=4,
                                           warmup_lr_init=args['lr']/(10.0**2))
        use_sched = True
    elif 'adam_step' in args['optimizer_type']:
        optimizer = torch.optim.Adam(param_list, lr=args['lr'], betas=(0.9, 0.99), weight_decay=args['wd'])
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)
        use_sched = True
    else:
        optimizer = torch.optim.Adam(param_list, lr=args['lr'], amsgrad=False)

    if args['adv_DG']:
        adv_params = [p for n, p in net.named_parameters() if 'adv' in n]
        optimizer_disc = torch.optim.Adam(adv_params, lr=args['lr'], amsgrad=True)
    else:
        optimizer_disc = False

    # checkpoint dict
    checkpoint = {'args': args}
    args['time_to_update'] = True
    last_epoch_validation = False

    if test_mode:
        logging.info('Entering test mode only ...'); logger.write('Entering test mode only ...')
        args['alpha'] = 0.5; args['beta'] = 0.5
        max_batches = len(test_loader)
        batches_per_ep = min(args['batches_per_ep'], max_batches)

        test_result = forward(net, [], logger, test_loader, optimizer, args, path_dict,
                              writer=writer, rank_cond=rank_cond, epoch=0, mode='test',
                              batches_per_ep=batches_per_ep, last_epoch_valid=True,
                              csv_save_dir=path_dict['exp'])
        checkpoint['test_result'] = test_result

        os.makedirs(path_dict['results'], exist_ok=True)
        with open(os.path.join(path_dict['results'], 'test_results.pkl'), 'wb') as f:
            pickle.dump(checkpoint, f)
        return

    if validation_mode:
        logging.info('Entering validation mode only ...')
        args['alpha'] = 0.5; args['beta'] = 0.5
        valid_result = forward(net, [], logger, valid_loader, optimizer, args, path_dict,
                               writer=writer, rank_cond=rank_cond, epoch=0, mode='valid',
                               batches_per_ep=len(valid_loader), last_epoch_valid=True,
                               csv_save_dir=path_dict['exp'])
        checkpoint['valid_result'] = valid_result

        os.makedirs(path_dict['results'], exist_ok=True)
        with open(os.path.join(path_dict['results'], 'valid_results.pkl'), 'wb') as f:
            pickle.dump(checkpoint, f)
        return

    # train loop
    spiker = SpikeDetection() if args['remove_spikes'] else False
    logging.info('Entering train mode ...')

    if args['continue_training']:
        optimizer.load_state_dict(net_dict['optimizer'])
        epoch = net_dict['epoch'] + 1
    else:
        epoch = 0

    while epoch < args['epochs']:
        if args['time_to_update']:
            args['time_to_update'] = False
            if args['one_by_one_ds']:
                train_loader.dataset.sort('one_by_one_ds', args['batch_size'])
                valid_loader.dataset.sort('one_by_one_ds', args['batch_size'])
            else:
                train_loader.dataset.sort('ordered')
                valid_loader.dataset.sort('ordered')

        train_sampler.set_epoch(epoch)
        valid_sampler.set_epoch(epoch)

        logging.info('Starting epoch: %d' % epoch)

        if args['curr_learn_losses']:
            args['alpha'] = mod_scalar([0, args['epochs']], [0, 1]).get_scalar(epoch)
            args['beta']  = mod_scalar([10, 20], [0, 1]).get_scalar(epoch)
        else:
            args['alpha'] = 0.5
            args['beta']  = 0.5

        if args['dry_run']:
            train_batches_per_ep = len(train_loader)
            valid_batches_per_ep = len(valid_loader)
        else:
            train_batches_per_ep = args['batches_per_ep']
            valid_batches_per_ep = 10 if args['reduce_valid_samples'] else int(20000/(args['frames']*args['batch_size']))

        train_result = forward(net, spiker, logger, train_loader, optimizer, args, path_dict,
                               optimizer_disc=optimizer_disc, writer=writer, rank_cond=rank_cond,
                               epoch=epoch, mode='train', batches_per_ep=train_batches_per_ep)

        if epoch == args['epochs'] - 1:
            last_epoch_validation = True
            valid_batches_per_ep = len(valid_loader)

        valid_result = forward(net, spiker, logger, valid_loader, optimizer, args, path_dict,
                               writer=writer, rank_cond=rank_cond, epoch=epoch, mode='valid',
                               batches_per_ep=valid_batches_per_ep, last_epoch_valid=last_epoch_validation, csv_save_dir=path_dict['exp'])

        checkpoint['state_dict']   = move_to_single(net.state_dict())
        checkpoint['optimizer']    = optimizer.state_dict()
        checkpoint['epoch']        = epoch
        checkpoint['train_result'] = train_result
        checkpoint['valid_result'] = valid_result

        # save best
        if '3D' in args['early_stop_metric']:
            temp_score = checkpoint['valid_result']['gaze_3D_ang_deg_mean']
        elif '2D' in args['early_stop_metric']:
            temp_score = checkpoint['valid_result']['gaze_ang_deg_mean']
        else:
            temp_score = checkpoint['valid_result']['masked_rendering_iou_mean']
        early_stop(checkpoint)

        # also save rolling "last.pt"
        early_stop.save_checkpoint(temp_score, checkpoint, update_val_score=False, use_this_name_instead='last.pt')

        if 'use_sched' in locals() and use_sched:
            scheduler.step(epoch=epoch)
        epoch += 1


# -----------------------------------------------------------------------------#
# Entry point: get config, make exp folder, run train->valid->test
# -----------------------------------------------------------------------------#
def main():
    # args & paths from config.py
    args = cfg.get_args()
    paths = cfg.get_paths()

    # create unique experiment folder inside Results/
    paths, exp_leaf = create_experiment_folder_tree(paths, args)

    # (optional) dump the config used for this run
    os.makedirs(paths["results"], exist_ok=True)
    with open(os.path.join(paths["results"], "config_used.json"), "w") as f:
        json.dump({"args": args, "paths": paths}, f, indent=2)

    # DDP-ish setup
    if args['do_distributed']:
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
        world_size = torch.distributed.get_world_size()
    else:
        world_size = 1

    args['world_size'] = world_size
    args['batch_size'] = int(args['batch_size'] / world_size)

    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True

    torch.manual_seed(args['seed'])
    np.random.seed(args['seed'])
    random.seed(args['seed'])

    # Map minimal fields expected by training loops
    path_dict = {
        "repo_root": paths.get("repo_root", cfg.REPO_ROOT),
        "path_data": paths["path_data"],
        "results": paths["results"],
        "logs": paths["logs"],
        "figures": paths["figures"],
        "exp": paths["exp"],
        "pkl_override": paths.get("pkl_override", None),
    }

    # run
    if not args['only_test']:
        if not args['only_valid']:
            print('train mode')
            train(args, path_dict, validation_mode=False, test_mode=False)

            print('validation mode')
            train(args, path_dict, validation_mode=True, test_mode=False)

            print('test mode')
            train(args, path_dict, validation_mode=False, test_mode=True)

    if args['do_distributed']:
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()
    elif args['only_valid']:
        print('validation mode')
        train(args, path_dict, validation_mode=True, test_mode=False)
    elif args['only_test']:
        print('validation mode')
        train(args, path_dict, validation_mode=True, test_mode=False)
        print('test mode')
        train(args, path_dict, validation_mode=False, test_mode=True)


if __name__ == "__main__":
    main()
