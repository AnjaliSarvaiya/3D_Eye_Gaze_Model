
# config.py
import os

# ---- Disable W&B globally (safe if wandb is installed anywhere) ----
os.environ.setdefault("WANDB_DISABLED", "true")

# ---- Repo root (assumed to be this file's folder) ----
REPO_ROOT = os.path.abspath(os.path.dirname(__file__))

# ---- Your absolute data locations (edit if needed) ----
PATH_DATA      = "/media/ml/03C90A2604F5F8BA/Anjali/Model-aware_3D_Eye_Gaze-main/Datasets/All"
PATH_MASTERKEY = "/media/ml/03C90A2604F5F8BA/Anjali/Model-aware_3D_Eye_Gaze-main/Datasets/MasterKey"

# ---- Experiment tree + name ----
PATH_EXP_TREE = "/media/ml/03C90A2604F5F8BA/Anjali/Model-aware_3D_Eye_Gaze-main/Results"
EXP_NAME = "pretrained_sem"  # <-- you asked to keep this

# Derived folders
EXP_ROOT    = os.path.join(PATH_EXP_TREE, EXP_NAME)
RESULTS_DIR = os.path.join(EXP_ROOT, "results")
LOGS_DIR    = os.path.join(EXP_ROOT, "logs")
FIGS_DIR    = os.path.join(EXP_ROOT, "figures")

# ---- Optional: point directly to your split PKL (recommended) ----
# If None, code falls back to <REPO_ROOT>/cur_objs/<MODE>/cond_<CUR_OBJ>.pkl
PKL_PATH_OVERRIDE = os.path.join(REPO_ROOT, "cur_objs", "one_vs_one", "cond_TEyeD.pkl")

# ---- Defaults taken from args_maker.py (with your typical choices) ----
# Experiment / run control
MODE                  = "one_vs_one"
CUR_OBJ               = "TEyeD"
USE_PKL_FOR_DATALOAD  = True
PRODUCE_REND_MASK_PER_ITER = 2000
PERFORM_VALID         = 25

# Hyperparams
LR           = 8e-3
WD           = 2e-2
SEED         = 108
BATCH_SIZE   = 1
LOCAL_RANK   = 0
LR_DECAY     = 0
DROPOUT      = 0.0

# Model architecture
BASE_CHANNEL_SIZE     = 32
GROWTH_RATE           = 1.2
TRACK_RUNNING_STATS   = 0
EXTRA_DEPTH           = 0
GRAD_REV              = 0
ADV_DG                = 0
EQUI_VAR              = 0
NUM_BLOCKS            = 4
USE_FRN_TLU           = 0
USE_INSTANCE_NORM     = 0
USE_GROUP_NORM        = 0
USE_ADA_INSTANCE_NORM = 0
USE_ADA_INSTANCE_NORM_MIXUP = 0

# Discriminator
DISC_BASE_CHANNEL_SIZE = 32

# Paths (model + repo)
PATH_MODEL  = "/media/ml/03C90A2604F5F8BA/Anjali/Model-aware_3D_Eye_Gaze-main/last.pt"              # used mostly for test/valid-only loads
REPO_ROOT_OVERRIDE = REPO_ROOT

REDUCE_VALID_SAMPLES = 0
SAVE_EVERY           = 1

# Train/test toggles
EPOCHS          = 2
AUG_FLAG        = 1
ONE_BY_ONE_DS   = 0
EARLY_STOP      = 20
MIXED_PRECISION = 0
BATCHES_PER_EP  = 25
USE_GPU         = 1
REMOVE_SPIKES   = 0
PSEUDO_LABELS   = 0
FRAMES          = 4

# Extra model flags
USE_SCSE                 = 0
MAKE_ALEATORIC           = 0
SCALE_FACTOR             = 0.0
MAKE_UNCERTAIN           = 0
CONTINUE_TRAINING_PATH   = ""   # set to a .pt to resume optimizer/epoch too
REGRESSION_FROM_LATENT   = 1
CURR_LEARN_LOSSES        = 1
REGRESS_CHANNEL_GROW     = 0.0
MAXPOOL_IN_REGRESS_MOD   = -1
DILATION_IN_REGRESS_MOD  = 1
GROUPS                   = 1

# Heads / model selection
NET_SIMPLY_HEAD        = False
NET_SIMPLY_HEAD_TANH   = 1
NET_ELLSEG_HEAD        = False
NET_REND_HEAD          = True      # your usual setting for 3D head
MODEL_NAME             = "res_50_3"  # your model

# Losses / training knobs
TRAIN_DATA_PERCENTAGE = 1.0
LOSS_W_SUPERVISE = 1.0
LOSS_W_SUPERVISE_EYEBALL_CENTER = 0.15
LOSS_W_SUPERVISE_PUPIL_CENTER   = 0.0
LOSS_W_SUPERVISE_GV3D_L2        = 2.5
LOSS_W_SUPERVISE_GV3D_COS       = 2.5
LOSS_W_SUPERVISE_GV_UV          = 0.0

LOSS_W_ELLSEG          = 0.0
LOSS_REND_VECTORIZED   = True
TEMP_N_ANGLES          = 100
TEMP_N_RADIUS          = 50
LOSS_W_REND_GT_2_PRED  = 0.15
LOSS_W_REND_PRED_2_GT  = 0.15
LOSS_W_REND_PRED_2_GT_EDGE = 0.0  # you often enable this
LOSS_W_REND_DIAMETER   = 0.0
RANDOM_DATALOADER      = True

SCALE_BOUND_EYE        = "version_0"

# Pretrained / resume / test flags
WEIGHTS_PATH      = None       # if set, loads weights (like --weights_path)
PRETRAINED        = 0
PRETRAINED_RESNET = False
OPTIMIZER_TYPE    = "LAMB"
ONLY_TEST         = 0
ONLY_VALID        = 0
ONLY_TRAIN        = 0

# System
WORKERS                 = 8
NUM_BATCHES_TO_PLOT     = 10
DETECT_ANOMALY          = 0
GRAD_CLIP_NORM          = 0.0
NUM_SAMPLES_FOR_EMBEDD  = 200
DO_DISTRIBUTED          = 0
WORLD_SIZE              = 1      # not in args_maker, but used by samplers
DRY_RUN                 = False
SAVE_TEST_MAPS          = False

EARLY_STOP_METRIC       = "3D"   # "3D", "2D", or IoU-based in your code
SAVE_RESULTS_HERE       = ""     # for test-only override

# ----------------------------------------------------------------------
# Public API used by train.py / test_only.py
# ----------------------------------------------------------------------
def get_args():
    """Return a dict matching args_maker.py names (and train/test code usage)."""
    return {
        # Experiment basics
        "exp_name": EXP_NAME,
        "use_pkl_for_dataload": USE_PKL_FOR_DATALOAD,
        "produce_rend_mask_per_iter": PRODUCE_REND_MASK_PER_ITER,
        "perform_valid": PERFORM_VALID,

        # Hyperparams
        "lr": LR,
        "wd": WD,
        "seed": SEED,
        "batch_size": BATCH_SIZE,
        "local_rank": LOCAL_RANK,
        "lr_decay": LR_DECAY,
        "dropout": DROPOUT,

        # Model specifics
        "base_channel_size": BASE_CHANNEL_SIZE,
        "growth_rate": GROWTH_RATE,
        "track_running_stats": TRACK_RUNNING_STATS,
        "extra_depth": EXTRA_DEPTH,
        "grad_rev": GRAD_REV,
        "adv_DG": ADV_DG,
        "equi_var": EQUI_VAR,
        "num_blocks": NUM_BLOCKS,
        "use_frn_tlu": USE_FRN_TLU,
        "use_instance_norm": USE_INSTANCE_NORM,
        "use_group_norm": USE_GROUP_NORM,
        "use_ada_instance_norm": USE_ADA_INSTANCE_NORM,
        "use_ada_instance_norm_mixup": USE_ADA_INSTANCE_NORM_MIXUP,

        # Disc
        "disc_base_channel_size": DISC_BASE_CHANNEL_SIZE,

        # Paths & repo
        "path_exp_tree": PATH_EXP_TREE,
        "path_data": PATH_DATA,
        "path2MasterKey": PATH_MASTERKEY,
        "path_model": PATH_MODEL,
        "repo_root": REPO_ROOT_OVERRIDE,
        "reduce_valid_samples": REDUCE_VALID_SAMPLES,
        "save_every": SAVE_EVERY,

        # Train/test control
        "mode": MODE,
        "epochs": EPOCHS,
        "cur_obj": CUR_OBJ,
        "aug_flag": AUG_FLAG,
        "one_by_one_ds": ONE_BY_ONE_DS,
        "early_stop": EARLY_STOP,
        "mixed_precision": MIXED_PRECISION,
        "batches_per_ep": BATCHES_PER_EP,
        "use_GPU": USE_GPU,
        "remove_spikes": REMOVE_SPIKES,
        "pseudo_labels": PSEUDO_LABELS,
        "frames": FRAMES,

        # Extra model flags
        "use_scSE": USE_SCSE,
        "make_aleatoric": MAKE_ALEATORIC,
        "scale_factor": SCALE_FACTOR,
        "make_uncertain": MAKE_UNCERTAIN,
        "continue_training": CONTINUE_TRAINING_PATH,
        "regression_from_latent": REGRESSION_FROM_LATENT,
        "curr_learn_losses": CURR_LEARN_LOSSES,
        "regress_channel_grow": REGRESS_CHANNEL_GROW,
        "maxpool_in_regress_mod": MAXPOOL_IN_REGRESS_MOD,
        "dilation_in_regress_mod": DILATION_IN_REGRESS_MOD,
        "groups": GROUPS,

        # Heads & model selection
        "net_simply_head": NET_SIMPLY_HEAD,
        "net_simply_head_tanh": NET_SIMPLY_HEAD_TANH,
        "net_ellseg_head": NET_ELLSEG_HEAD,
        "net_rend_head": NET_REND_HEAD,
        "model": MODEL_NAME,

        # Loss weights
        "train_data_percentage": TRAIN_DATA_PERCENTAGE,
        "loss_w_supervise": LOSS_W_SUPERVISE,
        "loss_w_supervise_eyeball_center": LOSS_W_SUPERVISE_EYEBALL_CENTER,
        "loss_w_supervise_pupil_center": LOSS_W_SUPERVISE_PUPIL_CENTER,
        "loss_w_supervise_gaze_vector_3D_L2": LOSS_W_SUPERVISE_GV3D_L2,
        "loss_w_supervise_gaze_vector_3D_cos_sim": LOSS_W_SUPERVISE_GV3D_COS,
        "loss_w_supervise_gaze_vector_UV": LOSS_W_SUPERVISE_GV_UV,
        "loss_w_ellseg": LOSS_W_ELLSEG,
        "loss_rend_vectorized": LOSS_REND_VECTORIZED,
        "temp_n_angles": TEMP_N_ANGLES,
        "temp_n_radius": TEMP_N_RADIUS,
        "loss_w_rend_gt_2_pred": LOSS_W_REND_GT_2_PRED,
        "loss_w_rend_pred_2_gt": LOSS_W_REND_PRED_2_GT,
        "loss_w_rend_pred_2_gt_edge": LOSS_W_REND_PRED_2_GT_EDGE,
        "loss_w_rend_diameter": LOSS_W_REND_DIAMETER,
        "random_dataloader": RANDOM_DATALOADER,

        "scale_bound_eye": SCALE_BOUND_EYE,

        # Pretrained / resume / test
        "weights_path": WEIGHTS_PATH,
        "pretrained": PRETRAINED,
        "pretrained_resnet": PRETRAINED_RESNET,
        "optimizer_type": OPTIMIZER_TYPE,
        "only_test": ONLY_TEST,
        "only_valid": ONLY_VALID,
        "only_train": ONLY_TRAIN,

        # System
        "workers": WORKERS,
        "num_batches_to_plot": NUM_BATCHES_TO_PLOT,
        "detect_anomaly": DETECT_ANOMALY,
        "grad_clip_norm": GRAD_CLIP_NORM,
        "num_samples_for_embedding": NUM_SAMPLES_FOR_EMBEDD,
        "do_distributed": DO_DISTRIBUTED,
        "world_size": WORLD_SIZE,  # train.py needs this for samplers
        "dry_run": DRY_RUN,
        "save_test_maps": SAVE_TEST_MAPS,

        # Early stop / outputs
        "early_stop_metric": EARLY_STOP_METRIC,
        "save_results_here": SAVE_RESULTS_HERE,
    }

def get_paths():
    """Return the path dict the training / testing code expects."""
    return {
        "repo_root": REPO_ROOT,
        "path_data": PATH_DATA,
        "exp": EXP_ROOT,
        "results": RESULTS_DIR,
        "logs": LOGS_DIR,
        "figures": FIGS_DIR,
        "pkl_override": PKL_PATH_OVERRIDE,  # None or absolute path to cond_*.pkl
    }
