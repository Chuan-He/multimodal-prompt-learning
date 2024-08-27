import argparse
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import torch

from dassl.utils import setup_logger, set_random_seed, collect_env_info
from dassl.config import get_cfg_default
from dassl.engine import build_trainer
from trainers.constants import get_dataset_specified_config
from trainers.constants_1 import get_dataset_specified_config_1
from trainers.constants_2 import get_dataset_specified_config_2

# custom
import datasets.oxford_pets
import datasets.oxford_flowers
import datasets.fgvc_aircraft
import datasets.dtd
import datasets.eurosat
import datasets.stanford_cars
import datasets.food101
import datasets.sun397
import datasets.caltech101
import datasets.ucf101
import datasets.imagenet

import datasets.imagenet_sketch
import datasets.imagenetv2
import datasets.imagenet_a
import datasets.imagenet_r

import trainers.coop
import trainers.cocoop
import trainers.zsclip
import trainers.maple
import trainers.independentVL
import trainers.vpt

import trainers.kgcoop
import trainers.kvpt
import trainers.kvpt_i
import trainers.kvpt_t
import trainers.kdpt
import trainers.kdpt_m
import trainers.kdpt_maple
import trainers.kdpt_maple_1_1
import trainers.kdpt_maple_2
import trainers.kdpt_maple_i
import trainers.kmvpt
import trainers.kmdpt
import trainers.kmvpt_i
import trainers.kmvpt_t

def print_args(args, cfg):
    print("***************")
    print("** Arguments **")
    print("***************")
    optkeys = list(args.__dict__.keys())
    optkeys.sort()
    for key in optkeys:
        print("{}: {}".format(key, args.__dict__[key]))
    print("************")
    print("** Config **")
    print("************")
    print(cfg)


def reset_cfg(cfg, args):
    if args.root:
        cfg.DATASET.ROOT = args.root

    if args.output_dir:
        cfg.OUTPUT_DIR = args.output_dir

    if args.resume:
        cfg.RESUME = args.resume

    if args.seed:
        cfg.SEED = args.seed

    if args.source_domains:
        cfg.DATASET.SOURCE_DOMAINS = args.source_domains

    if args.target_domains:
        cfg.DATASET.TARGET_DOMAINS = args.target_domains

    if args.transforms:
        cfg.INPUT.TRANSFORMS = args.transforms

    if args.trainer:
        cfg.TRAINER.NAME = args.trainer

    if args.backbone:
        cfg.MODEL.BACKBONE.NAME = args.backbone

    if args.head:
        cfg.MODEL.HEAD.NAME = args.head
    


def extend_cfg(cfg):
    """
    Add new config variables.

    E.g.
        from yacs.config import CfgNode as CN
        cfg.TRAINER.MY_MODEL = CN()
        cfg.TRAINER.MY_MODEL.PARAM_A = 1.
        cfg.TRAINER.MY_MODEL.PARAM_B = 0.5
        cfg.TRAINER.MY_MODEL.PARAM_C = False
    """
    from yacs.config import CfgNode as CN
    
    cfg.TRAINER.COOP = CN()
    cfg.TRAINER.COOP.N_CTX = 2  # number of context vectors
    cfg.TRAINER.COOP.CSC = False  # class-specific context
    cfg.TRAINER.COOP.CTX_INIT = "a photo of a"  # initialization words
    cfg.TRAINER.COOP.PREC = "fp16"  # fp16, fp32, amp
    cfg.TRAINER.COOP.CLASS_TOKEN_POSITION = "end"  # 'middle' or 'end' or 'front'

    cfg.TRAINER.COCOOP = CN()
    cfg.TRAINER.COCOOP.N_CTX = 2  # number of context vectors
    cfg.TRAINER.COCOOP.CTX_INIT = "a photo of a"  # initialization words
    cfg.TRAINER.COCOOP.PREC = "fp16"  # fp16, fp32, amp

    
    # Config for MaPLe
    cfg.TRAINER.MAPLE = CN()
    cfg.TRAINER.MAPLE.N_CTX = 1  # number of context vectors
    cfg.TRAINER.MAPLE.CTX_INIT = "a photo of a"  # initialization words
    cfg.TRAINER.MAPLE.PREC = "fp16"  # fp16, fp32, amp
    cfg.TRAINER.MAPLE.PROMPT_DEPTH = 9 # Max 12, minimum 0, for 1 it will act as shallow MaPLe (J=1)


    # Config for independent Vision Language prompting (independent-vlp)
    cfg.TRAINER.IVLP = CN()
    cfg.TRAINER.IVLP.N_CTX_VISION = 2  # number of context vectors at the vision branch
    cfg.TRAINER.IVLP.N_CTX_TEXT = 2  # number of context vectors at the language branch
    cfg.TRAINER.IVLP.CTX_INIT = "a photo of a"  # initialization words (only for language prompts)
    cfg.TRAINER.IVLP.PREC = "fp16"  # fp16, fp32, amp
    # If both variables below are set to 0, 0, will the config will degenerate to COOP model
    cfg.TRAINER.IVLP.PROMPT_DEPTH_VISION = 9 # Max 12, minimum 0, for 0 it will act as shallow MaPLe (J=1)
    cfg.TRAINER.IVLP.PROMPT_DEPTH_TEXT = 9  # Max 12, minimum 0, for 0 it will act as shallow MaPLe (J=1)


    # Config for only vision side prompting
    cfg.TRAINER.VPT = CN()
    cfg.TRAINER.VPT.N_CTX_VISION = 2  # number of context vectors at the vision branch
    cfg.TRAINER.VPT.CTX_INIT = "a photo of a"  # initialization words
    cfg.TRAINER.VPT.PREC = "fp16"  # fp16, fp32, amp
    cfg.TRAINER.VPT.PROMPT_DEPTH_VISION = 1  # if set to 1, will represent shallow vision prompting only


    cfg.TRAINER.KgCoOp = CN()
    cfg.TRAINER.KgCoOp.N_CTX = 1  # number of context vectors
    cfg.TRAINER.KgCoOp.CTX_INIT = "a photo of a"  # initialization words
    cfg.TRAINER.KgCoOp.CSC = False  # class-specific context
    cfg.TRAINER.KgCoOp.PREC = "fp16"  # fp16, fp32, amp
    cfg.TRAINER.KgCoOp.CLASS_TOKEN_POSITION = "end"  # 'middle' or 'end' or 'front'


    # Config for KPT
    cfg.TRAINER.KVPT = CN()
    cfg.TRAINER.KVPT.N_CTX = 2  # number of context vectors
    cfg.TRAINER.KVPT.CTX_INIT = "a photo of a"  # initialization words
    cfg.TRAINER.KVPT.PREC = "fp16"  # fp16, fp32, amp
    cfg.TRAINER.KVPT.W = 1.0
    cfg.TRAINER.KVPT.N = 4
    #cfg.DATASET.NUM_SHOTS = 16


    cfg.TRAINER.KVPT_I = CN()
    cfg.TRAINER.KVPT_I.N_CTX = 2  # number of context vectors
    cfg.TRAINER.KVPT_I.CTX_INIT = "a photo of a"  # initialization words
    cfg.TRAINER.KVPT_I.PREC = "fp16"  # fp16, fp32, amp
    cfg.TRAINER.KVPT_I.W = 1.0
    cfg.TRAINER.KVPT_I.N = 4
    #cfg.DATASET.NUM_SHOTS = 16


    cfg.TRAINER.KVPT_T = CN()
    cfg.TRAINER.KVPT_T.N_CTX = 2  # number of context vectors
    cfg.TRAINER.KVPT_T.CTX_INIT = "a photo of a"  # initialization words
    cfg.TRAINER.KVPT_T.PREC = "fp16"  # fp16, fp32, amp
    cfg.TRAINER.KVPT_T.W = 1.0
    cfg.TRAINER.KVPT_T.N = 4
    #cfg.DATASET.NUM_SHOTS = 16


    cfg.TRAINER.KDPT = CN()
    cfg.TRAINER.KDPT.N_CTX = 2  # number of context vectors
    cfg.TRAINER.KDPT.CTX_INIT = "a photo of a"  # initialization words
    cfg.TRAINER.KDPT.PREC = "fp16"  # fp16, fp32, amp
    cfg.TRAINER.KDPT.W = 1.0
    cfg.TRAINER.KDPT.N = 4
    #cfg.DATASET.NUM_SHOTS = 16


    cfg.TRAINER.KMVPT = CN()
    cfg.TRAINER.KMVPT.N_CTX = 2  # number of context vectors
    cfg.TRAINER.KMVPT.CTX_INIT = "a photo of a"  # initialization words
    cfg.TRAINER.KMVPT.PREC = "fp16"  # fp16, fp32, amp
    cfg.TRAINER.KMVPT.W = 1.0
    cfg.TRAINER.KMVPT.N = 4
    #cfg.DATASET.NUM_SHOTS = 16


    cfg.TRAINER.KMDPT = CN()
    cfg.TRAINER.KMDPT.N_CTX = 2  # number of context vectors
    cfg.TRAINER.KMDPT.CTX_INIT = "a photo of a"  # initialization words
    cfg.TRAINER.KMDPT.PREC = "fp16"  # fp16, fp32, amp
    cfg.TRAINER.KMDPT.W = 1.0
    cfg.TRAINER.KMDPT.N = 4
    #cfg.DATASET.NUM_SHOTS = 16


    cfg.TRAINER.KMVPT_T = CN()
    cfg.TRAINER.KMVPT_T.N_CTX = 2  # number of context vectors
    cfg.TRAINER.KMVPT_T.CTX_INIT = "a photo of a"  # initialization words
    cfg.TRAINER.KMVPT_T.PREC = "fp16"  # fp16, fp32, amp
    cfg.TRAINER.KMVPT_T.W = 1.0
    cfg.TRAINER.KMVPT_T.N = 4
    #cfg.DATASET.NUM_SHOTS = 16


    cfg.TRAINER.KMVPT_I = CN()
    cfg.TRAINER.KMVPT_I.N_CTX = 2  # number of context vectors
    cfg.TRAINER.KMVPT_I.CTX_INIT = "a photo of a"  # initialization words
    cfg.TRAINER.KMVPT_I.PREC = "fp16"  # fp16, fp32, amp
    cfg.TRAINER.KMVPT_I.W = 1.0
    cfg.TRAINER.KMVPT_I.N = 4
    #cfg.DATASET.NUM_SHOTS = 16


    cfg.TRAINER.KMDPTM = CN()
    cfg.TRAINER.KMDPTM.N_CTX = 2  # number of context vectors
    cfg.TRAINER.KMDPTM.CTX_INIT = "a photo of a"  # initialization words
    cfg.TRAINER.KMDPTM.PREC = "fp16"  # fp16, fp32, amp
    cfg.TRAINER.KMDPTM.W = 1.0
    cfg.TRAINER.KMDPTM.N = 4
    #cfg.DATASET.NUM_SHOTS = 16


    cfg.TRAINER.KDPT_M = CN()
    cfg.TRAINER.KDPT_M.N_CTX = 2  # number of context vectors
    cfg.TRAINER.KDPT_M.CTX_INIT = "a photo of a"  # initialization words
    cfg.TRAINER.KDPT_M.PREC = "fp16"  # fp16, fp32, amp
    cfg.TRAINER.KDPT_M.W = 1.0
    cfg.TRAINER.KDPT_M.N = 4
    #cfg.DATASET.NUM_SHOTS = 16


    cfg.TRAINER.KDPT_MAPLE = CN()
    cfg.TRAINER.KDPT_MAPLE.N_CTX = 2  # number of context vectors
    cfg.TRAINER.KDPT_MAPLE.CTX_INIT = "a photo of a"  # initialization words
    cfg.TRAINER.KDPT_MAPLE.PREC = "fp16"  # fp16, fp32, amp
    cfg.TRAINER.KDPT_MAPLE.N = 4
    cfg.TRAINER.KDPT_MAPLE.PROMPT_DEPTH = 9
    #cfg.DATASET.NUM_SHOTS = 16


    cfg.TRAINER.KDPT_MAPLE_1 = CN()
    cfg.TRAINER.KDPT_MAPLE_1.N_CTX = 2  # number of context vectors
    cfg.TRAINER.KDPT_MAPLE_1.CTX_INIT = "a photo of a"  # initialization words
    cfg.TRAINER.KDPT_MAPLE_1.PREC = "fp16"  # fp16, fp32, amp
    cfg.TRAINER.KDPT_MAPLE_1.W = 1.0
    cfg.TRAINER.KDPT_MAPLE_1.N = 4
    cfg.TRAINER.KDPT_MAPLE_1.PROMPT_DEPTH = 9
    #cfg.DATASET.NUM_SHOTS = 16


    cfg.TRAINER.KDPT_MAPLE_1_1 = CN()
    cfg.TRAINER.KDPT_MAPLE_1_1.N_CTX = 1  # number of context vectors
    cfg.TRAINER.KDPT_MAPLE_1_1.CTX_INIT = "a photo of a"  # initialization words
    cfg.TRAINER.KDPT_MAPLE_1_1.PREC = "fp16"  # fp16, fp32, amp
    cfg.TRAINER.KDPT_MAPLE_1_1.W = 1.0
    cfg.TRAINER.KDPT_MAPLE_1_1.PROMPT_DEPTH = 9
    cfg.TRAINER.KDPT_MAPLE_1_1.N = 4
    #cfg.DATASET.NUM_SHOTS = 16


    cfg.TRAINER.KDPT_MAPLE_2 = CN()
    cfg.TRAINER.KDPT_MAPLE_2.N_CTX = 2  # number of context vectors
    cfg.TRAINER.KDPT_MAPLE_2.CTX_INIT = "a photo of a"  # initialization words
    cfg.TRAINER.KDPT_MAPLE_2.PREC = "fp16"  # fp16, fp32, amp
    cfg.TRAINER.KDPT_MAPLE_2.N = 4
    cfg.TRAINER.KDPT_MAPLE_2.PROMPT_DEPTH = 9
 

    cfg.TRAINER.KDPT_MAPLE_I = CN()
    cfg.TRAINER.KDPT_MAPLE_I.N_CTX = 1  # number of context vectors
    cfg.TRAINER.KDPT_MAPLE_I.CTX_INIT = "a photo of a"  # initialization words
    cfg.TRAINER.KDPT_MAPLE_I.PREC = "fp16"  # fp16, fp32, amp
    cfg.TRAINER.KDPT_MAPLE_I.PROMPT_DEPTH = 9
    cfg.TRAINER.KDPT_MAPLE_I.N = 4
    #cfg.DATASET.NUM_SHOTS = 16
    cfg.TRAINER.W = 1.0
    cfg.TRAINER.W1 = 1.0
    cfg.TRAINER.W2 = 1.0
    cfg.TRAINER.W3 = 1.0
    cfg.DATASET.SUBSAMPLE_CLASSES = "all"  # all, base or new

def setup_cfg(args):
    cfg = get_cfg_default()
    extend_cfg(cfg)

    # 1. From the dataset config file
    if args.dataset_config_file:
        cfg.merge_from_file(args.dataset_config_file)

    # 2. From the method config file
    if args.config_file:
        cfg.merge_from_file(args.config_file)

    # 3. From input arguments
    reset_cfg(cfg, args)

    # 4. From optional input arguments
    cfg.merge_from_list(args.opts)

    # 5. Override dataset specific config
    cfg.merge_from_list(get_dataset_specified_config_2(cfg.DATASET.NAME))

    cfg.freeze()

    return cfg


def main(args):
    cfg = setup_cfg(args)
    if cfg.SEED >= 0:
        print("Setting fixed seed: {}".format(cfg.SEED))
        set_random_seed(cfg.SEED)
    setup_logger(cfg.OUTPUT_DIR)

    if torch.cuda.is_available() and cfg.USE_CUDA:
        torch.backends.cudnn.benchmark = True

    print_args(args, cfg)
    print("Collecting env info ...")
    print("** System info **\n{}\n".format(collect_env_info()))

    trainer = build_trainer(cfg)

    if args.eval_only:
        trainer.load_model(args.model_dir, epoch=args.load_epoch)
        trainer.test()
        return

    if not args.no_train:
        trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="", help="path to dataset")
    parser.add_argument("--output-dir", type=str, default="", help="output directory")
    parser.add_argument(
        "--resume",
        type=str,
        default="",
        help="checkpoint directory (from which the training resumes)",
    )
    parser.add_argument(
        "--seed", type=int, default=-1, help="only positive value enables a fixed seed"
    )
    parser.add_argument(
        "--source-domains", type=str, nargs="+", help="source domains for DA/DG"
    )
    parser.add_argument(
        "--target-domains", type=str, nargs="+", help="target domains for DA/DG"
    )
    parser.add_argument(
        "--transforms", type=str, nargs="+", help="data augmentation methods"
    )
    parser.add_argument(
        "--config-file", type=str, default="", help="path to config file"
    )
    parser.add_argument(
        "--dataset-config-file",
        type=str,
        default="",
        help="path to config file for dataset setup",
    )
    parser.add_argument("--trainer", type=str, default="", help="name of trainer")
    parser.add_argument("--backbone", type=str, default="", help="name of CNN backbone")
    parser.add_argument("--head", type=str, default="", help="name of head")
    parser.add_argument("--eval-only", action="store_true", help="evaluation only")
    parser.add_argument(
        "--model-dir",
        type=str,
        default="",
        help="load model from this directory for eval-only mode",
    )
    parser.add_argument(
        "--load-epoch", type=int, help="load model weights at this epoch for evaluation"
    )
    parser.add_argument(
        "--no-train", action="store_true", help="do not call trainer.train()"
    )
    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="modify config options using the command-line",
    )
    args = parser.parse_args()
    main(args)
