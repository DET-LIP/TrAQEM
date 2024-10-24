# ------------------------------------------------------------------------
# OV DETR
# Copyright (c) S-LAB, Nanyang Technological University. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

import argparse
import datetime
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

import datasets
import datasets.samplers as samplers
import util.misc as utils
from datasets import build_dataset, get_coco_api_from_dataset
from models import build_model


def get_args_parser():
    parser = argparse.ArgumentParser("OV DETR Detector", add_help=False)
    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--lr_backbone_names", default=["backbone.0"], type=str, nargs="+")
    parser.add_argument("--lr_backbone", default=2e-5, type=float)
    parser.add_argument(
        "--lr_linear_proj_names",
        default=["reference_points", "sampling_offsets"],
        type=str,
        nargs="+",
    )
    parser.add_argument("--lr_linear_proj_mult", default=0.1, type=float)
    parser.add_argument("--batch_size", default=2, type=int)
    parser.add_argument("--weight_decay", default=1e-4, type=float)
    parser.add_argument("--epochs", default=1500, type=int)
    parser.add_argument("--lr_drop", default=40, type=int)
    parser.add_argument("--lr_drop_epochs", default=None, type=int, nargs="+")
    parser.add_argument(
        "--clip_max_norm", default=0.05, type=float, help="gradient clipping max norm"
    )
    parser.add_argument("--sgd", action="store_true")

    # Variants of Deformable DETR
    parser.add_argument("--with_box_refine", default=False, action="store_true")
    parser.add_argument("--two_stage", default=False, action="store_true")

    # Model parameters
    parser.add_argument(
        "--frozen_weights",
        type=str,
        default=None,
        help="Path to the pretrained model. If set, only the mask head will be trained",
    )

    # * Backbone
    parser.add_argument(
        "--backbone", default="resnet50", type=str, help="Name of the convolutional backbone to use"
    )
    parser.add_argument(
        "--dilation",
        action="store_true",
        help="If true, we replace stride with dilation in the last convolutional block (DC5)",
    )
    parser.add_argument(
        "--position_embedding",
        default="sine",
        type=str,
        choices=("sine", "learned"),
        help="Type of positional embedding to use on top of the image features",
    )
    parser.add_argument(
        "--position_embedding_scale", default=2 * np.pi, type=float, help="position / size * scale"
    )
    parser.add_argument(
        "--num_feature_levels", default=4, type=int, help="number of feature levels"
    )

    # * Transformer
    parser.add_argument(
        "--enc_layers", default=6, type=int, help="Number of encoding layers in the transformer"
    )
    parser.add_argument(
        "--dec_layers", default=6, type=int, help="Number of decoding layers in the transformer"
    )
    parser.add_argument(
        "--dim_feedforward",
        default=1024,
        type=int,
        help="Intermediate size of the feedforward layers in the transformer blocks",
    )
    parser.add_argument(
        "--hidden_dim",
        default=256,
        type=int,
        help="Size of the embeddings (dimension of the transformer)",
    )
    parser.add_argument(
        "--dropout", default=0.1, type=float, help="Dropout applied in the transformer"
    )
    parser.add_argument(
        "--nheads",
        default=8,
        type=int,
        help="Number of attention heads inside the transformer's attentions",
    )
    parser.add_argument("--num_queries", default=300, type=int, help="Number of query slots")
    parser.add_argument("--dec_n_points", default=4, type=int)
    parser.add_argument("--enc_n_points", default=4, type=int)

    # * Segmentation
    parser.add_argument(
        "--masks", action="store_true", help="Train segmentation head if the flag is provided"
    )

    # Loss
    parser.add_argument(
        "--no_aux_loss",
        dest="aux_loss",
        action="store_false",
        help="Disables auxiliary decoding losses (loss at each layer)",
    )

    # * Matcher
    parser.add_argument(
        "--set_cost_class", default=3, type=float, help="Class coefficient in the matching cost"
    )
    parser.add_argument(
        "--set_cost_bbox", default=5, type=float, help="L1 box coefficient in the matching cost"
    )
    parser.add_argument(
        "--set_cost_giou", default=2, type=float, help="giou box coefficient in the matching cost"
    )

    # * Loss coefficients
    parser.add_argument("--mask_loss_coef", default=1, type=float)
    parser.add_argument("--dice_loss_coef", default=1, type=float)
    parser.add_argument("--cls_loss_coef", default=3, type=float)
    parser.add_argument("--bbox_loss_coef", default=5, type=float)
    parser.add_argument("--giou_loss_coef", default=2, type=float)
    parser.add_argument("--focal_alpha", default=0.25, type=float)

    parser.add_argument("--feature_loss_coef", default=2, type=float)
    parser.add_argument("--label_map", default=False, action="store_true")
    parser.add_argument("--max_len", default=15, type=int)
    parser.add_argument(
        "--clip_feat_path",
        default="./clip_feat_coco.pkl",
        type=str,
    )
    parser.add_argument("--prob", default=0.5, type=float)

    # dataset parameters
    parser.add_argument("--dataset_file", default="coco")
    parser.add_argument("--coco_path", default="./data/coco", type=str)
    parser.add_argument("--coco_panoptic_path", type=str)
    parser.add_argument("--lvis_path", default="./data/lvis", type=str)
    parser.add_argument("--remove_difficult", action="store_true")

    parser.add_argument("--output_dir", default="", help="path where to save, empty for no saving")
    parser.add_argument("--device", default="cuda", help="device to use for training / testing")
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--resume", default="", help="resume from checkpoint")
    parser.add_argument("--start_epoch", default=0, type=int, metavar="N", help="start epoch")
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--num_workers", default=4, type=int)
    parser.add_argument("--eval_period", default=1, type=int)
    parser.add_argument(
        "--cache_mode", default=False, action="store_true", help="whether to cache images on memory"
    )
    parser.add_argument("--amp", default=False, action="store_true")

    return parser


def main(args):
    # Initialize distributed mode and print git SHA for version tracking
    utils.init_distributed_mode(args)
    print("git:\n  {}\n".format(utils.get_sha()))

    # Ensure frozen weights are only used with segmentation
    if args.frozen_weights is not None:
        assert args.masks, "Frozen training is meant for segmentation only"
    print(args)

    # Set device and seed for reproducibility
    device = torch.device(args.device)
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Build the model, criterion, and postprocessors
    model, criterion, postprocessors = build_model(args)
    model.to(device)
    model_without_ddp = model

    # Log the number of model parameters
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("number of params:", n_parameters)

    # Build datasets and create data loaders
    dataset_train = build_dataset(image_set="train", args=args)
    dataset_val = build_dataset(image_set="val", args=args)

    if args.distributed:
        if args.cache_mode:
            sampler_train = samplers.NodeDistributedSampler(dataset_train)
            sampler_val = samplers.NodeDistributedSampler(dataset_val, shuffle=False)
        else:
            sampler_train = samplers.DistributedSampler(dataset_train)
            sampler_val = samplers.DistributedSampler(dataset_val, shuffle=False)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, args.batch_size, drop_last=True
    )
    data_loader_train = DataLoader(
        dataset_train, batch_sampler=batch_sampler_train,
        collate_fn=utils.collate_fn, num_workers=args.num_workers, pin_memory=True
    )
    data_loader_val = DataLoader(
        dataset_val, args.batch_size, sampler=sampler_val,
        drop_last=False, collate_fn=utils.collate_fn,
        num_workers=args.num_workers, pin_memory=True
    )

    # Create an optimizer and learning rate scheduler
    def match_name_keywords(n, name_keywords):
        return any(b in n for b in name_keywords)

    param_dicts = [
        {
            "params": [p for n, p in model_without_ddp.named_parameters() if p.requires_grad and not
                       match_name_keywords(n, args.lr_backbone_names) and not
                       match_name_keywords(n, args.lr_linear_proj_names)],
            "lr": args.lr,
        },
        {
            "params": [p for n, p in model_without_ddp.named_parameters() if p.requires_grad and
                       match_name_keywords(n, args.lr_backbone_names)],
            "lr": args.lr_backbone,
        },
        {
            "params": [p for n, p in model_without_ddp.named_parameters() if p.requires_grad and
                       match_name_keywords(n, args.lr_linear_proj_names)],
            "lr": args.lr * args.lr_linear_proj_mult,
        },
    ]
    optimizer = (torch.optim.SGD(param_dicts, lr=args.lr, momentum=0.9,
                                 weight_decay=args.weight_decay) if args.sgd else
                 torch.optim.AdamW(param_dicts, lr=args.lr, weight_decay=args.weight_decay))

    lr_scheduler = MinLRScheduler(optimizer, min_lr=1e-7)

    # Set up DistributedDataParallel if in distributed mode
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    # Build datasets and load pre-trained weights if needed
    base_ds = (get_coco_api_from_dataset(datasets.coco.build("val", args)) if
               args.dataset_file == "coco_panoptic" else get_coco_api_from_dataset(dataset_val))

    if args.frozen_weights is not None:
        checkpoint = torch.load(args.frozen_weights, map_location="cpu")
        model_without_ddp.detr.load_state_dict(checkpoint["model"])

    # Resume training from a checkpoint if specified
    output_dir = Path(args.output_dir)
    if args.resume:
        checkpoint = (torch.hub.load_state_dict_from_url(args.resume, map_location="cpu",
                                                         check_hash=True) if args.resume.startswith("https") else
                      torch.load(args.resume, map_location="cpu"))
        model_without_ddp.load_state_dict(checkpoint["model"], strict=False)

        if not args.eval and "optimizer" in checkpoint and "lr_scheduler" in checkpoint and "epoch" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
            lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
            args.start_epoch = checkpoint["epoch"] + 1

    if args.eval:
        evaluate_model(args, model, criterion, postprocessors, data_loader_val, base_ds, device)
        return

    print("Start training")
    start_training(args, model, model_without_ddp, criterion, optimizer, lr_scheduler,
                   data_loader_train, data_loader_val, base_ds, output_dir, device, n_parameters)


def evaluate_model(args, model, criterion, postprocessors, data_loader_val, base_ds, device):
    if args.dataset_file != "lvis":
        test_stats, coco_evaluator = evaluate(
            model, criterion, postprocessors, data_loader_val, base_ds,
            device, args.output_dir, args.label_map, args.amp
        )
        if args.output_dir:
            utils.save_on_master(coco_evaluator.coco_eval["bbox"].eval, Path(args.output_dir) / "eval.pth")
    else:
        lvis_evaluate(
            model, criterion, postprocessors, data_loader_val, base_ds,
            device, args.output_dir, args.label_map, args.amp
        )


def start_training(args, model, model_without_ddp, criterion, optimizer, lr_scheduler,
                   data_loader_train, data_loader_val, base_ds, output_dir, device, n_parameters):
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        train_stats = train_one_epoch(
            model, criterion, data_loader_train, optimizer, device, epoch,
            args.clip_max_norm, args.masks, args.amp
        )

        lr_scheduler.step()

        if args.output_dir:
            save_checkpoint(output_dir, model_without_ddp, optimizer, lr_scheduler, epoch, args)

        if epoch % args.eval_period == 0:
            test_stats, coco_evaluator = evaluate(
                model, criterion, postprocessors, data_loader_val, base_ds,
                device, args.output_dir, args.label_map, args.amp
            )
            log_stats = {**{f"train_{k}": v for k, v in train_stats.items()},
                         **{f"test_{k}": v for k, v in test_stats.items()},
                         "epoch": epoch, "n_parameters": n_parameters}
        else:
            log_stats = {**{f"train_{k}": v for k, v in train_stats.items()},
                         "epoch": epoch, "n_parameters": n_parameters}

        if args.output_dir and utils.is_main_process():
            save_logs(output_dir, log_stats)

    print("Training completed in {}".format(str(datetime.timedelta(seconds=int(time.time() - start_time)))))


def save_checkpoint(output_dir, model_without_ddp, optimizer, lr_scheduler, epoch, args):
    checkpoint_paths = [output_dir / "checkpoint.pth"]
    if (epoch + 1) % args.lr_drop == 0 or (epoch + 1) % 5 == 0:
        checkpoint_paths.append(output_dir / f"checkpoint{epoch:04}.pth")
    for checkpoint_path in checkpoint_paths:
        utils.save_on_master({
            "model": model_without_ddp.state_dict(),
            "optimizer": optimizer.state_dict(),
            "lr_scheduler": lr_scheduler.state_dict(),
            "epoch": epoch, "args": args
        }, checkpoint_path)


def save_logs(output_dir, log_stats):
    with (output_dir / "log.txt").open("a") as f:
        f.write(json.dumps(log_stats) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "OV DETR training and evaluation script", parents=[get_args_parser()]
    )
    args = parser.parse_args()
    from engine_ov import evaluate, lvis_evaluate, train_one_epoch

    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
