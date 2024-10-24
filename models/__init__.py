import torch

from util.clip_utils import build_text_embedding_coco, build_text_embedding_lvis

from .backbone import build_backbone
from .deformable_transformer import build_deforamble_transformer
from .matcher import build_matcher
from .model import OVDETR
from .post_process import OVPostProcess, PostProcess, PostProcessSegm
from .segmentation import DETRsegm
from .set_criterion import OVSetCriterion


def build_model(args):
    # Determine the number of classes based on the dataset
    if args.dataset_file == "coco":
        num_classes = 8  # Example number of classes, modify as needed
    elif args.dataset_file == "lvis":
        num_classes = 1204  # Example number of classes for LVIS
    else:
        raise NotImplementedError("Only COCO and LVIS datasets are supported")

    device = torch.device(args.device)

    # Build the backbone of the model
    backbone = build_backbone(args)

    # Build the transformer component of the model
    transformer = build_deforamble_transformer(args)

    # Create zero-shot weights based on the dataset
    if args.dataset_file == "coco":
        zeroshot_w = build_text_embedding_coco()  # Loads zero-shot embeddings for COCO dataset
    elif args.dataset_file == "lvis":
        zeroshot_w = build_text_embedding_lvis()  # Loads zero-shot embeddings for LVIS dataset
    else:
        raise NotImplementedError

    # Initialize the main OVDETR model with the required components
    model = OVDETR(
        backbone,
        transformer,
        num_classes=num_classes,
        num_queries=args.num_queries,
        num_feature_levels=args.num_feature_levels,
        aux_loss=args.aux_loss,
        with_box_refine=args.with_box_refine,
        two_stage=args.two_stage,
        cls_out_channels=1,  # Class output channels; set appropriately
        dataset_file=args.dataset_file,
        zeroshot_w=zeroshot_w,
        max_len=args.max_len,
        clip_feat_path=args.clip_feat_path,
        prob=args.prob,
    )

    # Add segmentation head if the model is configured to use masks
    if args.masks:
        model = DETRsegm(model, freeze_detr=(args.frozen_weights is not None))

    # Build the matcher module for matching predicted and ground-truth boxes
    matcher = build_matcher(args)

    # Define the weight dictionary for losses
    weight_dict = {"loss_ce": args.cls_loss_coef, "loss_bbox": args.bbox_loss_coef, "loss_giou": args.giou_loss_coef}
    weight_dict["loss_embed"] = args.feature_loss_coef

    # Add weights for mask losses if segmentation is enabled
    if args.masks:
        weight_dict["loss_mask"] = args.mask_loss_coef
        weight_dict["loss_dice"] = args.dice_loss_coef

    # Handle auxiliary loss (i.e., losses at each intermediate layer of the model)
    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
        aux_weight_dict.update({k + "_enc": v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)

    # Define the set of loss types to be calculated
    losses = ["labels", "boxes", "embed"]
    if args.masks:
        losses = ["labels", "boxes", "masks"]

    # Create an instance of the criterion for calculating losses
    criterion = OVSetCriterion(
        num_classes=num_classes,
        matcher=matcher,
        weight_dict=weight_dict,
        losses=losses,
        focal_alpha=args.focal_alpha,
    )
    criterion.to(device)

    # Create the post-processing modules for inference
    postprocessors = {"bbox": OVPostProcess(num_queries=args.num_queries)}
    if args.masks:
        postprocessors["segm"] = PostProcessSegm()

    return model, criterion, postprocessors
