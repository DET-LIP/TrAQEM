import copy
import math
import torch
import torch.nn.functional as F
from torch import nn
from util.misc import NestedTensor, inverse_sigmoid, nested_tensor_from_tensor_list

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

class MLP(nn.Module):
    """Very simple multi-layer perceptron (also called FFN)"""
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

class MemoryBank(nn.Module):
    def __init__(self, memory_size, feature_dim):
        super().__init__()
        self.memory_size = memory_size
        self.feature_dim = feature_dim
        self.memory = {}  # Store the features per label

    def add(self, label, features):
        if label not in self.memory:
            self.memory[label] = []
        self.memory[label].append(features)
        if len(self.memory[label]) > self.memory_size:
            self.memory[label].pop(0)  # Remove the oldest feature if the memory exceeds the size limit

    def get_memory(self, label):
        if label in self.memory:
            return torch.stack(self.memory[label])
        else:
            return None

    def reset(self):
        self.memory = {}

class DeformableDETR(nn.Module):
    def __init__(
        self,
        backbone,
        transformer,
        num_classes,
        num_queries,
        num_feature_levels,
        aux_loss=True,
        with_box_refine=False,
        two_stage=False,
        cls_out_channels=8,
        memory_size=300,
        feature_dim=512
    ):
        super().__init__()
        self.num_queries = num_queries
        self.transformer = transformer
        hidden_dim = transformer.d_model
        self.class_embed = nn.Linear(hidden_dim, cls_out_channels)
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.num_feature_levels = num_feature_levels
        
        # Memory bank for storing visual features
        self.memory_bank = MemoryBank(memory_size=memory_size, feature_dim=feature_dim)

        # Initialize category-specific queries
        self.category_query_embed = nn.Embedding(num_classes, hidden_dim)
        
        # Optionally, a layer to refine selected queries
        self.query_refinement_layer = nn.Linear(hidden_dim, hidden_dim)

        if not two_stage:
            self.query_embed = nn.Embedding(num_queries, hidden_dim * 2)
        if num_feature_levels > 1:
            num_backbone_outs = len(backbone.strides)
            input_proj_list = []
            for _ in range(num_backbone_outs):
                in_channels = backbone.num_channels[_]
                input_proj_list.append(
                    nn.Sequential(
                        nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
                        nn.GroupNorm(32, hidden_dim),
                    )
                )
            for _ in range(num_feature_levels - num_backbone_outs):
                input_proj_list.append(
                    nn.Sequential(
                        nn.Conv2d(in_channels, hidden_dim, kernel_size=3, stride=2, padding=1),
                        nn.GroupNorm(32, hidden_dim),
                    )
                )
                in_channels = hidden_dim
            self.input_proj = nn.ModuleList(input_proj_list)
        else:
            self.input_proj = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Conv2d(backbone.num_channels[0], hidden_dim, kernel_size=1),
                        nn.GroupNorm(32, hidden_dim),
                    )
                ]
            )
        self.backbone = backbone
        self.aux_loss = aux_loss
        self.with_box_refine = with_box_refine
        self.two_stage = two_stage

        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        self.class_embed.bias.data = torch.ones(cls_out_channels) * bias_value
        nn.init.constant_(self.bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(self.bbox_embed.layers[-1].bias.data, 0)
        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)

        num_pred = (
            (transformer.decoder.num_layers + 1) if two_stage else transformer.decoder.num_layers
        )
        if with_box_refine:
            self.class_embed = _get_clones(self.class_embed, num_pred)
            self.bbox_embed = _get_clones(self.bbox_embed, num_pred)
            nn.init.constant_(self.bbox_embed[0].layers[-1].bias.data[2:], -2.0)
            self.transformer.decoder.bbox_embed = self.bbox_embed
        else:
            nn.init.constant_(self.bbox_embed.layers[-1].bias.data[2:], -2.0)
            self.class_embed = nn.ModuleList([self.class_embed for _ in range(num_pred)])
            self.bbox_embed = nn.ModuleList([self.bbox_embed for _ in range(num_pred)])
            self.transformer.decoder.bbox_embed = None
        if two_stage:
            self.transformer.decoder.class_embed = self.class_embed
            for box_embed in self.bbox_embed:
                nn.init.constant_(box_embed.layers[-1].bias.data[2:], 0.0)

    def get_outputs_class(self, layer, data):
        return layer(data)

    def forward(self, samples: NestedTensor, targets=None):
        if not isinstance(samples, NestedTensor):
            samples = nested_tensor_from_tensor_list(samples)
        features, pos = self.backbone(samples)

        srcs = []
        masks = []
        for l, feat in enumerate(features):
            src, mask = feat.decompose()
            srcs.append(self.input_proj[l](src))
            masks.append(mask)
            assert mask is not None
        if self.num_feature_levels > len(srcs):
            _len_srcs = len(srcs)
            for l in range(_len_srcs, self.num_feature_levels):
                if l == _len_srcs:
                    src = self.input_proj[l](features[-1].tensors)
                else:
                    src = self.input_proj[l](srcs[-1])
                m = samples.mask
                mask = F.interpolate(m[None].float(), size=src.shape[-2:]).to(torch.bool)[0]
                pos_l = self.backbone[1](NestedTensor(src, mask)).to(src.dtype)
                srcs.append(src)
                masks.append(mask)
                pos.append(pos_l)

        # Only use visual embeddings in the memory bank
        uniq_labels = torch.cat([t["labels"] for t in targets]) if targets else torch.tensor([])
        uniq_labels = torch.unique(uniq_labels) if len(uniq_labels) > 0 else torch.tensor([])

        # Update the memory bank with visual features from the current frame
        for i, label in enumerate(uniq_labels):
            visual_features = self.extract_visual_features(features)  # Extract visual features
            self.memory_bank.add(label.item(), visual_features)

        # Refine the selected category queries based on memory bank
        selected_queries = self.category_query_embed(uniq_labels) if len(uniq_labels) > 0 else None
        if selected_queries is not None:
            memory_features = [self.memory_bank.get_memory(label.item()) for label in uniq_labels]
            memory_features = torch.stack(memory_features).mean(dim=0) if len(memory_features) > 0 else None
            if memory_features is not None:
                selected_queries = selected_queries + memory_features

        refined_queries = self.query_refinement_layer(selected_queries) if selected_queries is not None else None

        # Pass the refined queries through the transformer
        (
            hs,
            init_reference,
            inter_references,
            enc_outputs_class,
            enc_outputs_coord_unact,
        ), _ = self.transformer(srcs, masks, pos, refined_queries)

        # Calculate outputs
        outputs_classes = []
        outputs_coords = []
        for lvl in range(hs.shape[0]):
            if lvl == 0:
                reference = init_reference
            else:
                reference = inter_references[lvl - 1]
            reference = inverse_sigmoid(reference)
            outputs_class = self.get_outputs_class(self.class_embed[lvl], hs[lvl])
            tmp = self.bbox_embed[lvl](hs[lvl])
            if reference.shape[-1] == 4:
                tmp += reference
            else:
                assert reference.shape[-1] == 2
                tmp[..., :2] += reference
            outputs_coord = tmp.sigmoid()
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)
        outputs_class = torch.stack(outputs_classes)
        outputs_coord = torch.stack(outputs_coords)
        out = {"pred_logits": outputs_class[-1], "pred_boxes": outputs_coord[-1]}
        if self.aux_loss:
            out["aux_outputs"] = self._set_aux_loss(outputs_class, outputs_coord)

        if self.two_stage:
            enc_outputs_coord = enc_outputs_coord_unact.sigmoid()
            out["enc_outputs"] = {"pred_logits": enc_outputs_class, "pred_boxes": enc_outputs_coord}
        return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        return [
            {"pred_logits": a, "pred_boxes": b}
            for a, b in zip(outputs_class[:-1], outputs_coord[:-1])
        ]

    def extract_visual_features(self, features):
        """
        Custom method to extract visual features from samples using the model backbone.
        This method aggregates visual features across multiple levels.
        """
        # Example approach: Take average feature vectors across spatial dimensions from the last level
        last_level_features = features[-1].tensors
        visual_features = last_level_features.mean(dim=[-2, -1])  # Mean pooling across spatial dimensions
        return visual_features

class OVDETR(DeformableDETR):
    def __init__(
        self,
        backbone,
        transformer,
        num_classes,
        num_queries,
        num_feature_levels,
        aux_loss=True,
        with_box_refine=False,
        two_stage=False,
        cls_out_channels=1,
        dataset_file="coco",
        zeroshot_w=None,
        max_len=15,
        clip_feat_path=None,
        prob=0.5,
        memory_size=100,
        feature_dim=512,
    ):
        super().__init__(
            backbone,
            transformer,
            num_classes,
            num_queries,
            num_feature_levels,
            aux_loss,
            with_box_refine,
            two_stage,
            cls_out_channels,
            memory_size,
            feature_dim
        )
        self.zeroshot_w = zeroshot_w.t()
        self.prob = prob
