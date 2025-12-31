import torch
import torch.nn as nn
import timm
from cbam import CBAM
from tripletattention import TripletAttention
from config import CONFIG


import torch
import torch.nn as nn
import timm
from timm.layers import Mlp

class ISICModel(nn.Module):
    def __init__(
        self,
        model_name,
        pretrained,
        replace_groups,
        attention_type,  
        meta_cnn_integration,
        meta_cols,
        num_classes=1,
        backbone_ckpt_path=None
    ):
        super().__init__()

        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,
            global_pool=''
        )

        if attention_type is not None:
            replace_se_with_new_attention(
                self.backbone,
                replace_groups,
                attention_type=attention_type,
            )

        if backbone_ckpt_path is not None:
            sd = torch.load(backbone_ckpt_path, map_location="cpu")
            self.backbone.load_state_dict(sd, strict=True)
            print(f"Backbone weights loaded from: {backbone_ckpt_path}")

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.2)

        backbone_out_dim = 1280  # EfficientNet-B0

        if meta_cols:
            meta_dim = len(meta_cols)
        else:
            meta_dim = 0

        self.meta_cnn_integration = meta_cnn_integration

        if meta_cnn_integration is None:
            self.mlp_out_dim = None
            fc_in_dim = backbone_out_dim

        else:
            if meta_cnn_integration == "hadamard":
                self.mlp_out_dim = backbone_out_dim
            elif meta_cnn_integration == "concatenation":
                self.mlp_out_dim = 256

            meta_hidden_dim = self.mlp_out_dim // 2

            self.meta_mlp = Mlp(
                in_features=meta_dim,
                hidden_features=meta_hidden_dim,
                out_features=self.mlp_out_dim,
                act_layer=nn.ReLU,
                drop=0.1
            )

            self.meta_ln = nn.LayerNorm(self.mlp_out_dim)

            if meta_cnn_integration == "hadamard":
                fc_in_dim = backbone_out_dim
            elif meta_cnn_integration == "concatenation":
                fc_in_dim = backbone_out_dim + self.mlp_out_dim

        self.fc = nn.Linear(fc_in_dim, num_classes)



    def forward(self, x, meta=None):
        x = self.backbone(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
    
        if self.meta_cnn_integration is None:
            return self.fc(x)
    
        meta_feat = self.meta_mlp(meta)
    
        if self.meta_cnn_integration == "concatenation":
            fused = torch.cat([x, meta_feat], dim=1)
    
        elif self.meta_cnn_integration == "hadamard":
            meta_feat = self.meta_ln(meta_feat)
            fused = x * (1 + meta_feat)
            
        return self.fc(fused)




def replace_se_with_new_attention(model: nn.Module, replace_groups, attention_type):

    stage_to_group = {0: 0, 1: 0, 2: 1, 3: 1, 4: 2, 5: 2, 6: 3}

    seen = replaced = 0

    for stage_idx, stage in enumerate(model.blocks):
        gid = stage_to_group[stage_idx]
        do_replace = bool(replace_groups[gid])

        for block in stage:
            if not hasattr(block, "se"):
                continue

            seen += 1
            if not do_replace:
                continue

            if attention_type == "triplet":
                block.se = TripletAttention(no_spatial=False)
            else:
                in_channels = block.se.conv_reduce.in_channels
                block.se = CBAM(gate_channels=in_channels)

            replaced += 1

    print(f"Attention {attention_type}: {replaced}/{seen} SE blocks replaced")
