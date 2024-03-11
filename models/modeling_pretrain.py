import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from functools import partial

from models.modeling_finetune_amd import Block, _cfg, PatchEmbed, get_sinusoid_encoding_table
from timm.models.registry import register_model
from timm.models.layers import trunc_normal_ as __call_trunc_normal_


def trunc_normal_(tensor, mean=0., std=1.):
    __call_trunc_normal_(tensor, mean=mean, std=std, a=-std, b=std)


__all__ = [
    'pretrain_mae_base_patch16_224', 
    'pretrain_mae_large_patch16_224', 
]


class PretrainVisionTransformerEncoder(nn.Module):
    def __init__(self, use_checkpoint=False, img_size=224, patch_size=16, in_chans=3, num_classes=0, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, init_values=None, tubelet_size=2,
                 use_learnable_pos_emb=False, teacher_dim=1024,layer_to_get=None):
        super().__init__()
        self.layer_to_get = layer_to_get
        self.use_checkpoint = use_checkpoint
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,tubelet_size=tubelet_size)
        num_patches = self.patch_embed.num_patches

        if use_learnable_pos_emb:
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        else:
            # sine-cosine positional embeddings 
            self.pos_embed = get_sinusoid_encoding_table(num_patches, embed_dim)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                init_values=init_values)
            for i in range(depth)])
        self.norm =  norm_layer(embed_dim)
        
        self.projection_heads = nn.ModuleList([
                                    nn.Linear(embed_dim, teacher_dim, bias=False)
                                    for _ in range(len(self.layer_to_get['dir']))]
                                )

        if use_learnable_pos_emb:
            trunc_normal_(self.pos_embed, std=.02)
        
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed'}

    def forward_features(self, x, mask):
        _, _, T, _, _ = x.shape
        x = self.patch_embed(x).contiguous()
        x = x + self.pos_embed.type_as(x).to(x.device).clone().detach()

        B, _, C = x.shape
        x_vis = x[~mask].reshape(B, -1, C)  # ~mask means visible

        dir_student_feats = []
        gen_student_feats = []
        for l, blk in enumerate(self.blocks):
            if self.use_checkpoint:
                x_vis = checkpoint(blk, x_vis)
            else:
                x_vis = blk(x_vis)
            if l + 1 in self.layer_to_get['dir']:
                dir_student_feats.append(x_vis) # feat for direct align
            if l + 1 in self.layer_to_get['gen']:
                gen_student_feats.append(x_vis) # feat for generation align
        dir_student_feats_proj = []
        gen_student_feats_proj = []
        x_vis = self.norm(x_vis)
        
        for i,feat in enumerate(dir_student_feats):
            feat = self.projection_heads[i](feat)
            dir_student_feats_proj.append(feat)
            
        for i,feat in enumerate(gen_student_feats):
            feat = self.projection_heads[i](feat)
            gen_student_feats_proj.append(feat)
                
        return x_vis, dir_student_feats_proj, gen_student_feats_proj

    def forward(self, x, mask):
        x, dir_student_feat, gen_student_feat = self.forward_features(x, mask)
        return x, dir_student_feat, gen_student_feat

class PretrainVisionTransformerDecoder(nn.Module):
    def __init__(self, use_checkpoint=False, patch_size=16, num_classes=768, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, init_values=None, tubelet_size=2
                 ):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.num_classes = num_classes
        assert num_classes == 3 * tubelet_size * patch_size ** 2 
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.patch_size = patch_size

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                init_values=init_values)
            for i in range(depth)])
        self.norm =  norm_layer(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        self.apply(self._init_weights)


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed'}

    def forward(self, x, return_token_num):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint(blk, x)
            else:
                x = blk(x)

        if return_token_num > 0:
            x = self.head(self.norm(x[:, -return_token_num:]))  # only return the mask tokens predict pixels
        else:
            x = self.head(self.norm(x))  # [B, N, 3*16^2]
        return x


class Generator(nn.Module):
    def __init__(self, use_checkpoint=False, patch_size=16, num_classes=768, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, init_values=None, tubelet_size=2
                 ):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.num_classes = num_classes
        assert num_classes == 3 * tubelet_size * patch_size ** 2
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.patch_size = patch_size

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                init_values=init_values)
            for i in range(depth)])
        self.norm =  norm_layer(embed_dim)
        self.apply(self._init_weights)


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed'}

    def forward(self, x, return_token_num):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint(blk, x)
            else:
                x = blk(x)

        return x[:, -return_token_num:]


class PretrainAMD(nn.Module):
    def __init__(self,
                 use_checkpoint = False,
                 generator_depth=2,
                 img_size=224, 
                 patch_size=16,
                 encoder_in_chans=3, 
                 encoder_num_classes=0, 
                 encoder_embed_dim=768, 
                 encoder_depth=12,
                 encoder_num_heads=12, 
                 decoder_num_classes=1536,
                 decoder_embed_dim=512, 
                 decoder_depth=8,
                 decoder_num_heads=8, 
                 mlp_ratio=4., 
                 qkv_bias=False, 
                 qk_scale=None, 
                 drop_rate=0., 
                 attn_drop_rate=0.,
                 drop_path_rate=0., 
                 norm_layer=nn.LayerNorm, 
                 init_values=0.,
                 use_learnable_pos_emb=False,
                 tubelet_size=2,
                 num_classes=0, # avoid the error from create_fn in timm
                 in_chans=0, # avoid the error from create_fn in timm
                 teacher_dim = 1024,
                 layer_to_get = None
                 ):
        super().__init__()
        self.teacher_dim = teacher_dim
        self.encoder = PretrainVisionTransformerEncoder(
            use_checkpoint = use_checkpoint,
            img_size=img_size, 
            patch_size=patch_size, 
            in_chans=encoder_in_chans,
            num_classes=encoder_num_classes, 
            embed_dim=encoder_embed_dim, 
            depth=encoder_depth,
            num_heads=encoder_num_heads, 
            mlp_ratio=mlp_ratio, 
            qkv_bias=qkv_bias, 
            qk_scale=qk_scale, 
            drop_rate=drop_rate, 
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate, 
            norm_layer=norm_layer, 
            init_values=init_values,
            tubelet_size=tubelet_size,
            use_learnable_pos_emb=use_learnable_pos_emb,
            teacher_dim=teacher_dim,
            layer_to_get=layer_to_get)

        # create generator and mask token in it
        self.generators = nn.ModuleList([
            Generator(
                use_checkpoint=use_checkpoint,
                patch_size=patch_size,
                num_classes=decoder_num_classes,
                embed_dim=teacher_dim,
                depth=generator_depth,
                num_heads=teacher_dim//64, # each head handles 64 dim
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop_rate=drop_rate,
                attn_drop_rate=attn_drop_rate,
                drop_path_rate=drop_path_rate,
                norm_layer=norm_layer,
                init_values=init_values,
                tubelet_size=tubelet_size
            )
        for _ in layer_to_get['gen']])
        
        self.generator_mask_tokens = [nn.Parameter(torch.zeros(1, 1, teacher_dim))
                                      for _ in layer_to_get['gen']]

        # decoder for reconstruct pixels
        self.decoder = PretrainVisionTransformerDecoder(
            use_checkpoint=use_checkpoint,
            patch_size=patch_size,
            num_classes=decoder_num_classes,
            embed_dim=decoder_embed_dim, 
            depth=decoder_depth,
            num_heads=decoder_num_heads,
            mlp_ratio=mlp_ratio, 
            qkv_bias=qkv_bias, 
            qk_scale=qk_scale, 
            drop_rate=drop_rate, 
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate, 
            norm_layer=norm_layer, 
            init_values=init_values,
            tubelet_size=tubelet_size)

        self.encoder_to_decoder = nn.Linear(encoder_embed_dim, decoder_embed_dim, bias=False)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.pos_embed = get_sinusoid_encoding_table(self.encoder.patch_embed.num_patches, decoder_embed_dim)

        self.pos_embed_generator = get_sinusoid_encoding_table(self.encoder.patch_embed.num_patches,
                                                                teacher_dim)

        trunc_normal_(self.mask_token, std=.02)
        for i in range(len(self.generator_mask_tokens)):
            trunc_normal_(self.generator_mask_tokens[i], std=.02)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'mask_token'}

    def forward(self, x, mask_all):
        mask, diff, _ = mask_all
        x_vis, dir_feat_student, gen_feat_student = self.encoder(x, mask)
        x_vis = self.encoder_to_decoder(x_vis)
        B, N, C = x_vis.shape
        # pred for generation align
        expand_pos_embed_generator = self.pos_embed_generator.expand(B, -1, -1).type_as(x).to(
            x.device).clone().detach()
        pos_emd_vis_generation = expand_pos_embed_generator[~mask].reshape(B, -1, self.teacher_dim)
        pos_emd_mask_generation = expand_pos_embed_generator[diff].reshape(B, -1, self.teacher_dim)

        gen_feat_student_pred = []
        for i,stu_feat in enumerate(gen_feat_student):
            stu_feat = torch.cat(
                    [stu_feat + pos_emd_vis_generation, self.generator_mask_tokens[i].to(x.device) + pos_emd_mask_generation],
                    dim=1)
            stu_feat = self.generators[i](stu_feat, diff[0].sum())
            gen_feat_student_pred.append(stu_feat)
            
        # reconstruction for pixels
        expand_pos_embed = self.pos_embed.expand(B, -1, -1).type_as(x).to(x.device).clone().detach()
        pos_emd_vis = expand_pos_embed[~mask].reshape(B, -1, C)
        pos_emd_mask = expand_pos_embed[mask].reshape(B, -1, C)
        x_full = torch.cat([x_vis + pos_emd_vis, self.mask_token + pos_emd_mask], dim=1)

        x = self.decoder(x_full, pos_emd_mask.shape[1])

        return x, dir_feat_student, gen_feat_student_pred


@register_model
def pretrain_mae_small_patch16_224(pretrained=False, **kwargs):
    model = PretrainAMD(
        img_size=224,
        patch_size=16,
        encoder_embed_dim=384,
        encoder_depth=12,
        encoder_num_heads=6,
        encoder_num_classes=0,
        decoder_num_classes=1536,
        decoder_embed_dim=192,
        decoder_num_heads=3,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.load(
            kwargs["init_ckpt"], map_location="cpu"
        )
        model.load_state_dict(checkpoint["model"])
    return model

@register_model
def pretrain_mae_base_patch16_224(pretrained=False, **kwargs):
    model = PretrainAMD(
        img_size=224,
        patch_size=16, 
        encoder_embed_dim=768, 
        encoder_depth=12, 
        encoder_num_heads=12,
        encoder_num_classes=0,
        decoder_num_classes=1536,
        decoder_embed_dim=384,
        decoder_num_heads=6,
        mlp_ratio=4,
        qkv_bias=True, 
        norm_layer=partial(nn.LayerNorm, eps=1e-6), 
        **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.load(
            kwargs["init_ckpt"], map_location="cpu"
        )
        model.load_state_dict(checkpoint["model"])
    return model
 

@register_model
def pretrain_mae_large_patch16_224(pretrained=False, **kwargs):
    model = PretrainAMD(
        img_size=224,
        patch_size=16, 
        encoder_embed_dim=1024, 
        encoder_depth=24, 
        encoder_num_heads=16,
        encoder_num_classes=0,
        decoder_num_classes=1536,
        decoder_embed_dim=512,
        decoder_num_heads=8,
        mlp_ratio=4, 
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), 
        **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.load(
            kwargs["init_ckpt"], map_location="cpu"
        )
        model.load_state_dict(checkpoint["model"])
    return model