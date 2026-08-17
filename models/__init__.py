from .guided_diffusion.unet import UNetModel, UNetModelAttn


def get_flow_model(config):
    num_classes = getattr(config, 'num_classes', None)
    
    if config.layout:
        model = UNetModelAttn(
            image_size=config.image_size // 8,
            in_channels=config.num_in_channels,
            model_channels=config.nf,
            out_channels=config.num_out_channels,
            num_res_blocks=config.num_res_blocks,
            attention_resolutions=config.attn_resolutions,
            dropout=config.dropout,
            channel_mult=config.ch_mult,
            conv_resample=config.resamp_with_conv,
            dims=2,
            num_classes=num_classes,
            use_checkpoint=False,
            use_fp16=False,
            num_heads=config.num_heads,
            num_head_channels=config.num_head_channels,
            num_heads_upsample=config.num_head_upsample,
            use_scale_shift_norm=config.use_scale_shift_norm,
            resblock_updown=config.resblock_updown,
            use_new_attention_order=config.use_new_attention_order,
            use_spatial_transformer=True,
            transformer_depth=3,
            context_dim=512,
            legacy=True,
        )
    else:
        model = UNetModel(
            image_size=config.image_size // 8,
            in_channels=config.num_in_channels,
            model_channels=config.nf,
            out_channels=config.num_out_channels,
            num_res_blocks=config.num_res_blocks,
            attention_resolutions=config.attn_resolutions,
            dropout=config.dropout,
            channel_mult=config.ch_mult,
            conv_resample=config.resamp_with_conv,
            dims=2,
            num_classes=num_classes,
            use_checkpoint=False,
            use_fp16=False,
            num_heads=config.num_heads,
            num_head_channels=config.num_head_channels,
            num_heads_upsample=config.num_head_upsample,
            use_scale_shift_norm=config.use_scale_shift_norm,
            resblock_updown=config.resblock_updown,
            use_new_attention_order=config.use_new_attention_order,
        )

    return model
