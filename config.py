import torch

CONFIG = dict(
    project_name="PL-RSNA-2024-Lumbar-Spine-Classification",
    weights_path="data/efficientnet_b0.ra_in1k.pth",
    artifact_name="rsnaEffNetModel",
    load_kernel=None,
    load_last=True,
    n_folds=5,
    backbone="efficientnet_b0.ra_in1k",  # tf_efficientnetv2_s_in21ft1k
    img_size=384,
    n_slice_per_c=16,
    in_chans=1,

    drop_rate=0.,
    drop_rate_last=0.3,
    drop_path_rate=0.,
    p_mixup=0.5,
    p_rand_order_v1=0.2,
    lr=1e-3,

    out_dim=3,
    epochs=2,
    batch_size=8,
    device=torch.device("cuda") if torch.cuda.is_available() else "cpu",
    seed=2024,

    patience=7,
)
