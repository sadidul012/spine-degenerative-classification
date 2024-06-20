import torch

CONFIG = dict(
    project_name="PL-RSNA-2024-Lumbar-Spine-Classification",
    weights_path="data/efficientnet_b0-0.pth",
    # data_path="/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification",
    data_path="/mnt/Cache/rsna-2024-lumbar-spine-degenerative-classification",
    artifact_name="rsnaEffNetModel",
    load_kernel=None,
    load_last=True,
    n_folds=10,
    backbone="efficientnet_b4",  # tf_efficientnetv2_s_in21ft1k
    # backbone="efficientnet_b0.ra_in1k",  # tf_efficientnetv2_s_in21ft1k
    img_size=384,
    n_slice_per_c=16,
    in_chans=1,

    drop_rate=0.,
    drop_rate_last=0.3,
    drop_path_rate=0.,
    p_mixup=0.5,
    p_rand_order_v1=0.2,
    lr=1e-4,

    out_dim=3,
    epochs=15,
    batch_size=24,
    device=torch.device("cuda") if torch.cuda.is_available() else "cpu",
    seed=8620,

    patience=7,
    AUG_PROB=0.75
)
