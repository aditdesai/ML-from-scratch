def get_config():
    return {
        "d_model": 12,
        "n_classes": 10,
        "img_size": (32, 32),
        "patch_size": (16, 16),
        "dropout": 0.1,
        "n_channels": 1,
        "n_heads": 3,
        "n_layers": 3,
        "batch_size": 128,
        "epochs": 5,
        "lr": 0.005
    }