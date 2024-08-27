
def get_dataset_specified_config_1_1(dataset):
    """Get dataset specific."""
    cfg = {
        "ImageNet": {
            "TRAINER.KDPT_MAPLE_1_1.PROMPT_DEPTH": 3,
            "TRAINER.KDPT_MAPLE_1_1.N_CTX": 2,
            "TRAINER.W_T": 4.0,
            "TRAINER.W_I": 1.0,
        },
        "ImageNetSketch": {
            "TRAINER.KDPT_MAPLE_1_1.PROMPT_DEPTH": 3,
            "TRAINER.KDPT_MAPLE_1_1.N_CTX": 2,
            "TRAINER.W_T": 4.0,
            "TRAINER.W_I": 1.0,
        },
        "ImageNetV2": {
            "TRAINER.KDPT_MAPLE_1_1.PROMPT_DEPTH": 3,
            "TRAINER.KDPT_MAPLE_1_1.N_CTX": 2,
            "TRAINER.W_T": 4.0,
            "TRAINER.W_I": 1.0,
        },
        "ImageNetA": {
            "TRAINER.KDPT_MAPLE_1_1.PROMPT_DEPTH": 3,
            "TRAINER.KDPT_MAPLE_1_1.N_CTX": 2,
            "TRAINER.W_T": 4.0,
            "TRAINER.W_I": 1.0,
        },
        "ImageNetR": {
            "TRAINER.KDPT_MAPLE_1_1.PROMPT_DEPTH": 3,
            "TRAINER.KDPT_MAPLE_1_1.N_CTX": 2,
            "TRAINER.W_T": 4.0,
            "TRAINER.W_I": 1.0,
        },
        "Caltech101": {
            "TRAINER.KDPT_MAPLE_1_1.PROMPT_DEPTH": 3,
            "TRAINER.KDPT_MAPLE_1_1.N_CTX": 2,
            "TRAINER.W_T": 4.0,
            "TRAINER.W_I": 1.0,
        },
        "OxfordPets": {
            "TRAINER.KDPT_MAPLE_1_1.PROMPT_DEPTH": 3,
            "TRAINER.KDPT_MAPLE_1_1.N_CTX": 2,
            "TRAINER.W_T": 4.0,
            "TRAINER.W_I": 1.0,
        },
        "StanfordCars": {
            "TRAINER.KDPT_MAPLE_1_1.PROMPT_DEPTH": 3,
            "TRAINER.KDPT_MAPLE_1_1.N_CTX": 2,
            "TRAINER.W_T": 4.0,
            "TRAINER.W_I": 1.0,
        },
        "OxfordFlowers": {
            "TRAINER.KDPT_MAPLE_1_1.PROMPT_DEPTH": 3,
            "TRAINER.KDPT_MAPLE_1_1.N_CTX": 2,
            "TRAINER.W_T": 4.0,
            "TRAINER.W_I": 1.0,
        },
        "Food101": {
            "TRAINER.KDPT_MAPLE_1_1.PROMPT_DEPTH": 3,
            "TRAINER.KDPT_MAPLE_1_1.N_CTX": 2,
            "TRAINER.W_T": 4.0,
            "TRAINER.W_I": 1.0,
        },
        "FGVCAircraft": {
            "TRAINER.KDPT_MAPLE_1_1.PROMPT_DEPTH": 3,
            "TRAINER.KDPT_MAPLE_1_1.N_CTX": 2,
            #"TRAINER.KDPT_MAPLE_1_1.N": 4,
            "TRAINER.W_T": 4.0,
            "TRAINER.W_I": 1.0,
            #"TRAINER.COS": False,
            #"TRAINER.KDPT_MAPLE_1_1.LAMDA": 0.0,
            #"DATALOADER.TRAIN_X.BATCH_SIZE": 4,
            #"OPTIM.MAX_EPOCH": 8,
        },
        "SUN397": {
            "TRAINER.KDPT_MAPLE_1_1.PROMPT_DEPTH": 3,
            "TRAINER.KDPT_MAPLE_1_1.N_CTX": 2,
        },
        "DescribableTextures": {
            "TRAINER.KDPT_MAPLE_1_1.PROMPT_DEPTH": 3,
            "TRAINER.KDPT_MAPLE_1_1.N_CTX": 2,
        },
        "EuroSAT": {
            "TRAINER.KDPT_MAPLE_1_1.PROMPT_DEPTH": 3,
            "TRAINER.KDPT_MAPLE_1_1.N_CTX": 2,
        },
        "UCF101": {
            "TRAINER.KDPT_MAPLE_1_1.PROMPT_DEPTH": 3,
            "TRAINER.KDPT_MAPLE_1_1.N_CTX": 2,
        },
    }.get(dataset, {})

    return " ".join([f"{k} {v}" for k, v in cfg.items()]).split(" ")
