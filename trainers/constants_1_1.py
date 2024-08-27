
def get_dataset_specified_config_1_1(dataset):
    """Get dataset specific."""
    cfg = {
        "ImageNet": {
            "TRAINER.KDPT_MAPLE_1_1.PROMPT_DEPTH": 9,
            "TRAINER.KDPT_MAPLE_1_1.N_CTX": 2,
            "TRAINER.W_T": 8.0,
            "TRAINER.W_I": 2.0,
        },
        "Caltech101": {
            "TRAINER.KDPT_MAPLE_1_1.PROMPT_DEPTH": 9,
            "TRAINER.KDPT_MAPLE_1_1.N_CTX": 2,
            "TRAINER.W_T": 10.0,
            "TRAINER.W_I": 2.0,
        },
        "OxfordPets": {
            "TRAINER.KDPT_MAPLE_1_1.PROMPT_DEPTH": 9,
            "TRAINER.KDPT_MAPLE_1_1.N_CTX": 2,
            "TRAINER.W_T": 10.0,
            "TRAINER.W_I": 2.0,
        },
        "StanfordCars": {
            "TRAINER.KDPT_MAPLE_1_1.PROMPT_DEPTH": 9,
            "TRAINER.KDPT_MAPLE_1_1.N_CTX": 2,
            "TRAINER.W_T": 10.0,
            "TRAINER.W_I": 2.0,
            "TRAINER.KDPT_MAPLE_1_1.LAMDA": 0.0,
        },
        "OxfordFlowers": {
            "TRAINER.KDPT_MAPLE_1_1.PROMPT_DEPTH": 9,
            "TRAINER.KDPT_MAPLE_1_1.N_CTX": 2,
            "TRAINER.W_T": 10.0,
            "TRAINER.W_I": 2.0,
        },
        "Food101": {
            "TRAINER.KDPT_MAPLE_1_1.PROMPT_DEPTH": 9,
            "TRAINER.KDPT_MAPLE_1_1.N_CTX": 2,
            "TRAINER.W_T": 10.0,
            "TRAINER.W_I": 2.0,
        },
        "FGVCAircraft": {
            "TRAINER.KDPT_MAPLE_1_1.PROMPT_DEPTH": 9,
            "TRAINER.KDPT_MAPLE_1_1.N_CTX": 4,
            #"TRAINER.KDPT_MAPLE_1_1.N": 4,
            "TRAINER.W_T": 15.0,
            "TRAINER.W_I": 5.0,
            #"TRAINER.COS": False,
            #"TRAINER.KDPT_MAPLE_1_1.LAMDA": 0.0,
            #"DATALOADER.TRAIN_X.BATCH_SIZE": 4,
            #"OPTIM.MAX_EPOCH": 8,
        },
        "SUN397": {
            "TRAINER.KDPT_MAPLE_1_1.PROMPT_DEPTH": 9,
            "TRAINER.KDPT_MAPLE_1_1.N_CTX": 2,
            "TRAINER.W_T": 10.0,
            "TRAINER.W_I": 2.0,
        },
        "DescribableTextures": {
            "TRAINER.KDPT_MAPLE_1_1.PROMPT_DEPTH": 9,
            "TRAINER.KDPT_MAPLE_1_1.N_CTX": 2,
            "TRAINER.W_T": 10.0,
            "TRAINER.W_I": 2.0,
            "TRAINER.KDPT_MAPLE_1_1.LAMDA": 0.0,
        },
        "EuroSAT": {
            "TRAINER.KDPT_MAPLE_1_1.PROMPT_DEPTH": 9,
            "TRAINER.KDPT_MAPLE_1_1.N_CTX": 2,
            "TRAINER.W_T": 10.0,
            "TRAINER.W_I": 4.0,
        },
        "UCF101": {
            "TRAINER.KDPT_MAPLE_1_1.PROMPT_DEPTH": 9,
            "TRAINER.KDPT_MAPLE_1_1.N_CTX": 2,
            "TRAINER.W_T": 8.0,
            "TRAINER.W_I": 2.0,
            "TRAINER.KDPT_MAPLE_1_1.LAMDA": 0.0,
        },
    }.get(dataset, {})

    return " ".join([f"{k} {v}" for k, v in cfg.items()]).split(" ")
