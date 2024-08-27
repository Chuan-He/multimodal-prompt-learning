
def get_dataset_specified_config_2(dataset):
    """Get dataset specific."""
    cfg = {
        "ImageNet": {
            "TRAINER.KDPT_MAPLE_2.PROMPT_DEPTH": 9,
            "TRAINER.KDPT_MAPLE_2.N_CTX": 4,
            "TRAINER.W": 4.0,
            "TRAINER.W1": 0.0,
            "TRAINER.W2": 4.0,
            "OPTIM.MAX_EPOCH": 5,
        },
        # "ImageNetSketch": {
        #     "TRAINER.KDPT_MAPLE_2.PROMPT_DEPTH": 3,
        #     "TRAINER.KDPT_MAPLE_2.N_CTX": 2,
        #     "TRAINER.W": 1.0,
        #     #"OPTIM.MAX_EPOCH": 5,
        # },
        # "ImageNetV2": {
        #     "TRAINER.KDPT_MAPLE_2.PROMPT_DEPTH": 3,
        #     "TRAINER.KDPT_MAPLE_2.N_CTX": 2,
        #     "TRAINER.W": 1.0,
        #     #"OPTIM.MAX_EPOCH": 5,
        # },
        # "ImageNetA": {
        #     "TRAINER.KDPT_MAPLE_2.PROMPT_DEPTH": 3,
        #     "TRAINER.KDPT_MAPLE_2.N_CTX": 2,
        #     "TRAINER.W": 1.0,
        #     #"OPTIM.MAX_EPOCH": 5,
        # },
        # "ImageNetR": {
        #     "TRAINER.KDPT_MAPLE_2.PROMPT_DEPTH": 3,
        #     "TRAINER.KDPT_MAPLE_2.N_CTX": 2,
        #     "TRAINER.W": 1.0,
        #     #"OPTIM.MAX_EPOCH": 5,
        # },
        "Caltech101": {
            "TRAINER.KDPT_MAPLE_2.PROMPT_DEPTH": 9,
            "TRAINER.KDPT_MAPLE_2.N_CTX": 4,
            "TRAINER.W": 4.0,
            "TRAINER.W1": 0.0,
            "TRAINER.W2": 4.0,
            "OPTIM.MAX_EPOCH": 8,
        },
        "OxfordPets": {
            "TRAINER.KDPT_MAPLE_2.PROMPT_DEPTH": 9,
            "TRAINER.KDPT_MAPLE_2.N_CTX": 4,
            "TRAINER.W": 2.0,
            "TRAINER.W1": 0.0,
            "TRAINER.W2": 4.0,
            "OPTIM.MAX_EPOCH": 8,
        },
        "StanfordCars": {
            "TRAINER.KDPT_MAPLE_2.PROMPT_DEPTH": 9,
            "TRAINER.KDPT_MAPLE_2.N_CTX": 4,
            "TRAINER.W": 2.0,
            "TRAINER.W1": 0.0,
            "TRAINER.W2": 4.0,
            "OPTIM.MAX_EPOCH": 8,
        },
        "OxfordFlowers": {
            "TRAINER.KDPT_MAPLE_2.PROMPT_DEPTH": 9,
            "TRAINER.KDPT_MAPLE_2.N_CTX": 4,
            "TRAINER.W": 8.0,
            "TRAINER.W1": 8.0,
            "TRAINER.W2": 8.0,
            "OPTIM.MAX_EPOCH": 8,
        },
        "Food101": {
            "TRAINER.KDPT_MAPLE_2.PROMPT_DEPTH": 9,
            "TRAINER.KDPT_MAPLE_2.N_CTX": 4,
            "TRAINER.W": 4.0,
            "TRAINER.W1": 0.0,
            "TRAINER.W2": 4.0,
            "OPTIM.MAX_EPOCH": 8,
        },
        "FGVCAircraft": {
            "TRAINER.KDPT_MAPLE_2.PROMPT_DEPTH": 9,
            "TRAINER.KDPT_MAPLE_2.N_CTX": 4,
            "TRAINER.W": 4.0,    # cos_text
            "TRAINER.W1": 4.0,   # scl_text
            "TRAINER.W2": 4.0,   # scl_image
            "TRAINER.W3": 4.0,   # cos_image
            "OPTIM.MAX_EPOCH": 8,
        },
        "SUN397": {
            "TRAINER.KDPT_MAPLE_2.PROMPT_DEPTH": 9,
            "TRAINER.KDPT_MAPLE_2.N_CTX": 4,
            "TRAINER.W": 4.0,
            "TRAINER.W1": 0.0,
            "TRAINER.W2": 4.0,
            "OPTIM.MAX_EPOCH": 8,
        },
        "DescribableTextures": {
            "TRAINER.KDPT_MAPLE_2.PROMPT_DEPTH": 9,
            "TRAINER.KDPT_MAPLE_2.N_CTX": 4,
            "TRAINER.W": 4.0,
            "TRAINER.W1": 2.0,
            "TRAINER.W2": 4.0,
            #"OPTIM.MAX_EPOCH": 8,
        },
        "EuroSAT": {
            "TRAINER.KDPT_MAPLE_2.PROMPT_DEPTH": 9,
            "TRAINER.KDPT_MAPLE_2.N_CTX": 2,
            "TRAINER.W": 8.0,    # cos_text
            "TRAINER.W1": 0.0,   # scl_text
            "TRAINER.W2": 4.0,   # scl_image
            "TRAINER.W3": 1.0,   # cos_image
            "OPTIM.MAX_EPOCH": 8,
        },
        "UCF101": {
            "TRAINER.KDPT_MAPLE_2.PROMPT_DEPTH": 9,
            "TRAINER.KDPT_MAPLE_2.N_CTX": 4,
            "TRAINER.W": 4.0,
            "TRAINER.W1": 0.0,
            "TRAINER.W2": 4.0,
            "OPTIM.MAX_EPOCH": 8,
        },
    }.get(dataset, {})

    return " ".join([f"{k} {v}" for k, v in cfg.items()]).split(" ")
