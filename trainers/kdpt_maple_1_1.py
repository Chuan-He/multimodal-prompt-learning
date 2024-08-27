import os.path as osp

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.metrics import compute_accuracy
from dassl.utils import load_pretrained_weights, load_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler

from clip import clip
import copy
from trainers.imagenet_templates import *
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

import numpy as np
from sklearn.manifold import TSNE
from numpy import reshape
import pandas as pd
from trainers.visualization import compute_prototype
import random
from trainers.maml import *

_tokenizer = _Tokenizer()

def load_clip_to_cpu_0(cfg):
    backbone_name = cfg.MODEL.BACKBONE.NAME
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")
    design_details = {"trainer": 'CoOp',
                    "vision_depth": 0,
                    "language_depth": 0, "vision_ctx": 0,
                    "language_ctx": 0,
                    "maple_length": 0}
    model = clip.build_model(state_dict or model.state_dict(), design_details)

    return model

dataset_name_mapping = {
    "Caltech101": "caltech",
    "DescribableTextures": "dtd",
    "EuroSAT": "eurosat",
    "FGVCAircraft": "fgvc",
    "Food101": "food101",
    "ImageNet": "imagenet",
    "ImageNetA": "imagenet_a",
    "ImageNetR": "imagenet_r",
    "ImageNetSketch": "imagenet_sketch",
    "ImageNetV2": "imagenetv2",
    "OxfordFlowers": "oxford_flowers",
    "OxfordPets": "oxford_pets",
    "StanfordCars": "stanford_cars",
    "SUN397": "sun397",
    "UCF101": "ucf101",
}

CUSTOM_TEMPLATES1 = {
    "OxfordPets": "a photo of a {}, a type of pet.",
    "OxfordFlowers": "a photo of a {}, a type of flower.",
    #"OxfordFlowers": "a photo of a flower {}",
    "FGVCAircraft": "a photo of a {}, a type of aircraft.",
    #"FGVCAircraft": "a photo of a {}.",
    "DescribableTextures": "a photo of a {}, a type of texture.",
    "EuroSAT": "a centered satellite photo of {}.",
    #"EuroSAT": "a photo of a {}.",
    "StanfordCars": "a photo of a {}.",
    "Food101": "a photo of {}, a type of food.",
    "SUN397": "a photo of a {}.",
    "Caltech101": "a photo of a {}.",
    "UCF101": "a photo of a person doing {}.",
    "ImageNet": "a photo of a {}.",
    "ImageNetSketch": "a photo of a {}.",
    "ImageNetV2": "a photo of a {}.",
    "ImageNetA": "a photo of a {}.",
    "ImageNetR": "a photo of a {}."
}

CUSTOM_TEMPLATES = {
    "OxfordPets": "a photo of a {}.",
    "OxfordFlowers": "a photo of a {}.",
    "FGVCAircraft": "a photo of a {}.",
    "DescribableTextures": "a photo of a {}.",
    "EuroSAT": "a photo of a {}.",
    "StanfordCars": "a photo of a {}.",
    "Food101": "a photo of {}.",
    "SUN397": "a photo of a {}.",
    "Caltech101": "a photo of a {}.",
    "UCF101": "a photo of a {}.",
    "ImageNet": "a photo of a {}.",
    "ImageNetSketch": "a photo of a {}.",
    "ImageNetV2": "a photo of a {}.",
    "ImageNetA": "a photo of a {}.",
    "ImageNetR": "a photo of a {}."
}

template_dict = {   #'Caltech101': ["a photo of a {}.","this is a photo {}.","this is picture of {}.","one picture of a {}.", "a picture of a {}.","a painting of a {}.","a photo of the {}.","a painting of the {}."],
                    'Caltech101': ['a photo of a {}.','a painting of a {}.','a plastic {}.','a sculpture of a {}.','a sketch of a {}.','a tattoo of a {}.','a toy {}.','a rendition of a {}.'],
                    #'DescribableTextures':['a photo of a texture {}.', "this is a photo texture {}.","this is a picture texture {}.","one picture of a texture {}.",
                    #                       'a photo of a texture', "a photo of a pattern","a photo of a thing","a photo of a object"],
                    'DescribableTextures':['a photo of a {} texture.','a photo of a {} pattern.','a photo of a {} thing.','a photo of a {} object.',
                                           'a photo of the {} texture.','a photo of the {} pattern.','a photo of the {} thing.','a photo of the {} object.',],
                    'EuroSAT':['a centered satellite photo of {}.', 'a centered satellite picture of {}.','one centered satellite photo of a {}.','a satellite photo of a {}.',
                               'one centered satellite picture of {}.','a centered satellite photo of a {}.','this is centered satellite photo of {}.','a centered satellite photo of the {}.'], 
                    'FGVCAircraft':['this is aircraft picture of {}.','one picture of the aircraft {}.','a photo of the aircraft {}.','a photo of a {}.','a photo of the {}.'
                                    'a photo of a {}, a type of aircraft.', 'a photo of the {}, a type of aircraft.', 'this is aircraft photo of {}.'],
                    'Food101':['a photo of a {}.', 'this is a photo of {}.', 'this is a picture of {}.','one picture of a {}.',
                               'a photo of {}, a type of food.','a photo of the food {}.', 'this is photo of {}, a type of food.', 'this is picture of {}, a type of food.'],
                    #'ImageNet':["a photo of a","this is a photo", "itap of a", "a origami"],
                    'ImageNet':["itap of a {}.", "a bad photo of the {}.", "a origami {}.", "a photo of the large {}.", 
                                "a {} in a video game.", "art of the {}.", "a photo of the small {}.", "a photo of a {}."],
                    'ImageNetSketch':["itap of a {}.", "a bad photo of the {}.", "a origami {}.", "a photo of the large {}.", 
                                "a {} in a video game.", "art of the {}.", "a photo of the small {}.", "a photo of a {}."],
                    'ImageNetV2':["itap of a {}.", "a bad photo of the {}.", "a origami {}.", "a photo of the large {}.", 
                                "a {} in a video game.", "art of the {}.", "a photo of the small {}.", "a photo of a {}."],
                    'ImageNetA':["itap of a {}.", "a bad photo of the {}.", "a origami {}.", "a photo of the large {}.", 
                                "a {} in a video game.", "art of the {}.", "a photo of the small {}.", "a photo of a {}."],
                    'ImageNetR':["itap of a {}.", "a bad photo of the {}.", "a origami {}.", "a photo of the large {}.", 
                                "a {} in a video game.", "art of the {}.", "a photo of the small {}.", "a photo of a {}."],
                    #'OxfordFlowers':['a photo of a flower', 'one picture of a flower','this is flower picture of','one picture of a flower'],
                    'OxfordFlowers':['a photo of a {}.', 'one picture of flower {}.',' a photo of flower {}.','this is photo of {}, a type of flower.',
                                     'a photo of a {}, a type of flower.','a picture of a {}, a type of flower.','a photo of the {}, a type of flower.','this is picture of {}, a type of flower.'],
                    #'OxfordPets':['a photo of a pet', 'one picture of a pet','this is pet picture of','one picture of a pet'],
                    'OxfordPets':['a photo of a {}.','a photo of the {}.','one picture of a {}.', 'this is photo of {}.',
                                  'a photo of a {}, a type of pet.','a photo of the {}, a type of pet.','one picture of a {}, a type of pet.','this is photo of {}, a type of pet.'],
                    #'StanfordCars':["a photo of a","this is a photo ","this is picture of","one picture of a"],
                    'StanfordCars':['a photo of a {}.', 'a photo of the {}.', 'a photo of my {}.', 'i love my {}!', 
                                    'a photo of my dirty {}.', 'a photo of my clean {}.', 'a photo of my new {}.', 'a photo of my old {}.',],
                    'SUN397':["a photo of a {}.","this is photo of {}.","this is picture of {}.","one picture of a {}.",
                              'a photo of the {}.',"this is a photo of {}.","this is a picture of {}.","one picture of the {}."],
                    #'UCF101':['a photo of a person doing', 'this is a photo people doing', 'this is picture of people doing', 'one picture of a person doing', 'a picture of a person doing', 'this is a photo of people doing'],
                    'UCF101':['a photo of a person {}.', 'a video of a person {}.', 'a example of a person {}.', 'a demonstration of a person {}.',
                                'a photo of the person {}.', 'a video of the person {}.', 'a example of the person {}.', 'a demonstration of the person {}.'],}

def load_clip_to_cpu(cfg):
    backbone_name = cfg.MODEL.BACKBONE.NAME
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")
    design_details = {"trainer": 'KDPT_MAPLE_1_1',
                      "vision_depth": 0,
                      "language_depth": 0, "vision_ctx": 0,
                      "language_ctx": 0,
                      "maple_length": cfg.TRAINER.KDPT_MAPLE_1_1.N_CTX}
    model = clip.build_model(state_dict or model.state_dict(), design_details)

    return model


class TextEncoder_MMP(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts, compound_prompts_deeper_text):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        # Pass as the list, as nn.sequential cannot process multiple arguments in the forward pass
        combined = [x, compound_prompts_deeper_text, 0]  # third argument is the counter which denotes depth of prompt
        outputs = self.transformer(combined)
        x = outputs[0]  # extract the x back from here
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)
        text_features = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection
        return text_features


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


# class Adapter(nn.Module):
#     def __init__(self, c_in, reduction=4):
#         super(Adapter, self).__init__()
#         self.fc = nn.Sequential(
#             nn.Linear(c_in, c_in // reduction, bias=False),
#             nn.ReLU(inplace=True),
#             nn.Linear(c_in // reduction, c_in, bias=False),
#             nn.ReLU(inplace=True),
#         )
#         # self.fc = nn.Linear(c_in, c_in, bias=True)

#     def forward(self, x):
#         x = self.fc(x)
#         return x
    
class MultiModalPromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = cfg.TRAINER.KDPT_MAPLE_1_1.N_CTX
        ctx_init = cfg.TRAINER.KDPT_MAPLE_1_1.CTX_INIT
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg.INPUT.SIZE[0]
        N = cfg.TRAINER.KDPT_MAPLE_1_1.N   # the number of our text prompts 
        self.N = N

        self.compound_prompts_depth = cfg.TRAINER.KDPT_MAPLE_1_1.PROMPT_DEPTH  # max=12, but will create 11 such shared prompts

        ctx_vectors = torch.empty(self.N, n_ctx, ctx_dim, dtype=dtype)
        nn.init.normal_(ctx_vectors, std=0.02)
        prompt_prefix_list = [" ".join(["X"] * n_ctx) for _ in range(self.N)]

        if ctx_init and (n_ctx) <= 4:
            # use given words to initialize context vectors
            ctx_init_sent = [ctx_init for _ in range(self.N)]
            for i, ctx_init in enumerate(ctx_init_sent):
                ctx_init = ctx_init.replace("_", " ")
                n_ctx = n_ctx
                prompt = clip.tokenize(ctx_init)
                with torch.no_grad():
                    embedding = clip_model.token_embedding(prompt).type(dtype)
                ctx_vectors[i] = embedding[0, 1: 1 + n_ctx, :]
                prompt_prefix_list[i] = ctx_init
        
        self.ctx = nn.Parameter(ctx_vectors) # parameters of text prompt to be learned
        self.proj = nn.Linear(ctx_dim, 768)
        self.proj.half()

        print('KDPT_MAPLE_1_1 design: Multi-modal Prompt Learning')
        print(f'Layers: {self.compound_prompts_depth} N: {cfg.TRAINER.KDPT_MAPLE_1_1.N} N_CTX: {n_ctx}')
        self.compound_prompts_text = nn.ParameterList([nn.Parameter(torch.empty(self.N, n_ctx, 512))
                                                      for _ in range(self.compound_prompts_depth - 1)])
        for single_para in self.compound_prompts_text:
            nn.init.normal_(single_para, std=0.02)
        self.single_layer = nn.Linear(ctx_dim, 768)
        self.compound_prompt_projections = _get_clones(self.single_layer, self.compound_prompts_depth-1)
        '''
        if ctx_init_flag:
            ctx_list = template_dict[cfg.DATASET.NAME]
            ctx_vectors_list = []
            prompt_prefix_list = []
            
            for i in range(N):
                ctx_init = ctx_list[i].replace("_", " ")
                prompt = clip.tokenize(ctx_init)
                with torch.no_grad():
                    embedding = clip_model.token_embedding(prompt).type(dtype)
                ctx_vectors = embedding[0, 1 : 1 + n_ctx, :]
                ctx_vectors_list.append(ctx_vectors)
                prompt_prefix = ctx_init
                prompt_prefix_list.append(prompt_prefix)
          
            ctx_vectors = torch.stack(ctx_vectors_list) # N, n_ctx, ctx_dim
        '''
        #else:
        #    ctx_vectors = torch.empty(N, n_ctx, ctx_dim, dtype=dtype)

        #    nn.init.normal_(ctx_vectors, std=0.02)

        #    prompt_prefix = " ".join(["X"] * n_ctx)
        #self.w = nn.Parameter(torch.zeros(1, ctx_dim, device='cuda', dtype=dtype), requires_grad=True)

        
        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [each_prompt + " " + name + "." for name in classnames for each_prompt in prompt_prefix_list]
        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])  # (n_cls * number_prompts, n_tkn)
        '''
        prompt_list = []
        if ctx_init:
            for i in range(N):
                prompt_prefix = prompt_prefix_list[i]
                #prompts = [prompt_prefix + " " + name + "." for name in classnames] # 100
                prompts = [prompt_prefix.format(name.replace("_", " ")) for name in classnames] # 100
                tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts]) # 100x77
                prompt_list.append(tokenized_prompts)
            tokenized_prompts = torch.cat(prompt_list)
        '''
        #else:
        #    tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])  # (n_cls, n_tkn)
        #    tokenized_prompts = tokenized_prompts.repeat(N,1)
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)
        
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx:, :])  # CLS, EOS
        
        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens
        
        # zsclip text features
        with torch.no_grad():
            clip_model_image_0 = load_clip_to_cpu_0(cfg)
            clip_model_image_0.to('cuda')
            self.zs_image_encoder = clip_model_image_0.visual
            clip_model_text_0 = clip_model_image_0
            clip_model_text_0.to('cuda')
            #prompts_ = [prompt_prefix + " " + name + "." for name in classnames]        
            # temp = CUSTOM_TEMPLATES[cfg.DATASET.NAME]
            # prompts_ = [temp.format(c.replace("_", " ")) for c in classnames]
            # print(f"ZSCLIP Prompts: {prompts_}")
            # prompts_ = torch.cat([clip.tokenize(p) for p in prompts_])
            # prompts_ = prompts_.to('cuda')

            # text_features_ori = clip_model_0.encode_text(prompts_)
            # text_features_ori = text_features_ori / text_features_ori.norm(dim=-1, keepdim=True)

        # self.text_features_ori = text_features_ori
        all_teacher_features = []
        with torch.no_grad():
            for single_template in IMAGENET_TEMPLATES:
                x = [single_template.replace("{}", name) for name in classnames]
                x_tokenized = torch.cat([clip.tokenize(p) for p in x])
                text_features = clip_model_text_0.encode_text(x_tokenized.cuda())
                all_teacher_features.append(text_features.unsqueeze(1))

        self.text_features_ori = torch.cat(all_teacher_features, dim=1).mean(dim=1)
    
    def construct_prompts(self, ctx, prefix, suffix, label=None):
        # dim0 is either batch_size (during training) or n_cls (during testing)
        # ctx: context tokens, with shape of (dim0, n_ctx, ctx_dim)
        # prefix: the sos token, with shape of (n_cls, 1, ctx_dim)
        # suffix: remaining tokens, with shape of (n_cls, *, ctx_dim)

        # if label is not None:
        #     prefix = prefix[label]
        #     suffix = suffix[label]
        if ctx.dim() == 3:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1,-1)
        #ctx = ctx.permute(1, 0, 2, 3) #  N 100 16 512

        ctx = ctx.contiguous().view(self.N*self.n_cls,self.n_ctx,ctx.shape[3])
        prompts = torch.cat(
            [
                prefix,  # (dim0, 1, dim)
                ctx,  # (dim0, n_ctx, dim)
                suffix,  # (dim0, *, dim)
            ],
            dim=1,
        )
        return prompts
    
    def forward(self):

        ctx = self.ctx
        if ctx.dim() == 3:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1, -1)    #### (n_cls, number_prompts, prompts_len, prompts_dim)
        ctx = ctx.contiguous().view(self.n_cls*self.N, self.n_ctx, ctx.shape[3])  #### n_cls, number_prompts, n_ctx, dim

        prefix = self.token_prefix       ### n_cls * number_prompts, x, dim
        suffix = self.token_suffix       ### n_cls * number_prompts, x, dim
        prompts = torch.cat(
            [
                prefix,    ## (n_cls, number_prompts, 1, dim)
                ctx,       ## (n_cls, number_prompts, n_ctx, dim)
                suffix,    ## (n_cls, number_prompots, *, dim)
            ],
            dim=1,
        )

        visual_deep_prompts = []
        for index, layer in enumerate(self.compound_prompt_projections):
            visual_deep_prompts.append(layer(self.compound_prompts_text[index]))

        return prompts, self.proj(self.ctx), self.compound_prompts_text, visual_deep_prompts


class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.prompt_learner = MultiModalPromptLearner(cfg, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder_MMP(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        # self.image_adapter = Adapter(512).half()
        # self.text_adapter = Adapter(512, 4).half()

        self.vision_prompt_number = self.text_prompt_number = self.N = cfg.TRAINER.KDPT_MAPLE_1_1.N #+ len(clip_template[cfg.DATASET.NAME])
        # self.lamda = cfg.TRAINER.KDPT_MAPLE_1_1.LAMDA
        self.n_cls = len(classnames)

        self.ori_embedding = self.prompt_learner.text_features_ori

    def forward(self, image):
        batch_size, img_c, img_h, img_w = image.shape
        prompts, shared_ctx, deeper_text_prompts, deeper_vision_prompts = self.prompt_learner()
        tokenized_prompts = self.tokenized_prompts
        logit_scale = self.logit_scale.exp()

        text_features = self.text_encoder(prompts, tokenized_prompts, deeper_text_prompts)
        image_1 = image.unsqueeze(1).expand(-1, self.N, -1, -1, -1).contiguous().view(-1, img_c, img_h, img_w)
        shared_ctx = shared_ctx.unsqueeze(0).expand(batch_size, -1, -1, -1)
        shared_ctx = shared_ctx.contiguous().view(image_1.shape[0], shared_ctx.shape[-2], shared_ctx.shape[-1])
        image_features = self.image_encoder(image_1.type(self.dtype), shared_ctx, deeper_vision_prompts)

        vis_dim = image_features.shape[-1]
        ctx_dim = text_features.shape[-1]
        text_features = text_features.contiguous().view(self.prompt_learner.n_cls, self.text_prompt_number, ctx_dim)  #### (text_prompt_number, n_cls, dim)
        
        image_features = image_features.contiguous().view(batch_size, self.vision_prompt_number, vis_dim)  #### (vision_prompt_number * batch_size, 199, dim)

        image_feature_pool = image_features.mean(dim=1)
        text_feature_pool = text_features.mean(dim=1)

        image_feature_pool = image_feature_pool / image_feature_pool.norm(dim=-1, keepdim=True)
        text_feature_pool = text_feature_pool / text_feature_pool.norm(dim=-1, keepdim=True)
        logits = logit_scale * image_feature_pool @ text_feature_pool.t()

        if self.prompt_learner.training:
            with torch.no_grad():
                text_features_ori = self.ori_embedding
                #text_features_ori = text_features_ori.unsqueeze(1).expand(-1, self.N, -1)
                text_features_ori = text_features_ori / text_features_ori.norm(dim=-1, keepdim=True)
                image_features_ori = self.prompt_learner.zs_image_encoder(image.type(self.dtype))
                #image_features_ori = image_features_ori.unsqueeze(1).expand(-1, self.N, -1)
                image_features_ori = image_features_ori / image_features_ori.norm(dim=-1, keepdim=True)
                #logits_ref = logit_scale * image_features_ori @ text_features_ori.t()

            loss_scl_text = F.l1_loss(text_feature_pool, text_features_ori.cuda(), reduction='mean')
            loss_scl_image = F.l1_loss(image_feature_pool, image_features_ori.cuda(), reduction='mean')

            return logits, loss_scl_text, loss_scl_image

        return logits
    
    def forward_mix(self, image, label):
        tokenized_prompts = self.tokenized_prompts
        logit_scale = self.logit_scale.exp()

        # generate mixed sample
        self.alpha = 0.1
        lam = np.random.beta(self.alpha, self.alpha)

        batch_size = image.size()[0]
        index = torch.randperm(batch_size).cuda()
        target_a = label
        target_b = label[index]
        mixed_image = lam * image + (1 - lam) * image[index,:]

        batch_size, img_c, img_h, img_w = mixed_image.shape
        prompts, shared_ctx, deeper_text_prompts, deeper_vision_prompts = self.prompt_learner()
        tokenized_prompts = self.tokenized_prompts
        logit_scale = self.logit_scale.exp()

        text_features = self.text_encoder(prompts, tokenized_prompts, deeper_text_prompts)
        image_1 = mixed_image.unsqueeze(1).expand(-1, self.N, -1, -1, -1).contiguous().view(-1, img_c, img_h, img_w)
        shared_ctx = shared_ctx.unsqueeze(0).expand(batch_size, -1, -1, -1)
        shared_ctx = shared_ctx.contiguous().view(image_1.shape[0], shared_ctx.shape[-2], shared_ctx.shape[-1])
        image_features = self.image_encoder(image_1.type(self.dtype), shared_ctx, deeper_vision_prompts)

        vis_dim = image_features.shape[-1]
        ctx_dim = text_features.shape[-1]
        text_features = text_features.contiguous().view(self.prompt_learner.n_cls, self.text_prompt_number, ctx_dim)  #### (text_prompt_number, n_cls, dim)
        image_features = image_features.contiguous().view(batch_size, self.vision_prompt_number, vis_dim)  #### (vision_prompt_number * batch_size, 199, dim)

        image_feature_pool = image_features.mean(dim=1)
        text_feature_pool = text_features.mean(dim=1)

        image_feature_pool = image_feature_pool / image_feature_pool.norm(dim=-1, keepdim=True)
        text_feature_pool = text_feature_pool / text_feature_pool.norm(dim=-1, keepdim=True)
        logits = logit_scale * image_feature_pool @ text_feature_pool.t()

        # Now calculate the frozen pre-trained features
        fixed_embeddings = self.ori_embedding  # precomputed pre-trained frozen textual features
        fixed_embeddings = fixed_embeddings / fixed_embeddings.norm(dim=-1, keepdim=True)
        with torch.no_grad():
            zero_shot_features = self.prompt_learner.zs_image_encoder(mixed_image.type(self.dtype))
            zero_shot_features = zero_shot_features / zero_shot_features.norm(dim=-1, keepdim=True)

        # compute output
        ce_loss = F.cross_entropy(logits, target_a) * lam + F.cross_entropy(logits, target_b) * (1. - lam)

        reg_text = F.l1_loss(text_feature_pool, fixed_embeddings, reduction='mean')
        reg_image = F.l1_loss(image_feature_pool, zero_shot_features, reduction='mean')

        return ce_loss, reg_text, reg_image

@TRAINER_REGISTRY.register()
class KDPT_MAPLE_1_1(TrainerX):
    """
    It is based on PLOT.
    """
    
    def check_cfg(self, cfg):
        assert cfg.TRAINER.KDPT_MAPLE_1_1.PREC in ["fp16", "fp32", "amp"]
    
    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames
        #all_classnames = self.dm.dataset.all_class_names
        self.dataset =  cfg.DATASET.NAME
        
        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)
        
        print("Building custom CLIP")
        self.model = CustomCLIP(cfg, classnames, clip_model)
        self.vnet = VNet(cfg, self.device)
        self.w_t = cfg.TRAINER.W_T
        self.w_i = cfg.TRAINER.W_I

        print("Turning off gradients in both the image and the text encoder")
        name_to_update = "prompt_learner"

        for name, param in self.model.named_parameters():
            if name_to_update not in name:
                param.requires_grad_(False)
            if 'zs_image_encoder' in name or 'clip_model_text_0' in name:
                param.requires_grad_(False)

     # Double check
        enabled = set()
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                enabled.add(name)
        print(f"Parameters to be updated: {enabled}")
        
        # if cfg.MODEL.INIT_WEIGHTS:
        #     load_pretrained_weights(self.model.prompt_learner, cfg.MODEL.INIT_WEIGHTS)
        
        self.model.to(self.device)
        # NOTE: we give whole model to the optimizer, but only prompt_learner will be optimized
        self.optim = build_optimizer(self.model, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.vnet_optim = build_optimizer(self.vnet, cfg.OPTIM_VNET)
        self.vnet_sched = build_lr_scheduler(self.vnet_optim, cfg.OPTIM_VNET)
        self.register_model("prompt_learner", self.model, self.optim, self.sched)
        # torch.autograd.set_detect_anomaly(True)

    
    def forward_backward(self, batch):
        image, label = self.parse_batch_train(batch)

        # logits, reg_text, reg_image = self.model(image)
        # loss_ce = F.cross_entropy(logits, label)
        
        # self.optim.zero_grad()
        # model = gradient_update(model, loss_ce, self.w_t*reg_text, self.w_i*reg_image, grad_func=self.vnet)
        # self.optim.step()

        self.adapt_lr = 0.002
        self.fast_adaptation = False
        maml = MAML(self.model, lr=self.adapt_lr, first_order=self.fast_adaptation)
        task_model = maml.clone(allow_nograd=True)
        logits, reg_text, reg_image = task_model(image)
        loss_ce = F.cross_entropy(logits, label)

        task_model.adapt(loss_ce, self.w_t*reg_text, self.w_i*reg_image, allow_nograd=True, grad_func=self.vnet, allow_unused=True)
        ce_loss, reg_text, reg_image = task_model.forward_mix(image, label)
        loss_ce = ce_loss

        self.optim.zero_grad()
        self.vnet_optim.zero_grad()
        loss_ce.backward()
        self.optim.step()
        self.vnet_optim.step()

        loss_summary = {"loss": loss_ce.item(),
                         "acc": compute_accuracy(logits, label)[0].item()}
        
        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()
            
        return loss_summary
    
    def parse_batch_train(self, batch):
        input = batch["img"]
        label = batch["label"]
        input = input.to(self.device)
        label = label.to(self.device)
        return input, label
    
    # def model_inference(self, input):
    #     output = self.model(input)
    #     with torch.no_grad():
    #         compute_prototype(self.model, self.dm.test_loader)
    #     return output #self.model(input)
    
    def load_model(self, directory, epoch=None):
        if not directory:
            print("Note that load_model() is skipped as no pretrained model is given")
            return

        names = self.get_model_names()
        print(names)

        # By default, the best model is loaded
        model_file = "model.pth.tar" + str(epoch)

        if epoch is not None:
            model_file = "model.pth.tar-" + str(epoch)

        for name in names:
            model_path = osp.join(directory, name, model_file)

            if not osp.exists(model_path):
                raise FileNotFoundError('Model not found at "{}"'.format(model_path))

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]
            epoch = checkpoint["epoch"]

            # Ignore fixed token vectors
            if "prompt_learner.token_prefix" in state_dict:
                del state_dict["prompt_learner.token_prefix"]

            if "prompt_learner.token_suffix" in state_dict:
                del state_dict["prompt_learner.token_suffix"]

            print("Loading weights to {} " 'from "{}" (epoch = {})'.format(name, model_path, epoch))
            self._models[name].load_state_dict(state_dict, strict=False)
