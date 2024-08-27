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
import seaborn as sns
import pandas as pd

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

CUSTOM_TEMPLATES = {
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
    design_details = {"trainer": 'KDPT_MAPLE',
                      "vision_depth": 0,
                      "language_depth": 0, "vision_ctx": 0,
                      "language_ctx": 0,
                      "maple_length": cfg.TRAINER.KDPT_MAPLE.N_CTX}
    model = clip.build_model(state_dict or model.state_dict(), design_details)

    return model


class TextEncoder_MaPLe(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype
        self.N = 4

    def forward(self, prompts, tokenized_prompts, compound_prompts_deeper_text):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        # Pass as the list, as nn.sequential cannot process multiple arguments in the forward pass
        combined = [x, compound_prompts_deeper_text, 0]  # third argument is the counter which denotes depth of prompt
        outputs = self.transformer(combined)
        x = outputs[0]  # extract the x back from here
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        text_features = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection
        #### token_embeddings
        x = x.contiguous().view(x.shape[0] // self.N, self.N, x.shape[-2], x.shape[-1])
        tokenized_prompts = tokenized_prompts.contiguous().view(x.shape[0], self.N, -1)  ## todo: please check it
        normalized_xout = []
        for i in range(x.shape[0]):
            prompt_emb = x[i, :, :tokenized_prompts[i,0].argmax()-1]
            normalized_xout.append(prompt_emb / prompt_emb.norm(dim=-1, keepdim=True))

        return text_features, normalized_xout
    
class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        text_features = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        #### token_embeddings
        x = x.contiguous().view(x.shape[0] // self.number_text_prompts, self.number_text_prompts, x.shape[-2], x.shape[-1])
        tokenized_prompts = tokenized_prompts.contiguous().view(x.shape[0], self.number_text_prompts, -1)  ## todo: please check it
        normalized_xout = []
        for i in range(x.shape[0]):
            prompt_emb = x[i, :, :tokenized_prompts[i,0].argmax()-1]
            normalized_xout.append(prompt_emb / prompt_emb.norm(dim=-1, keepdim=True))

        return text_features, normalized_xout

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

def normalize(logit):
    mean = logit.mean(dim=-1, keepdims=True)
    stdv = logit.std(dim=-1, keepdims=True)
    return (logit - mean) / (1e-7 + stdv)

def kd_loss_norm(logits_student_in, logits_teacher_in, temperature, logit_stand=True):
    logits_student = normalize(logits_student_in) if logit_stand else logits_student_in
    logits_teacher = normalize(logits_teacher_in) if logit_stand else logits_teacher_in
    log_pred_student = F.log_softmax(logits_student / temperature, dim=1)
    pred_teacher = F.softmax(logits_teacher / temperature, dim=1)
    loss_kd = F.kl_div(log_pred_student, pred_teacher, reduction="none").sum(1).mean()
    loss_kd *= temperature**2
    return loss_kd

def kd_loss(logits_student, logits_teacher, temperature, reduce=True):
    log_pred_student = F.log_softmax(logits_student / temperature, dim=1)
    pred_teacher = F.softmax(logits_teacher / temperature, dim=1)
    if reduce:
        loss_kd = F.kl_div(log_pred_student, pred_teacher, reduction="none").sum(1).mean()
    else:
        loss_kd = F.kl_div(log_pred_student, pred_teacher, reduction="none").sum(1)
    loss_kd *= temperature**2
    return loss_kd

def cc_loss(logits_student, logits_teacher, temperature, reduce=True):
    batch_size, class_num = logits_teacher.shape
    pred_student = F.softmax(logits_student / temperature, dim=1)
    pred_teacher = F.softmax(logits_teacher / temperature, dim=1)
    student_matrix = torch.mm(pred_student.transpose(1, 0), pred_student)
    teacher_matrix = torch.mm(pred_teacher.transpose(1, 0), pred_teacher)
    if reduce:
        consistency_loss = ((teacher_matrix - student_matrix) ** 2).sum() / class_num
    else:
        consistency_loss = ((teacher_matrix - student_matrix) ** 2) / class_num
    return consistency_loss

def bc_loss(logits_student, logits_teacher, temperature, reduce=True):
    batch_size, class_num = logits_teacher.shape
    pred_student = F.softmax(logits_student / temperature, dim=1)
    pred_teacher = F.softmax(logits_teacher / temperature, dim=1)
    student_matrix = torch.mm(pred_student, pred_student.transpose(1, 0))
    teacher_matrix = torch.mm(pred_teacher, pred_teacher.transpose(1, 0))
    if reduce:
        consistency_loss = ((teacher_matrix - student_matrix) ** 2).sum() / batch_size
    else:
        consistency_loss = ((teacher_matrix - student_matrix) ** 2) / batch_size
    return consistency_loss

class MultiModalPromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = cfg.TRAINER.KDPT_MAPLE.N_CTX
        ctx_init = cfg.TRAINER.KDPT_MAPLE.CTX_INIT
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg.INPUT.SIZE[0]
        N = cfg.TRAINER.KDPT_MAPLE.N   # the number of our text prompts 
        #M = len(clip_template[cfg.DATASET.NAME])
        self.N = N #+ M

        self.compound_prompts_depth = cfg.TRAINER.KDPT_MAPLE.PROMPT_DEPTH  # max=12, but will create 11 such shared prompts

        ctx_vectors = torch.empty(self.N, n_ctx, ctx_dim, dtype=dtype)
        nn.init.normal_(ctx_vectors, std=0.02)
        prompt_prefix_list = [" ".join(["X"] * n_ctx) for _ in range(self.N)]

        if ctx_init and (n_ctx) <= 4:
            # use given words to initialize context vectors
            ctx_init_sent = ctx_init.split('\t')[:self.N]
            for i, ctx_init in enumerate(ctx_init_sent):
                ctx_init = ctx_init.replace("_", " ")
                n_ctx = n_ctx
                prompt = clip.tokenize(ctx_init)
                with torch.no_grad():
                    embedding = clip_model.token_embedding(prompt).type(dtype)
                ctx_vectors[i] = embedding[0, 1: 1 + n_ctx, :]
                prompt_prefix_list[i] = ctx_init  
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
        self.ctx = nn.Parameter(ctx_vectors) # parameters of text prompt to be learned
        self.proj = nn.Linear(ctx_dim, 768)
        self.proj.half()
        
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

        print('KDPT_MAPLE design: Multi-modal Prompt Learning')
        print(self.compound_prompts_depth)
        self.compound_prompts_text = nn.ParameterList([nn.Parameter(torch.empty(self.N, n_ctx, 512))
                                                      for _ in range(self.compound_prompts_depth - 1)])
        for single_para in self.compound_prompts_text:
            nn.init.normal_(single_para, std=0.02)
        self.single_layer = nn.Linear(ctx_dim, 768)
        self.compound_prompt_projections = _get_clones(self.single_layer, self.compound_prompts_depth-1)
        
        # zsclip text features
        clip_model_0 = load_clip_to_cpu_0(cfg)
        clip_model_0.to('cuda')
        self.clip_model_0 = clip_model_0
        #prompts_ = [prompt_prefix + " " + name + "." for name in classnames]        
        temp = CUSTOM_TEMPLATES[cfg.DATASET.NAME]
        prompts_ = [temp.format(c.replace("_", " ")) for c in classnames]
        print(f"ZSCLIP Prompts: {prompts_}")
        prompts_ = torch.cat([clip.tokenize(p) for p in prompts_])
        prompts_ = prompts_.to('cuda')

        with torch.no_grad():
            text_features_ori = clip_model_0.encode_text(prompts_)
            text_features_ori = text_features_ori / text_features_ori.norm(dim=-1, keepdim=True)

        self.text_features_ori = text_features_ori
    
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
        ctx = ctx.permute(1, 0, 2, 3) #  N 100 16 512

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

def ct_dis(cost, dis):
    """
    cost: batch_size, m, n
    dis: batch_size, m,n
    """
    forward_pi = torch.softmax(-dis, dim=-1)
    backward_pi = torch.softmax(-dis, dim=-2)
    forward_cost = (forward_pi * cost).sum(-1).mean(-1)
    backward_cost = (backward_pi * cost).sum(-2).mean(-1)
    return forward_cost, backward_cost

def Sinkhorn(K, u, v, max_iter=100):
    r = torch.ones_like(u, dtype=K.dtype, device=K.device)
    c = torch.ones_like(v, dtype=K.dtype, device=K.device)
    thresh = 1e-2
    T0 = torch.matmul(r.unsqueeze(-1), c.unsqueeze(-2)) * K
    for i in range(max_iter):
        r0 = r
        # print(i, K.shape, c.shape, u.shape, torch.isnan(K).any())
        r = u / torch.matmul(K, c.unsqueeze(-1)).squeeze(-1)
        c = v / torch.matmul(K.permute(0, 2, 1).contiguous(), r.unsqueeze(-1)).squeeze(-1)
        err = (r - r0).abs().mean()
        if err.item() < thresh:
            break

    T = torch.matmul(r.unsqueeze(-1), c.unsqueeze(-2)) * K

    if torch.isnan(T).any():
        return T0

    return T
    
class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.prompt_learner = MultiModalPromptLearner(cfg, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder_MaPLe(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype

        self.vision_prompt_number = self.text_prompt_number = self.N = cfg.TRAINER.KDPT_MAPLE.N #+ len(clip_template[cfg.DATASET.NAME])
        self.n_cls = len(classnames)

        self.ori_embedding = self.prompt_learner.text_features_ori

    def forward(self, image):
        batch_size, img_c, img_h, img_w = image.shape
        prompts, shared_ctx, deeper_text_prompts, deeper_vision_prompts = self.prompt_learner()
        tokenized_prompts = self.tokenized_prompts
        logit_scale = self.logit_scale.exp()
        
        #image_features_pool = []
        #for ctx in shared_ctx:
            #image_features_pool.append(self.image_encoder(image.type(self.dtype), ctx))
        #image_features_pool = torch.stack(image_features_pool)
        text_features, normalized_toutputs = self.text_encoder(prompts, tokenized_prompts, deeper_text_prompts)

        image_1 = image.unsqueeze(1).expand(-1, self.N, -1, -1, -1).contiguous().view(-1, img_c, img_h, img_w)
        shared_ctx = shared_ctx.unsqueeze(0).expand(batch_size, -1, -1, -1)
        shared_ctx = shared_ctx.contiguous().view(image_1.shape[0], shared_ctx.shape[-2], shared_ctx.shape[-1])
        image_outputs = self.image_encoder(image_1.type(self.dtype), shared_ctx, deeper_vision_prompts)

        image_features = image_outputs[:,0,:]   ### (number_vision_prompts * batch_size, dim)

        ###  normalized_toutputs: (list, each element is (number_text_prompts, n_tokens + 1, dim))
        ###  image_output: (number_vision_prompts * batch_size, number_patches + 1, dim)
        image_outputs = image_outputs / image_outputs.norm(dim=-1, keepdim=True)
        image_outputs = image_outputs.view(batch_size, self.vision_prompt_number, -1,
                                            image_outputs.shape[-1])
        image_outputs = image_outputs[:, :, 1:, :]   ### 4, bs, 196, d

        low_ot_loss = []
        image_plan = []
        for each_image_id in range(batch_size):
            each_image_features = image_outputs[each_image_id]  ### number_vision_prompts, n_patches +1, dim
            each_image_plan = []
            for cls_id, each_class_prompt in enumerate(normalized_toutputs):  ### number_text_prompts, n_tokens + 1, dim
                sim = torch.einsum('vnd, tld->vtnl', each_image_features,
                                    each_class_prompt).contiguous()  ### number_vision, number_text, n_patches + 1, n_tokens + 1
                sim = sim.view(-1, each_image_features.shape[1],
                                each_class_prompt.shape[1])  ### (number_vision * number_text, n_patches, n_tokens)
                wdist = 1.0 - sim
                # print(wdist.shape, each_image_id, cls_id)
                xx = torch.zeros(self.vision_prompt_number * self.text_prompt_number, each_image_features.shape[1],
                                    dtype=sim.dtype, device=sim.device).fill_(
                    1. / each_image_features.shape[1])
                yy = torch.zeros(self.vision_prompt_number * self.text_prompt_number, each_class_prompt.shape[1],
                                    dtype=sim.dtype, device=sim.device).fill_(
                    1. / each_class_prompt.shape[1])

                with torch.no_grad():
                    KK = torch.exp(-wdist / 0.1)
                    T = Sinkhorn(KK, xx, yy)    ### T: n_v, n_t, number_patch, num_tokens
                if torch.isnan(T).any():
                    return None      

                sim_op = torch.sum(T * sim, dim=(1, 2))  ##### OT distance: (number_vision * number_text, )
                sim_op = sim_op.contiguous().view(self.vision_prompt_number, self.text_prompt_number)
                low_ot_loss.append(sim_op)
                # print(sim_op.shape)
            # each_image_plan_mean = torch.stack(each_image_plan).mean(0)
            # image_plan.append(torch.topk(each_image_plan_mean,10,dim=-1)[0])
        low_ot_loss = 1.0 * torch.stack(low_ot_loss)  #### batch_size * n_cls, number_vision, number_text
        # print(low_ot_loss.shape)

        image_features = image_features.view(batch_size, self.vision_prompt_number,
                                                image_features.shape[-1])  #### (vision_prompt_number, batch_size, dim)
        text_features = text_features.view(self.prompt_learner.n_cls, self.text_prompt_number,
                                            text_features.shape[-1])  #### (text_prompt_number, n_cls, dim)

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        sim = torch.einsum('bvd, ntd->bnvt', image_features, text_features).contiguous()
        sim = sim.view(batch_size * self.prompt_learner.n_cls, self.vision_prompt_number, self.text_prompt_number)
        wdist = 1.0 - sim
        wdist += low_ot_loss 
        # wdist = low_ot_loss
        xx = torch.zeros(batch_size * self.prompt_learner.n_cls, self.vision_prompt_number,
                            dtype=sim.dtype, device=sim.device).fill_(1. / self.vision_prompt_number)
        yy = torch.zeros(batch_size * self.prompt_learner.n_cls, self.text_prompt_number,
                            dtype=sim.dtype, device=sim.device).fill_(1. / self.text_prompt_number)
        with torch.no_grad():
            KK = torch.exp(-wdist / 0.1)
            T = Sinkhorn(KK, xx, yy)  ### T: (batch_size * n_cls, vision_number, text_number)
        if torch.isnan(T).any():
            return None

        sim_op = torch.sum(T * sim, dim=(1, 2))  ##### OT distance: (batch_size * n_cls, )
        sim_op = sim_op.contiguous().view(batch_size, self.prompt_learner.n_cls)
        logits = logit_scale * sim_op

        cos = torch.nn.CosineSimilarity(dim=1,eps=1e-07)
        #text_feature = torch.mean(text_features_pool, dim=0)
        #text_feature = text_feature / text_feature.norm(dim=-1, keepdim=True)
        with torch.no_grad():
            text_features_ori = self.ori_embedding
            text_features_ori = text_features_ori / text_features_ori.norm(dim=-1, keepdim=True)
            image_features_ori = self.prompt_learner.clip_model_0.encode_image(image)
            image_features_ori = image_features_ori / image_features_ori.norm(dim=-1, keepdim=True)
            logits_ori = logit_scale * image_features_ori @ text_features_ori.t()

        score_text = 0.0
        text_features_pool = text_features.permute(1, 0, 2)
        for text_feature in text_features_pool:
            text_feature = text_feature / text_feature.norm(dim=-1, keepdim=True)
            sim_text = cos(text_feature,text_features_ori)
            score_text += 1.0 - torch.mean(sim_text)
        #score_text = score_text / self.N

        if self.prompt_learner.training:
            return logits, logits_ori, score_text

        return logits

@TRAINER_REGISTRY.register()
class KDPT_MAPLE(TrainerX):
    """
    It is based on PLOT.
    """
    
    def check_cfg(self, cfg):
        assert cfg.TRAINER.KDPT_MAPLE.PREC in ["fp16", "fp32", "amp"]
    
    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames
        #all_classnames = self.dm.dataset.all_class_names
        
        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)
        
        print("Building custom CLIP")
        self.model = CustomCLIP(cfg, classnames, clip_model)
        #self.w = cfg.TRAINER.KDPT_MAPLE.W
        #self.w = 0.5 #cfg.TRAINER.KMVPT.W

        print("Turning off gradients in both the image and the text encoder")
        name_to_update = "prompt_learner"

        for name, param in self.model.named_parameters():
            if 'clip_model_0' in name:
                param.requires_grad_(False)
            elif name_to_update not in name:
                # Make sure that VPT prompts are updated
                if "VPT" in name:
                    param.requires_grad_(True)
                else:
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
        self.register_model("prompt_learner", self.model, self.optim, self.sched)
        device_count = torch.cuda.device_count()
        if device_count > 1:
            print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
            #self.model = nn.DataParallel(self.model, device_ids=[0,1])
        
    def forward_backward(self, batch):
        image, label = self.parse_batch_train(batch)

        logits, logits_ori, score_text = self.model(image)

        loss_ce = F.cross_entropy(logits, label)
        loss_ori = F.cross_entropy(logits_ori, label)
        loss_kd = kd_loss(logits, logits_ori, 0.01)
        #loss_2 = (loss_ce - loss_ori).pow(2)
        loss = loss_ce + score_text + 2.0 * loss_kd #+ 0.05 * loss_2 
        self.model_backward_and_update(loss)
        
        output = logits
        loss_summary = {"loss": loss.item(),
                         "acc": compute_accuracy(output, label)[0].item()}
        
        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()
            
        return loss_summary
    
    def parse_batch_train(self, batch):
        input = batch["img"]
        label = batch["label"]
        input = input.to(self.device)
        label = label.to(self.device)
        return input, label
    
    
    def load_model(self, directory, epoch=None):
        if not directory:
            print("Note that load_model() is skipped as no pretrained model is given")
            return

        names = self.get_model_names()
        print(names)

        # By default, the best model is loaded
        model_file = "model.pth.tar-5" #+ str(epoch)

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
