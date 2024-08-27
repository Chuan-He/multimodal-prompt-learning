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
    #"OxfordFlowers": "a photo of a {}",
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

template_dict = {   'Caltech101': ["a photo of a","this is a photo","this is picture of","one picture of a"], 
                    'DescribableTextures':['a photo of a texture', "this is a photo texture","this is a picture texture","one picture of a texture"],
                    'EuroSAT':['a centered satellite photo of', 'a centered satellite picture of','this is centered satellite photo of','one centered satellite photo of a'], 
                    'FGVCAircraft':['a photo of an aircraft','a picture of an aircraft','this is aircraft picture of','one picture of an aircraft'],
                    'Food101':['a photo of a food', 'this is a food photo', ' this is food picture of','one picture of a food'], 
                    'ImageNet':["a photo of a","this is a photo ","this is a","one picture of a"],
                    'OxfordFlowers':['a photo of a flower', 'one picture of a flower','this is flower picture of','one picture of a flower'],
                    'OxfordPets':['a photo of a pet', 'one picture of a pet','this is pet picture of','one picture of a pet'],
                    'StanfordCars':["a photo of a","this is a photo ","this is picture of","one picture of a"],
                    #'StanfordCars':["a photo of a","a photo of a","a photo of a","a photo of a"],
                    'SUN397':["a photo of a","this is a photo","this is picture of","one picture of a"],
                    'UCF101':['a photo of a person doing', 'this is a photo people doing', 'this is picture of people doing', 'one picture of a person doing'],
                    #'UCF101':['a photo of a person doing', 'a photo of a person doing', 'a photo of a person doing', 'a photo of a person doing'],
                }


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
    design_details = {"trainer": 'KMDPT',
                      "vision_depth": 0,
                      "language_depth": 0, "vision_ctx": 0,
                      "language_ctx": 0,
                      "maple_length": cfg.TRAINER.KMDPT.N_CTX}
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
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x
    
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
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

class MultiModalPromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = cfg.TRAINER.KMDPT.N_CTX
        ctx_init_flag = cfg.TRAINER.KMDPT.CTX_INIT
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg.INPUT.SIZE[0]
        N = cfg.TRAINER.KMDPT.N   # the number of our text prompts 
        #M = len(clip_template[cfg.DATASET.NAME])
        self.N = N #+ M

        self.compound_prompts_depth = self.N  # max=12, but will create 11 such shared prompts

        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"
        
        if ctx_init_flag:
            ctx_list = template_dict[cfg.DATASET.NAME]
            #ctx_list1 = clip_template[cfg.DATASET.NAME]
            #n_ctx = len(ctx_list[0].split())
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
            '''
            for i in range(M):
                ctx_init = ctx_list1[i].replace("_", " ")
                prompt = clip.tokenize(ctx_init)
                with torch.no_grad():
                    embedding = clip_model.token_embedding(prompt).type(dtype)
                ctx_vectors = embedding[0, 1 : 1 + n_ctx, :]
                ctx_vectors_list.append(ctx_vectors)
                prompt_prefix = ctx_init
                prompt_prefix_list.append(prompt_prefix)
            '''               
            ctx_vectors = torch.stack(ctx_vectors_list) # N, n_ctx, ctx_dim
            
        #else:
        #    ctx_vectors = torch.empty(N, n_ctx, ctx_dim, dtype=dtype)

        #    nn.init.normal_(ctx_vectors, std=0.02)

        #    prompt_prefix = " ".join(["X"] * n_ctx)
        
        self.ctx = nn.Parameter(ctx_vectors) # parameters of text prompt to be learned
        
        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        #prompts = [prompt_prefix + " " + name + "." for name in classnames]

        prompt_list = []
        if ctx_init:
            for i in range(N):
                prompt_prefix = prompt_prefix_list[i]
                prompts = [prompt_prefix + " " + name + "." for name in classnames] # 100
                tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts]) # 100x77
                prompt_list.append(tokenized_prompts)
            '''
            for i in range(M):
                template = clip_template[cfg.DATASET.NAME][i]
                prompts = [template.format(c.replace("_", " ")) for c in classnames]
                tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts]) # 100x77
                prompt_list.append(tokenized_prompts)
            '''
            tokenized_prompts = torch.cat(prompt_list)
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

        print('KMDPT design: Multi-modal Prompt Learning')

        self.single_layer = nn.Linear(ctx_dim, 768)
        self.single_layer.half()
        #self.compound_prompt_projections = _get_clones(self.single_layer, self.N)
        
        # zsclip text features
        clip_model_0 = load_clip_to_cpu_0(cfg)
        clip_model_0.to('cuda:1')
        self.clip_model_0 = clip_model_0
        #prompts_ = [prompt_prefix + " " + name + "." for name in classnames]        
        temp = CUSTOM_TEMPLATES[cfg.DATASET.NAME]
        prompts_ = [temp.format(c.replace("_", " ")) for c in classnames]
        print(f"ZSCLIP Prompts: {prompts_}")
        prompts_ = torch.cat([clip.tokenize(p) for p in prompts_])
        prompts_ = prompts_.to('cuda:1')

        with torch.no_grad():
            text_features = clip_model_0.encode_text(prompts_)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        self.text_features = text_features
    
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
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)
            
        prefix = self.token_prefix
        suffix = self.token_suffix
        prompts = self.construct_prompts(ctx, prefix, suffix)

        #shared_ctx = []
        #for index, layer in enumerate(self.compound_prompt_projections):
        #    shared_ctx.append(layer(self.ctx[index,:,:]))
        #shared_ctx = torch.stack(shared_ctx)

        return prompts, self.single_layer(self.ctx)
        
        
class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.prompt_learner = MultiModalPromptLearner(cfg, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype

        self.N = cfg.TRAINER.KMDPT.N #+ len(clip_template[cfg.DATASET.NAME])
        self.n_cls = len(classnames)

        self.ori_embedding = self.prompt_learner.text_features

    def forward(self, image):

        prompts, shared_ctx = self.prompt_learner()
        tokenized_prompts = self.tokenized_prompts

        image_features_pool = self.image_encoder(image.type(self.dtype), shared_ctx)  
        self.dim = image_features_pool.shape[-1]
        image_features_pool = image_features_pool.contiguous().view(self.N, -1, self.dim)

        text_features_pool = self.text_encoder(prompts, tokenized_prompts).contiguous().view(self.N, self.n_cls, self.dim)

        with torch.no_grad():
            text_features_ori = self.ori_embedding
            text_features_ori = text_features_ori / text_features_ori.norm(dim=-1, keepdim=True)
            image_features_ori = self.prompt_learner.clip_model_0.encode_image(image)
            image_features_ori = image_features_ori / image_features_ori.norm(dim=-1, keepdim=True)

        cos = torch.nn.CosineSimilarity(dim=1,eps=1e-07)
        logit_scale = self.logit_scale.exp()

        logits = []
        score_texts = 0
        score_images = 0
        for text_feature,image_feature in zip(text_features_pool,image_features_pool):
            text_feature = text_feature / text_feature.norm(dim=-1, keepdim=True)
            image_feature = image_feature / image_feature.norm(dim=-1, keepdim=True)
            logit = logit_scale * image_feature @ text_feature.t()
            logits.append(logit)
            score_text = cos(text_feature,text_features_ori)
            score_texts += 1.0-torch.mean(score_text)
            score_image = cos(image_feature, image_features_ori)
            score_images += 1.0-torch.mean(score_image)    
        
        #score_text = torch.mean(torch.stack(score_texts), dim=0)
        #score_image = torch.mean(torch.stack(score_images), dim=0)

        if self.prompt_learner.training:
            return torch.mean(torch.stack(logits), dim=0), score_texts, score_images
        
        return torch.mean(torch.stack(logits), dim=0)


@TRAINER_REGISTRY.register()
class KMDPT(TrainerX):
    """
    It is based on PLOT.
    """
    
    def check_cfg(self, cfg):
        assert cfg.TRAINER.KMDPT.PREC in ["fp16", "fp32", "amp"]
    
    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames
        
        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)
        
        print("Building custom CLIP")
        self.model = CustomCLIP(cfg, classnames, clip_model)
        self.w = cfg.TRAINER.KMDPT.W

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
        
        self.device = torch.device("cuda:1")
        self.model.to(self.device)
        # NOTE: we give whole model to the optimizer, but only prompt_learner will be optimized
        self.optim = build_optimizer(self.model, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("prompt_learner", self.model, self.optim, self.sched)

        
    def forward_backward(self, batch):
        image, label = self.parse_batch_train(batch)

        logits,scores_text,scores_image = self.model(image)

        loss = F.cross_entropy(logits, label) 
        loss += self.w*scores_text + self.w*scores_image
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
        model_file = "model.pth.tar-10" #+ str(epoch)

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
