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

template_dict = {   #'Caltech101': ["a photo of a","this is a photo","this is picture of","one picture of a"],
                    'Caltech101': ["a photo of a","a painting of a","a photo of the","a painting of the"],
                    #'DescribableTextures':['a photo of a texture', "this is a photo texture","this is a picture texture","one picture of a texture"],
                    'DescribableTextures':['a photo of a texture', "a photo of a pattern","a photo of a thing","a photo of a object"],
                    'EuroSAT':['a centered satellite photo of', 'a centered satellite picture of','this is centered satellite photo of','one centered satellite photo of a'], 
                    #'EuroSAT':['a centered satellite photo of', 'a centered satellite photo of a','this is centered satellite photo of','a centered satellite photo of the'], 
                    #'FGVCAircraft':['a photo of an aircraft','a picture of an aircraft','this is aircraft picture of','one picture of an aircraft'],
                    #'FGVCAircraft':['a photo of a','a photo of the','a photo of an aircraft','a photo of the aircraft'],
                    'FGVCAircraft':['a photo of a','a photo of the','an aircraft photo of','a photo of the aircraft'],
                    'Food101':['a photo of a food', 'this is a food photo', ' this is food picture of','one picture of a food'], 
                    #'ImageNet':["a photo of a","this is a photo ","this is a","one picture of a"],
                    'ImageNet':["a photo of a","a bad photo of the","a photo of the large","a photo of the small"],
                    'ImageNetSketch':["a photo of a","a bad photo of the","a photo of the large","a photo of the small"],
                    'ImageNetV2':["a photo of a","a bad photo of the","a photo of the large","a photo of the small"],
                    'ImageNetA':["a photo of a","a bad photo of the","a photo of the large","a photo of the small"],
                    'ImageNetR':["a photo of a","a bad photo of the","a photo of the large","a photo of the small"],
                    #'OxfordFlowers':['a photo of a flower', 'one picture of a flower','this is flower picture of','one picture of a flower'],
                    'OxfordFlowers':['a photo of a flower', 'one picture of a flower',' a photo of a','one picture of a'],
                    #'OxfordPets':['a photo of a pet', 'one picture of a pet','this is pet picture of','one picture of a pet'],
                    'OxfordPets':['a photo of a pet', 'one picture of a pet','a photo of a','one picture of a'],
                    #'StanfordCars':["a photo of a","this is a photo ","this is picture of","one picture of a"],
                    'StanfordCars':["a photo of a","a photo of the","a photo of my","i love my"],
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
    design_details = {"trainer": 'KDPT_MAPLE_I',
                      "vision_depth": 0,
                      "language_depth": 0, "vision_ctx": 0,
                      "language_ctx": 0,
                      "maple_length": cfg.TRAINER.KDPT_MAPLE_I.N_CTX}
    model = clip.build_model(state_dict or model.state_dict(), design_details)

    return model


class TextEncoder_MaPLe(nn.Module):
    def __init__(self, clip_model):
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
        text_features = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection
        #### token_embeddings
        # x = x.contiguous().view(x.shape[0] // self.N, self.N, x.shape[-2], x.shape[-1])
        # tokenized_prompts = tokenized_prompts.contiguous().view(x.shape[0], self.N, -1)  ## todo: please check it
        # normalized_xout = []
        # for i in range(x.shape[0]):
        #     prompt_emb = x[i, :, :tokenized_prompts[i,0].argmax()-1]
        #     normalized_xout.append(prompt_emb / prompt_emb.norm(dim=-1, keepdim=True))

        return text_features #, normalized_xout
    
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
        n_ctx = cfg.TRAINER.KDPT_MAPLE_I.N_CTX
        ctx_init_flag = cfg.TRAINER.KDPT_MAPLE_I.CTX_INIT
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg.INPUT.SIZE[0]
        N = cfg.TRAINER.KDPT_MAPLE_I.N   # the number of our text prompts 
        #M = len(clip_template[cfg.DATASET.NAME])
        self.N = N #+ M

        self.compound_prompts_depth = cfg.TRAINER.KDPT_MAPLE_I.PROMPT_DEPTH  # max=12, but will create 11 such shared prompts

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
        self.proj = nn.Linear(ctx_dim, 768)
        self.proj.half()
        #vis_ctx_vectors = torch.empty(N, n_ctx, 768, dtype=dtype)
        #nn.init.normal_(vis_ctx_vectors, std=0.02)
        #self.vis_ctx = nn.Parameter(vis_ctx_vectors)
        
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

        print('KDPT_MAPLE_I design: Multi-modal Prompt Learning')
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
        self.text_encoder = TextEncoder_MaPLe(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype

        self.N = cfg.TRAINER.KDPT_MAPLE_I.N #+ len(clip_template[cfg.DATASET.NAME])
        self.n_cls = len(classnames)

        self.ori_embedding = self.prompt_learner.text_features

    def forward(self, image):
        batch_size, img_c, img_h, img_w = image.shape
        prompts, vis_ctx, deeper_text_prompts, deeper_vision_prompts = self.prompt_learner()
        tokenized_prompts = self.tokenized_prompts
        logit_scale = self.logit_scale.exp()
        
        text_features, normalized_toutputs = self.text_encoder(prompts, tokenized_prompts, deeper_text_prompts)

        image_1 = image.unsqueeze(1).expand(-1, self.N, -1, -1, -1).contiguous().view(-1, img_c, img_h, img_w)
        shared_ctx = shared_ctx.unsqueeze(0).expand(batch_size, -1, -1, -1)
        shared_ctx = shared_ctx.contiguous().view(image_1.shape[0], shared_ctx.shape[-2], shared_ctx.shape[-1])
        image_outputs = self.image_encoder(image_1.type(self.dtype), shared_ctx, deeper_vision_prompts)

        image_features = image_outputs[:,0,:]   ### (number_vision_prompts * batch_size, dim)
        
        with torch.no_grad():
            text_features_ori = self.ori_embedding
            text_features_ori = text_features_ori / text_features_ori.norm(dim=-1, keepdim=True)
            image_features_ori = self.prompt_learner.clip_model_0.encode_image(image)
            image_features_ori = image_features_ori / image_features_ori.norm(dim=-1, keepdim=True)
            logits_ori = logit_scale * image_features_ori @ text_features_ori.t()

        
        cos = torch.nn.CosineSimilarity(dim=1,eps=1e-07)
        #text_feature = torch.mean(text_features_pool, dim=0)
        #text_feature = text_feature / text_feature.norm(dim=-1, keepdim=True)
        logits = []
        
        image_feature = torch.mean(image_features_pool, dim=0)
        image_feature = image_feature / image_feature.norm(dim=-1, keepdim=True)
        for text_feature in text_features_pool:
            text_feature = text_feature / text_feature.norm(dim=-1, keepdim=True)
            logits.append(logit_scale * image_feature @ text_feature.t())
        logits = torch.mean(torch.stack(logits), dim=0)
        
        cos = torch.nn.CosineSimilarity(dim=1,eps=1e-07)
        score_text = 0.0
        for text_feature in text_features_pool:
            text_feature = text_feature / text_feature.norm(dim=-1, keepdim=True)
            sim_text = cos(text_feature,text_features_ori)
            score_text += 1.0 - torch.mean(sim_text)
        score_text = score_text / self.N

        if self.prompt_learner.training:
            return logits, score_text, logits_ori
        
        return logits

@TRAINER_REGISTRY.register()
class KDPT_MAPLE_I(TrainerX):
    """
    It is based on PLOT.
    """
    
    def check_cfg(self, cfg):
        assert cfg.TRAINER.KDPT_MAPLE_I.PREC in ["fp16", "fp32", "amp"]
    
    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames
        
        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)
        
        print("Building custom CLIP")
        self.model = CustomCLIP(cfg, classnames, clip_model)
        #self.w = cfg.TRAINER.KDPT_MAPLE_I.W
        self.w = 0.5 #cfg.TRAINER.KMVPT.W

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
        
    def forward_backward(self, batch):
        image, label = self.parse_batch_train(batch)

        logits, score_text, logits_ori= self.model(image)

        loss0 = F.cross_entropy(logits, label)
        loss_ori = F.cross_entropy(logits_ori, label)
        l3 = (loss0 - loss_ori).pow(2)

        loss = loss0 + score_text + l3 * 0.05
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
