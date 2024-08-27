import os.path as osp

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.metrics import compute_accuracy
from dassl.utils import load_pretrained_weights, load_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler
from tqdm import tqdm

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
                    'EuroSAT':['a centered satellite photo of', 'a centered satellite photo of', 'a centered satellite photo of', 'a centered satellite photo of'], 
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
    
class TextEncoder_MMP(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype
        # self.number_text_prompts = 2

    def forward(self, prompts, tokenized_prompts, compound_prompts_deeper_text):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        # Pass as the list, as nn.sequential cannot process multiple arguments in the forward pass
        combined = [x, compound_prompts_deeper_text, 0]  # third argument is the counter which denotes depth of prompt
        outputs = self.transformer(combined)
        x = outputs[0]  # extract the x back from here
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)
        x = x @ self.text_projection

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        text_features = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)]

        #### token_embeddings
        # x = x.contiguous().view(x.shape[0] // self.number_text_prompts, self.number_text_prompts, x.shape[-2], x.shape[-1])
        # tokenized_prompts = tokenized_prompts.contiguous().view(x.shape[0], self.number_text_prompts, -1)  ## todo: please check it
        # normalized_xout = []
        # for i in range(x.shape[0]):
        #     prompt_emb = x[i, :, :tokenized_prompts[i,0].argmax()-1]
        #     normalized_xout.append(prompt_emb / prompt_emb.norm(dim=-1, keepdim=True))

        return text_features #, normalized_xout

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

class Adapter(nn.Module):
    def __init__(self, c_in, reduction=4):
        super(Adapter, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(c_in, c_in // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(c_in // reduction, c_in, bias=False),
            nn.ReLU(inplace=True),
        )
        # self.fc = nn.Linear(c_in, c_in, bias=True)

    def forward(self, x):
        x = self.fc(x)
        return x

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
        self.N = N

        self.compound_prompts_depth = cfg.TRAINER.KDPT_MAPLE.PROMPT_DEPTH  # max=12, but will create 11 such shared prompts

        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"
        
        # if ctx_init_flag:
        #     ctx_list = template_dict[cfg.DATASET.NAME]
        #     #ctx_list1 = clip_template[cfg.DATASET.NAME]
        #     #n_ctx = len(ctx_list[0].split())
        #     ctx_vectors_list = []
        #     prompt_prefix_list = []
            
        #     for i in range(N):
        #         ctx_init = ctx_list[i].replace("_", " ")
        #         prompt = clip.tokenize(ctx_init)
        #         with torch.no_grad():
        #             embedding = clip_model.token_embedding(prompt).type(dtype)
        #         ctx_vectors = embedding[0, 1 : 1 + n_ctx, :]
        #         ctx_vectors_list.append(ctx_vectors)
        #         prompt_prefix = ctx_init
        #         prompt_prefix_list.append(prompt_prefix)
        #     '''
        #     for i in range(M):
        #         ctx_init = ctx_list1[i].replace("_", " ")
        #         prompt = clip.tokenize(ctx_init)
        #         with torch.no_grad():
        #             embedding = clip_model.token_embedding(prompt).type(dtype)
        #         ctx_vectors = embedding[0, 1 : 1 + n_ctx, :]
        #         ctx_vectors_list.append(ctx_vectors)
        #         prompt_prefix = ctx_init
        #         prompt_prefix_list.append(prompt_prefix)
        #     '''               
        #     ctx_vectors = torch.stack(ctx_vectors_list) # N, n_ctx, ctx_dim
            
        #else:
        #    ctx_vectors = torch.empty(N, n_ctx, ctx_dim, dtype=dtype)

        #    nn.init.normal_(ctx_vectors, std=0.02)

        #    prompt_prefix = " ".join(["X"] * n_ctx)
        ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
        nn.init.normal_(ctx_vectors, std=0.02)
        prompt_prefix = [" ".join(["X"] * n_ctx) for _ in range(N)]

        if ctx_init and (n_ctx) <= 4:
            # use given words to initialize context vectors
            ctx_init = ctx_init.replace("_", " ")
            #ctx_init = template_dict[cfg.DATASET.NAME][i]
            prompt = clip.tokenize(ctx_init)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1: 1 + n_ctx, :]
            prompt_prefix = ctx_init

        self.ctx = nn.Parameter(ctx_vectors) # parameters of text prompt to be learned
        self.vis_ctx = nn.Parameter(torch.randn(n_ctx, 768, dtype=dtype)) # parameters of text prompt to be learned
        #self.proj0 = nn.Linear(ctx_dim, 768)
        #self.proj0.half()
        # self.proj1 = nn.Linear(ctx_dim, 768)
        # self.proj1.half()
        
        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]   #### len: n_cls * number_prompts

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])  # (n_cls * number_prompts, n_tkn)
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
        self.compound_prompts_text = nn.ParameterList([nn.Parameter(torch.empty(n_ctx, 512))
                                                      for _ in range(self.compound_prompts_depth-1)])
        self.compound_prompts_vision = nn.ParameterList([nn.Parameter(torch.empty(n_ctx, 768))
                                                for _ in range(self.compound_prompts_depth-1)])
        for single_para in self.compound_prompts_text:
            nn.init.normal_(single_para, std=0.02)
        for single_para in self.compound_prompts_vision:
            nn.init.normal_(single_para, std=0.02)
        #self.single_layer0 = nn.Linear(ctx_dim, 768)
        #self.compound_prompt_projections0 = _get_clones(self.single_layer0, self.compound_prompts_depth-1)

        # self.single_layer1 = nn.Linear(ctx_dim, 768)
        # self.compound_prompt_projections1 = _get_clones(self.single_layer1, self.compound_prompts_depth-1)
        
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
        all_inter_text_feats = []
        with torch.no_grad():
            for single_template in IMAGENET_TEMPLATES:
                x = [single_template.replace("{}", name) for name in classnames]
                x_tokenized = torch.cat([clip.tokenize(p) for p in x])
                text_features = clip_model_text_0.encode_text(x_tokenized.cuda())
                text_feats = torch.cat([res.text_feat.permute(1, 0, 2).unsqueeze(0) for res in clip_model_text_0.transformer.resblocks])
                all_inter_text_feats.append(text_feats.unsqueeze(0))
                all_teacher_features.append(text_features.unsqueeze(1))
        self.text_features_ori = torch.cat(all_teacher_features, dim=1).mean(dim=1)
        self.inter_text_feats_ori = torch.cat(all_inter_text_feats, dim=0).mean(dim=0)
    
    def construct_prompts(self, ctx, prefix, suffix, label=None):
        # dim0 is either batch_size (during training) or n_cls (during testing)
        # ctx: context tokens, with shape of (dim0, n_ctx, ctx_dim)
        # prefix: the sos token, with shape of (n_cls, 1, ctx_dim)
        # suffix: remaining tokens, with shape of (n_cls, *, ctx_dim)

        # if label is not None:
        #     prefix = prefix[label]
        #     suffix = suffix[label]
        # if ctx.dim() == 3:
        #     ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1,-1)
        # ctx = ctx.permute(1, 0, 2, 3) #  N 100 16 512

        # ctx = ctx.contiguous().view(self.N*self.n_cls,self.n_ctx,ctx.shape[3])
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

        # visual_deep_prompts0 = []
        # for index, layer in enumerate(self.compound_prompt_projections0):
        #     visual_deep_prompts0.append(layer(self.compound_prompts_text[index]))

        # visual_deep_prompts1 = []
        # for index, layer in enumerate(self.compound_prompt_projections1):
        #     visual_deep_prompts1.append(layer(self.compound_prompts_text[index]))

        #visual_contexts0 = self.proj0(self.ctx)
        #visual_contexts1 = self.proj1(self.ctx)

        return prompts, self.vis_ctx, self.compound_prompts_text, self.compound_prompts_vision
        
        
class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.prompt_learner = MultiModalPromptLearner(cfg, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder_MaPLe(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype

        self.image_adapter = Adapter(512).half()

        self.N = cfg.TRAINER.KDPT_MAPLE.N #+ len(clip_template[cfg.DATASET.NAME])
        self.n_cls = len(classnames)

    def text_align_loss(self, feats, feats_ori, layers_from=0, layers_to=11):
        '''
        A feature distibution alignment L1 loss between mean and variance of the features
        '''
        distr_loss = 0
        transf_layers = layers_to
        for l in range(layers_from, transf_layers-1):
            distr_loss += F.l1_loss(feats[l], feats_ori[l], reduction='mean')
        return distr_loss
    
    def visual_align_loss(self, feats, feats_ori, layers_from=0, layers_to=11):
        '''
        A feature distibution alignment L1 loss between mean and variance of the features
        '''
        distr_loss = 0
        transf_layers = layers_to
        for l in range(layers_from, transf_layers-1):
            distr_loss += F.l1_loss(feats[l][:,:197,:], feats_ori[l][:,:197,:], reduction='mean')
        return distr_loss
    
    def forward(self, img, img0, label=None):

        prompts, vis_ctx, deeper_text_prompts, deeper_vision_prompts = self.prompt_learner()
        tokenized_prompts = self.tokenized_prompts
        
        text_features = self.text_encoder(prompts, tokenized_prompts, deeper_text_prompts)
        text_feats = torch.cat([res.text_feat.permute(1, 0, 2).unsqueeze(0) for res in self.text_encoder.transformer.resblocks])
        image_features = self.image_encoder(img0.to('cuda').type(self.dtype), vis_ctx, deeper_vision_prompts)
        visual_feats = torch.cat([res.visual_feat.permute(1, 0, 2).unsqueeze(0) for res in self.image_encoder.transformer.resblocks])

        self.dim = image_features.shape[-1]
        self.batch_size = image_features.shape[-2]

        # text_features_pool = text_features_pool.contiguous().view(self.n_cls, self.N, self.dim)

        with torch.no_grad():
            text_features_ori = self.prompt_learner.text_features_ori
            text_feats_ori = self.prompt_learner.inter_text_feats_ori
            text_features_ori = text_features_ori / text_features_ori.norm(dim=-1, keepdim=True)
            image_features_ori = self.prompt_learner.zs_image_encoder(img0.type(self.dtype))
            image_features_ori = image_features_ori / image_features_ori.norm(dim=-1, keepdim=True)
            visual_feats_ori = torch.cat([res.visual_feat.permute(1, 0, 2).unsqueeze(0) for res in self.prompt_learner.zs_image_encoder.transformer.resblocks])

        logit_scale = self.logit_scale.exp()
        cos = torch.nn.CosineSimilarity(dim=1,eps=1e-07)

        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        #image_features = 0.9 * image_features + 0.1 * self.image_adapter(image_features)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        
        logits = logit_scale * image_features @ text_features.t()

        sim_text = cos(text_features, text_features_ori)
        score_text = 1.0 - torch.mean(sim_text)
        sim_image = cos(image_features, image_features_ori)
        score_image = 1.0 - torch.mean(sim_image)

        loss_scl_text = F.l1_loss(text_features, text_features_ori.cuda(), reduction='mean')    
        loss_scl_image = F.l1_loss(image_features, image_features_ori.cuda(), reduction='mean')

        loss_align_text = self.text_align_loss(list(text_feats), list(text_feats_ori), 9, 11)
        loss_align_vision = self.visual_align_loss(list(visual_feats), list(visual_feats_ori), 9, 11)

        if self.prompt_learner.training:
            return logits, score_text, score_image, loss_scl_text, loss_scl_image, loss_align_text , loss_align_vision
        
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
        
        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)
        
        print("Building custom CLIP")
        self.model = CustomCLIP(cfg, classnames, clip_model)
        self.w1 = cfg.TRAINER.W1
        self.w2 = cfg.TRAINER.W2
        self.w3 = cfg.TRAINER.W3
        self.w4 = cfg.TRAINER.W4
        self.w5 = cfg.TRAINER.W5
        self.w6 = cfg.TRAINER.W6

        print("Turning off gradients in both the image and the text encoder")
        name_to_update = "prompt_learner"

        for name, param in self.model.named_parameters():
            if name_to_update not in name:
                # Make sure that VPT prompts are updated
                param.requires_grad_(False)
            if name_to_update in name:
                param.requires_grad_(True)
            if 'image_adapter' in name:
                param.requires_grad_(True)
            if 'zs_image_encoder' in name:
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

    @torch.no_grad()
    def test(self, split=None):
        """A generic testing pipeline."""
        self.set_model_mode("eval")
        self.evaluator.reset()

        if split is None:
            split = self.cfg.TEST.SPLIT

        if split == "val" and self.val_loader is not None:
            data_loader = self.val_loader
        else:
            split = "test"  # in case val_loader is None
            data_loader = self.test_loader

        print(f"Evaluate on the *{split}* set")

        #for batch_idx, batch in enumerate(tqdm(data_loader)):
        for batch_idx, batch in enumerate(tqdm(data_loader)):
            img, img0, label = self.parse_batch_test(batch)
            output = self.model_inference(img, img0)
            self.evaluator.process(output, label)

        results = self.evaluator.evaluate()

        for k, v in results.items():
            tag = f"{split}/{k}"
            self.write_scalar(tag, v, self.epoch)

        return list(results.values())[0]

    def model_inference(self, img, img0):
        return self.model(img, img0)
        
    def forward_backward(self, batch):
        img, img0, label = self.parse_batch_train(batch)

        model = self.model
        #optim = self.optim

        logits, score_text, score_image, scl_text, scl_image, loss_align_text, loss_align_vision = model(img, img0, label)
        #optim.zero_grad()
        loss_ce = F.cross_entropy(logits, label)
        loss = loss_ce 
        loss += self.w1 * score_text + self.w2 * score_image 
        loss += self.w3 * scl_text + self.w4 * scl_image
        loss += self.w5 * loss_align_text + self.w6 * loss_align_vision
        self.model_backward_and_update(loss)
        # loss.backward()
        # optim.step()

        output = logits
        loss_summary = {"loss": loss.item(),
                         "acc": compute_accuracy(output, label)[0].item()}

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary
    
    def parse_batch_train(self, batch):
        img = batch["img"]
        img0 = batch["img0"]
        img0 = img0.to(self.device)
        label = batch["label"]
        label = label.to(self.device)
        return img, img0, label
    
    def parse_batch_test(self, batch):
        img = batch["img"]
        img0 = batch["img0"]
        img0 = img0.to(self.device)
        label = batch["label"]
        label = label.to(self.device)
        return img, img0, label
    
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
