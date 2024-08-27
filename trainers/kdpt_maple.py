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
import numpy as np

import learn2learn as l2l
from meta_learning import MAML
from torch.autograd import grad

_tokenizer = _Tokenizer()


def load_clip_to_cpu(cfg, zero_shot_model=False):
    backbone_name = cfg.MODEL.BACKBONE.NAME
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")
    if not zero_shot_model:
        design_details = {"trainer": 'IVLP',
                          "vision_depth": cfg.TRAINER.KDPT_MAPLE.PROMPT_DEPTH,
                          "language_depth": cfg.TRAINER.KDPT_MAPLE.PROMPT_DEPTH,
                          "vision_ctx": cfg.TRAINER.KDPT_MAPLE.N_CTX,
                          "language_ctx": cfg.TRAINER.KDPT_MAPLE.N_CTX}
        model = clip.build_model(state_dict or model.state_dict(), design_details)
    else:
        # Return original CLIP model for generating frozen VL features
        design_details = {"trainer": 'IVLP',
                          "vision_depth": 0,
                          "language_depth": 0, "vision_ctx": 0,
                          "language_ctx": 0}
        model = clip.build_model(state_dict or model.state_dict(), design_details)
        return model
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
        # vis_vectors = torch.empty(n_ctx, 768, dtype=dtype)
        # nn.init.normal_(vis_vectors, std=0.02)
        # self.vis_ctx = nn.Parameter(vis_vectors) # parameters of text prompt to be learned
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
        
        # zsclip text features
        with torch.no_grad():
            clip_model_image_0 = load_clip_to_cpu(cfg, True)
            clip_model_image_0.to('cuda')
            self.zs_image_encoder = clip_model_image_0.visual
            clip_model_text_0 = load_clip_to_cpu(cfg, True)        # Also create frozen CLIP
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

        return prompts

def gradient_update(model, loss1, loss2, loss3, grad_func=None):
    diff_params = [p for p in model.parameters() if p.requires_grad]

    grad_params1 = grad(loss1,
                        diff_params,
                        retain_graph=True,
                        create_graph=True,
                        allow_unused=True)
    grad_params2 = grad(loss2,
                        diff_params,
                        retain_graph=True,
                        create_graph=True,
                        allow_unused=True)
    grad_params3 = grad(loss3,
                        diff_params,
                        retain_graph=True,
                        create_graph=True,
                        allow_unused=True)

    gradients = []
    grad_counter = 0
    # Handles gradients for non-differentiable parameters
    for name, param in model.named_parameters():
        if param.requires_grad:
            gradient1 = grad_params1[grad_counter]
            gradient2 = grad_params2[grad_counter]
            gradient3 = grad_params3[grad_counter]
            if grad_func:
                if gradient2 == None:
                    gradient = grad_func(gradient1.type(grad_func.dtype), gradient2, gradient3.type(grad_func.dtype), name)
                elif gradient3 == None:
                    gradient = grad_func(gradient1.type(grad_func.dtype), gradient2.type(grad_func.dtype), gradient3, name)
                else:
                    raise NotImplemented
            grad_counter += 1
        else:
            gradient = None
        gradients.append(gradient)
        
    if gradients is not None:
        params = list(model.parameters())
        if not len(gradients) == len(list(params)):
            msg = 'WARNING:maml_update(): Parameters and gradients have different length. ('
            msg += str(len(params)) + ' vs ' + str(len(gradients)) + ')'
            print(msg)
        for p, g in zip(params, gradients):
            if g is not None:
                p.grad = g.type(p.dtype)

    else:
        print("Gradients are not updated!")
    
    return model

## Original ###
class VNet(nn.Module):
    def __init__(self, cfg, clip_model):
        super().__init__()
        self.dtype = clip_model.dtype
        vision_ctx_dim = clip_model.visual.conv1.weight.size(0)*4
        text_ctx_dim = clip_model.ln_final.weight.shape[0]*4

        self.linear_vision_gamma = nn.ModuleList([nn.Sequential(nn.Linear(vision_ctx_dim*2, vision_ctx_dim//8, bias=False),nn.Linear(vision_ctx_dim//8, vision_ctx_dim//8, bias=False)).type(self.dtype) for i in range(cfg.TRAINER.KDPT_MAPLE.PROMPT_DEPTH)])
        self.linear_text_gamma = nn.ModuleList([nn.Sequential(nn.Linear(text_ctx_dim*2, text_ctx_dim//8, bias=False),nn.Linear(text_ctx_dim//8, text_ctx_dim//8, bias=False)).type(self.dtype) for i in range(cfg.TRAINER.KDPT_MAPLE.PROMPT_DEPTH)])
        
        
    def forward(self, gradients1, gradients2, gradients3, param_name):
        if "image_encoder" in param_name:
            if param_name == "image_encoder.VPT":
                linear_gamma = self.linear_vision_gamma[0]
            else:
                l_idx = int(param_name.split("image_encoder.transformer.resblocks.")[1].split(".VPT_shallow")[0])
                linear_gamma = self.linear_vision_gamma[l_idx]

        elif "text_encoder" in param_name or (param_name == "prompt_learner.ctx"):
            if param_name == "prompt_learner.ctx":
                linear_gamma = self.linear_text_gamma[0]
            else:
                l_idx = int(param_name.split("text_encoder.transformer.resblocks.")[1].split(".VPT_shallow")[0])
                linear_gamma = self.linear_text_gamma[l_idx]
        else:
            raise NotImplemented
            
        d_1, d_2 = gradients1.size()
        changed_gradients1, changed_gradients2, changed_gradients3 = None, None, None 
        if gradients2 == None:
            input_gradients = torch.cat((gradients1, gradients3), 0)
            gamma_t = torch.sigmoid(linear_gamma(input_gradients.reshape(1,-1))).repeat_interleave(8,-1).reshape(d_1, d_2)
            changed_gradients = gamma_t*(gradients3)*2
            changed_gradients = gradients1 + changed_gradients
            
        elif gradients3 == None:
            input_gradients = torch.cat((gradients1, gradients2), 0)
            gamma_t = torch.sigmoid(linear_gamma(input_gradients.reshape(1,-1))).repeat_interleave(8,-1).reshape(d_1, d_2)
            changed_gradients = gamma_t*(gradients2)*2
            changed_gradients = gradients1 + changed_gradients
        else:
            raise NotImplemented
        
        # beta_t = torch.sigmoid(linear_beta(gradients.reshape(1, -1))).reshape(d_1, d_2)

        return changed_gradients

class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.prompt_learner = MultiModalPromptLearner(cfg, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        self.total_epochs = 10
        #self.image_adapter = Adapter(512).half()

        self.n_cls = len(classnames)
    
    def text_align_loss1(self, feats, feats_ori, layers_from=0, layers_to=12):
        '''
        A feature distibution alignment L1 loss between mean and variance of the features
        '''
        layer_loss = []
        transf_layers = layers_to
        for l in range(layers_from, transf_layers):
            layer_loss.append(F.l1_loss(feats[l], feats_ori[l], reduction='mean'))
        #F.l1_loss(feats[l], feats_ori[l], reduction='mean') 
        #+ F.l1_loss(torch.var(feats[l], dim=1), torch.var(feats_ori[l], dim=1), reduction='mean')
        return layer_loss[-1]
    
    def visual_align_loss1(self, feats, feats_ori, layers_from=0, layers_to=12):
        '''
        A feature distibution alignment L1 loss between mean and variance of the features
        '''

        layer_loss = []
        transf_layers = layers_to
        for l in range(layers_from, transf_layers):
            feat = feats[l].mean(0)[:197,:]
            feat_ori = feats_ori[l].mean(0)[:197,:]
            layer_loss.append(F.l1_loss(feat, feat_ori, reduction='mean')) 
        #+ F.l1_loss(torch.var(feat, dim=1), torch.var(feat_ori, dim=1), reduction='mean')
        # weighted_loss = 0
        # for l in range(layers_from, transf_layers):
        #     weighted_loss += weights[n - l + layers_from] * layer_loss[l]
        return layer_loss[-1]

    def text_align_loss(self, text_features, text_features_ori):
        score = []
        cos = torch.nn.CosineSimilarity(dim=0, eps=1e-07)
        for text_feature, text_feature_ori in zip(list(text_features), list(text_features_ori)):
            sim = cos(text_feature, text_feature_ori)
            score.append(1.0 - sim)
        #var = torch.var(torch.stack(score), dim=0)
        #return var
        max_score = max(score)
        return max_score
    
    def visual_align_loss(self, image_features, image_features_ori):
        score = []
        cos = torch.nn.CosineSimilarity(dim=0, eps=1e-07)
        for image_feature, image_feature_ori in zip(list(image_features), list(image_features_ori)):
            sim = cos(image_feature, image_feature_ori)
            score.append(1.0 - sim)
        max_score = max(score)
        return max_score
    
    def forward(self, image, label=None, image_sup=None, label_sup=None, mix_ids=None, lam=None, mixup=False):

        prompts = self.prompt_learner()
        tokenized_prompts = self.tokenized_prompts
        logit_scale = self.logit_scale.exp()

        text_features = self.text_encoder(prompts, tokenized_prompts)
        text_feats = torch.cat([res.text_feat.permute(1, 0, 2).unsqueeze(0) for res in self.text_encoder.transformer.resblocks])
        image_features = self.image_encoder(image.to('cuda').type(self.dtype))
        visual_feats = torch.cat([res.visual_feat.permute(1, 0, 2).unsqueeze(0) for res in self.image_encoder.transformer.resblocks])

        self.dim = image_features.shape[-1]
        self.batch_size = image_features.shape[-2]

        if self.prompt_learner.training and mixup:
            b_size = image.size(0)
            image_features_sup = self.image_encoder(image_sup.type(self.dtype))
            image_features = lam.view(b_size, 1)*image_features + (1-lam).view(b_size,1)*image_features_sup[mix_ids]
            image_features = image_features.type(self.dtype)
            label_b = label_sup[mix_ids]

        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        logits = logit_scale * image_features @ text_features.t()

        if self.prompt_learner.training:
            # Now calculate the frozen pre-trained features
            fixed_embeddings = self.prompt_learner.text_features_ori  # precomputed pre-trained frozen textual features
            fixed_embeddings = fixed_embeddings / fixed_embeddings.norm(dim=-1, keepdim=True)
            with torch.no_grad():
                zero_shot_features = self.prompt_learner.zs_image_encoder(image.type(self.dtype))
                zero_shot_features = zero_shot_features / zero_shot_features.norm(dim=-1, keepdim=True)
                visual_feats_ori = torch.cat([res.visual_feat.permute(1, 0, 2).unsqueeze(0) for res in self.image_encoder.transformer.resblocks])
                # Compute pre-trained frozen visual features
                zero_shot_logits = logit_scale * zero_shot_features.cuda() @ fixed_embeddings.half().cuda().t()
            
            if mixup:
                ce_loss = (lam*F.cross_entropy(logits, label, reduce=False)+(1-lam)*F.cross_entropy(logits, label_b, reduce=False)).mean()
            else:
                ce_loss = F.cross_entropy(logits, label)


            #align_text = self.text_align_loss1(list(text_feats), list(text_feats_ori), 0, 12)
            align_vision = self.visual_align_loss1(list(visual_feats), list(visual_feats_ori), 0, 12)
            #loss_align_text = self.text_align_loss(text_features, text_features_ori)
            #loss_align_vision = self.visual_align_loss(image_features, image_features_ori)

            return ce_loss, text_features, fixed_embeddings, zero_shot_features, \
                image_features, zero_shot_logits, align_vision, logits
        else:
            return logits
        
        # with torch.no_grad():
        #     text_features_ori = self.prompt_learner.text_features_ori
            
        #     text_features_ori = text_features_ori / text_features_ori.norm(dim=-1, keepdim=True)
        #     image_features_ori = self.prompt_learner.zs_image_encoder(image.type(self.dtype))
        #     image_features_ori = image_features_ori / image_features_ori.norm(dim=-1, keepdim=True)
            
        #     zero_shot_logits = logit_scale * image_features_ori @ text_features_ori.t() 


        


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
        self.vnet = VNet(cfg, clip_model)


        print("Turning off gradients in both the image and the text encoder")
        name_to_update = "prompt_learner"

        for name, param in self.model.named_parameters():
            if name_to_update not in name:
                # Make sure that VPT prompts are updated
                if "VPT" in name:
                    param.requires_grad_(True)
                else:
                    param.requires_grad_(False)
            else:
                if "zs_image_encoder" in name:
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
        self.vnet.to(self.device)
        # NOTE: we give whole model to the optimizer, but only prompt_learner will be optimized
        self.optim = build_optimizer(self.model, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("prompt_learner", self.model, self.optim, self.sched)

        self.optim_vnet = build_optimizer(self.vnet, cfg.OPTIM_VNET)
        self.sched_vnet = build_lr_scheduler(self.optim_vnet, cfg.OPTIM_VNET)

        # Cosine scheduler
        self.w1 = cfg.TRAINER.W1
        self.w2 = cfg.TRAINER.W2
        self.w3 = cfg.TRAINER.W3
        self.w4 = cfg.TRAINER.W4
        self.w5 = cfg.TRAINER.W5
        self.w6 = cfg.TRAINER.W6

        self.total_epochs = 10
        self.step_counter = 1
        N = 10
        
        self.adapt_lr = 0.0005
        self.lr_ratio = 0.0005


        self.fast_adaptation = False
        
        self.mixup_alpha = 0.5
        self.mixup_beta = 0.5
        
    def forward_backward(self, batch):
        image, label = self.parse_batch_train(batch)
        cos = torch.nn.CosineSimilarity(dim=1,eps=1e-07)
        # logits, zero_shot_logits, score_text, score_image, scl_text, scl_image, loss_align_text, loss_align_vision = model(img0)
        # # Now calculate L_SCL_logits
        # L_SCL_logits = F.kl_div(
        #     F.log_softmax(logits / 1, dim=1),
        #     F.log_softmax(zero_shot_logits / 1, dim=1),
        #     reduction='sum',
        #     log_target=True
        # ) * (1 * 1) / logits.numel()
        # #optim.zero_grad()
        # loss_ce = F.cross_entropy(logits, label)
        # loss = loss_ce #+ L_SCL_logits
        # loss += self.w1 * score_text + self.w2 * score_image 
        # loss += self.w3 * scl_text + self.w4 * scl_image
        # loss += self.w5 * loss_align_text + self.w6 * loss_align_vision

        # self.model_backward_and_update(loss)
        # # loss.backward()
        # # optim.step()
        model = self.model
        optim = self.optim
        optim_vnet = self.optim_vnet
        
        lam = None
        loss_ce, normalized_text_features, zs_clip_text_embeddings, zs_image_embedd, image_ft, \
        zero_shot_logits, align_vision, logits = model(image, label, mixup=False)

        sim_text = cos(normalized_text_features, zs_clip_text_embeddings)
        score_text = 1.0 - torch.mean(sim_text)
        sim_image = cos(image_ft, zs_image_embedd)
        score_image = 1.0 - torch.mean(sim_image)

        scl_text = F.l1_loss(normalized_text_features, zs_clip_text_embeddings.cuda(), reduction='mean')    
        scl_image = F.l1_loss(image_ft, zs_image_embedd.cuda(), reduction='mean')
        rag_text = score_text * self.w1
        rag_image = align_vision * self.w6

        optim.zero_grad()
        model = gradient_update(model, loss_ce, rag_text, rag_image, grad_func=self.vnet)
        optim.step()

        maml = MAML(model, lr=self.adapt_lr, first_order=self.fast_adaptation)

        loss = torch.tensor(0.0)
        unique_label = torch.unique(label)
        if len(unique_label) != 1:
            qry_l = unique_label[torch.randperm(len(unique_label))][0]
            qry_ids = torch.where(label==qry_l)[0]
            sup_ids = torch.where(label!=qry_l)[0]
            x_sup, y_sup = image[sup_ids], label[sup_ids]
            x_qry, y_qry = image[qry_ids], label[qry_ids]

            b_size = x_qry.size(0)
            lam = torch.distributions.beta.Beta(self.mixup_alpha, self.mixup_beta).sample((b_size,)).to(image.device)
            mix_ids = torch.randint(x_sup.size(0), (x_qry.size(0),))

            task_model = maml.clone(allow_nograd=True)
            adaptation_loss_ce, normalized_text_features, zs_clip_text_embeddings, zs_image_embedd, image_ft, \
            zero_shot_logits, align_vision, logits = task_model(x_sup, y_sup, mixup=False)

            sim_text = cos(normalized_text_features, zs_clip_text_embeddings)
            score_text = 1.0 - torch.mean(sim_text)
            sim_image = cos(image_ft, zs_image_embedd)
            score_image = 1.0 - torch.mean(sim_image)

            scl_text = F.l1_loss(normalized_text_features, zs_clip_text_embeddings.cuda(), reduction='mean')    
            scl_image = F.l1_loss(image_ft, zs_image_embedd.cuda(), reduction='mean')
            adaptation_rag_text = score_text * self.w1
            adaptation_rag_image = align_vision * self.w6
            
            task_model.adapt(adaptation_loss_ce, adaptation_rag_text, adaptation_rag_image, allow_nograd=True, grad_func=self.vnet, allow_unused=True)
            loss2_ce, normalized_text_features, zs_clip_text_embeddings, zs_image_embedd, image_ft, \
            zero_shot_logits, align_vision, logits = task_model(x_qry, y_qry, x_sup, y_sup, mix_ids, lam=lam, mixup=True) 

            loss = loss2_ce* self.lr_ratio
            optim.zero_grad()
            optim_vnet.zero_grad()
            loss.backward()
            optim.step()
            optim_vnet.step()

        # output = logits
        # loss_summary = {"loss": loss.item(),
        #                  "acc": compute_accuracy(output, label)[0].item()}
        loss_summary = {"loss": loss.item()}

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
