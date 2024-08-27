import os.path as osp
from collections import OrderedDict
import math
import copy
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.metrics import compute_accuracy
from dassl.utils import load_pretrained_weights, load_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

from torch.nn import Dropout
from torch.nn.modules.utils import _pair
from functools import reduce
from operator import mul

from torch.profiler import profile, record_function, ProfilerActivity
from PIL import Image

_tokenizer = _Tokenizer()


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
    design_details = {"trainer": 'MMP',
                      "vision_depth": 0,
                      "language_depth": 0, "vision_ctx": 0,
                      "language_ctx": 0,
                      "maple_length": cfg.TRAINER.MMP.N_CTX}
    model = clip.build_model(state_dict or model.state_dict(), design_details)

    return model


class TextEncoder(nn.Module):
    def __init__(self, clip_model, number_text_prompts=4, number_tokens=4):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype
        self.number_text_prompts = number_text_prompts
        self.number_tokens=number_tokens

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
        x = x.contiguous().view(x.shape[0] // self.number_text_prompts, self.number_text_prompts, x.shape[-2], x.shape[-1])
        tokenized_prompts = tokenized_prompts.contiguous().view(x.shape[0], self.number_text_prompts, -1)  ## todo: please check it
        normalized_xout = []
        for i in range(x.shape[0]):
            prompt_emb = x[i, :, :tokenized_prompts[i,0].argmax()-1]
            normalized_xout.append(prompt_emb / prompt_emb.norm(dim=-1, keepdim=True))

        return text_features, normalized_xout

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)
    
class InferenceBlock(nn.Module):
    def __init__(self, input_units, d_theta, output_units):
        """
        :param d_theta: dimensionality of the intermediate hidden layers.
        :param output_units: dimensionality of the output.
        :return: batch of outputs.
        """
        super(InferenceBlock, self).__init__()
        self.module = nn.Sequential(
            nn.Linear(input_units, d_theta, bias=True),
            nn.ELU(inplace=True),
            nn.Linear(d_theta, d_theta, bias=True),
            nn.ELU(inplace=True),
            nn.Linear(d_theta, output_units, bias=True),
        )

    def forward(self, inps):
        out = self.module(inps)
        return out


class Amortized(nn.Module):
    def __init__(self, input_units=400, d_theta=400, output_units=400):
        super(Amortized, self).__init__()
        self.output_units = output_units
        self.weight_mean = InferenceBlock(input_units, d_theta, output_units)
        self.weight_log_variance = InferenceBlock(input_units, d_theta, output_units)

    def forward(self, inps):
        weight_mean = self.weight_mean(inps)
        weight_log_variance = self.weight_log_variance(inps)
        return weight_mean, weight_log_variance

class PromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = cfg.TRAINER.MMP.N_CTX

        self.text_prompts_number = cfg.TRAINER.MMP.TEXT_PROMPT_NUMBER
        self.vision_prompts_number = cfg.TRAINER.MMP.VISION_PROMPT_NUMBER
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        vis_dim = 768
        sigma = 1e-5
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg.INPUT.SIZE[0]
        # Default is 1, which is compound shallow prompting
        assert cfg.TRAINER.MMP.TEXT_PROMPT_DEPTH >= 1
        self.compound_prompts_depth = cfg.TRAINER.MMP.TEXT_PROMPT_DEPTH  # max=12, but will create 11 such shared prompts.
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        # random initialization
        print("Initializing a generic context")
        ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
        nn.init.normal_(ctx_vectors, std=0.02)
        prompt_prefix = " ".join(["X"] * n_ctx)

        print('MaPLe design: Multi-modal Prompt Learning')
        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of MaPLe context words (tokens): {n_ctx}")
        # These below, related to the shallow prompts
        # Linear layer so that the tokens will project to 512 and will be initialized from 768
        self.proj = nn.Linear(ctx_dim, vis_dim)
        self.proj.half()
        self.ctx = nn.Parameter(ctx_vectors)
        # These below parameters related to the shared prompts
        # Define the compound prompts for the deeper layers

        # Minimum can be 1, which defaults to shallow MaPLe
        # compound prompts
        self.compound_prompts_text = nn.ParameterList([nn.Parameter(torch.empty(n_ctx, ctx_dim))
                                                      for _ in range(self.compound_prompts_depth - 1)])
        for single_para in self.compound_prompts_text:
            nn.init.normal_(single_para, std=0.02)
        # Also make corresponding projection layers, for each prompt
        single_layer = nn.Linear(ctx_dim, vis_dim)
        self.compound_prompt_projections = _get_clones(single_layer, self.compound_prompts_depth - 1)

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])  # (n_cls, n_tkn)
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx:, :])  # CLS, EOS

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens

        self.meta_net = nn.Sequential(OrderedDict([
            ("linear1", nn.Linear(ctx_dim, ctx_dim * 2)),
            ("relu", nn.ReLU(inplace=True)),
            ("linear2", nn.Linear(ctx_dim * 2, ctx_dim))
        ]))
        
        self.meta_net.half()

    def construct_prompts(self, ctx, prefix, suffix, label=None):
        # dim0 is either batch (during training) or n_cls (during testing)
        # ctx: context tokens, with shape of (dim0, n_ctx, ctx_dim)
        # prefix: the sos token, with shape of (n_cls, 1, ctx_dim)
        # suffix: remaining tokens, with shape of (n_cls, *, ctx_dim)

        if label is not None:
            prefix = prefix[label]
            suffix = suffix[label]

        prompts = torch.cat(
            [
                prefix,  # (dim0, 1, dim)
                ctx,  # (dim0, n_ctx, dim)
                suffix,  # (dim0, *, dim)
            ],
            dim=1,
        )

        return prompts
    
    def forward_vision(self):
        ctx = self.proj(self.ctx)
        compound_prompts_vision = []
        for index, layer in enumerate(self.compound_prompt_projections):
            compound_prompts_vision.append(layer(self.compound_prompts_text[index]))
        return ctx, compound_prompts_vision
    
    def forward(self, im_features):
        # [TXT_NUM_TOKENS, dim] = [16, 512] (default)
        ctx = self.ctx
        #ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)             # (1, n_ctx, ctx_dim)

        prefix = self.token_prefix
        suffix = self.token_suffix
        #prompts = self.construct_prompts(ctx, prefix, suffix) 
        
        bias = self.meta_net(im_features)  # (batch, ctx_dim)
        bias = bias.unsqueeze(1)           # (batch, 1, ctx_dim)
        
        ctx = ctx.unsqueeze(0)             # (1, n_ctx, ctx_dim)
        ctx_shifted = ctx + bias           # (batch, n_ctx, ctx_dim)
        # Use instance-conditioned context tokens for all classes
        prompts = []
        for ctx_shifted_i in ctx_shifted:
            ctx_i = ctx_shifted_i.unsqueeze(0).expand(self.n_cls, -1, -1)
            pts_i = self.construct_prompts(ctx_i, prefix, suffix)  # (n_cls, n_tkn, ctx_dim)
            prompts.append(pts_i)
        prompts = torch.stack(prompts) # (b, n_cls, n_tkn, ctx_dim)

        deep_compound_prompts = []
        for i in range(self.compound_prompts_depth - 1):
            ctx = self.compound_prompts_text[i].unsqueeze(0)             # (1, n_ctx, ctx_dim)
            ctx_shifted = ctx + bias           # (batch, n_ctx, ctx_dim)
        
            # Use instance-conditioned context tokens for all classes
            compound_prompts = []
            for ctx_shifted_i in ctx_shifted:
                ctx_i = ctx_shifted_i.unsqueeze(0).expand(self.n_cls, -1, -1)
                #pts_i = self.construct_prompts(ctx_i, prefix, suffix)  # (n_cls, n_tkn, ctx_dim)
                compound_prompts.append(ctx_i) # list (n_cls, n_tkn, ctx_dim)
            deep_compound_prompts.append(torch.stack(compound_prompts)) # list (b, n_cls, n_tkn, ctx_dim)
        deep_compound_prompts = torch.stack(deep_compound_prompts) # (depth, b, n_cls, n_tkn, ctx_dim)
        deep_compound_prompts = deep_compound_prompts.permute(1, 0, 2, 3, 4) # (b, depth, n_cls, n_tkn, ctx_dim)

        return prompts, deep_compound_prompts

    
class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.prompt_learner = PromptLearner(cfg, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model, number_text_prompts=cfg.TRAINER.MMP.TEXT_PROMPT_NUMBER, number_tokens=cfg.TRAINER.MMP.N_CTX)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        self.vision_prompt_number = cfg.TRAINER.MMP.VISION_PROMPT_NUMBER
        self.text_prompt_number = cfg.TRAINER.MMP.TEXT_PROMPT_NUMBER
       
    def forward(self, image, label=None):

        tokenized_prompts = self.tokenized_prompts
        logit_scale = self.logit_scale.exp()

        image_features = self.image_encoder(image.type(self.dtype), self.prompt_learner.ctx, self.prompt_learner.compound_prompts_vision)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        prompts, deep_compound_prompts_txt  = self.prompt_learner(image_features)
        
        logits = []
        for pts_i, imf_i, deep_compound_prompts_i in zip(prompts, image_features, deep_compound_prompts_txt):
            text_features = self.text_encoder(pts_i, tokenized_prompts, list(deep_compound_prompts_i))
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            l_i = logit_scale * imf_i @ text_features.t()
            logits.append(l_i)
        logits = torch.stack(logits)

        if self.prompt_learner.training:
            return F.cross_entropy(logits, label)

        return logits

@TRAINER_REGISTRY.register()
class MMP(TrainerX):
    def check_cfg(self, cfg):
        assert cfg.TRAINER.MMP.PREC in ["fp16", "fp32", "amp"]

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)

        if cfg.TRAINER.MMP.PREC == "fp32" or cfg.TRAINER.MMP.PREC == "amp":
            # CLIP's default precision is fp16
            clip_model.float()

        print("Building custom CLIP")
        self.model = CustomCLIP(cfg, classnames, clip_model)

        print("Turning off gradients in both the image and the text encoder")
        name_to_update = "prompt_learner"

        for name, param in self.model.named_parameters():
            if name_to_update not in name:
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

        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model, cfg.MODEL.INIT_WEIGHTS)

        self.model.to(self.device)
        # NOTE: only give prompt_learner to the optimizer
        self.optim = build_optimizer(self.model, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("PromptLearner", self.model, self.optim, self.sched)

        self.scaler = GradScaler() if cfg.TRAINER.MMP.PREC == "amp" else None

        # Note that multi-gpu training could be slow because CLIP's size is
        # big, which slows down the copy operation in DataParallel
        device_count = torch.cuda.device_count()
        if device_count > 1:
            print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
            self.model = nn.DataParallel(self.model)

    def forward_backward(self, batch):
        image, label, images = self.parse_batch_train(batch)

        model = self.model
        optim = self.optim
        scaler = self.scaler

        prec = self.cfg.TRAINER.MMP.PREC
        if prec == "amp":
            with autocast():
                loss = model(image, label, images)
            optim.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()
        else:
            loss = model(image, label)
            optim.zero_grad()
            loss.backward()
            optim.step()

        loss_summary = {"loss": loss.item()}

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

    def parse_batch_train(self, batch):
        input = batch["img"]
        label = batch["label"]
        input = input.to(self.device)
        label = label.to(self.device)
        images = []
        for each in batch['impath']:
            # images.append(np.asarray(Image.open(each))/255.0)
            images.append(Image.open(each, mode='r'))
            # images.append(cv2.imread(each))
        # print(len(images), images[0].shape)
        return input, label, images

    def load_model(self, directory, epoch=None):
        if not directory:
            print("Note that load_model() is skipped as no pretrained model is given")
            return

        names = self.get_model_names()

        # By default, the best model is loaded
        model_file = "model-best.pth.tar"

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
            # set strict=False
            self._models[name].load_state_dict(state_dict, strict=False)
