#!/usr/bin/env python3

import traceback
from torch.autograd import grad

from learn2learn.algorithms.base_learner import BaseLearner
from learn2learn.utils import clone_module, update_module

import torch
import torch.nn as nn

class VNet(nn.Module):
    def __init__(self, module, layers):
        super().__init__()
        self.module = module
        self.layers = layers
        Gates = []
        for _ in range(layers):
            Gates.append(2)
        self.Gates = Gates

    def update_ctx(self, grads_ce, grads_regt, grads_regi, name, param):
        if name == "prompt_learner.ctx":
            l_idx = 0
        else:
            l_idx = int(name.split("prompt_learner.compound_prompts_text.")[1]) + 1
        
        grads_ce_regt = grads_ce + grads_regt
        grads_ce_regt_norm = grads_ce_regt / torch.linalg.norm(grads_ce_regt)
        grads_regi_norm = grads_regi / torch.linalg.norm(grads_regi)

        angle = torch.dot(grads_ce_regt_norm.flatten(), grads_regi_norm.flatten())
        
        if angle > 0:
            param.grad = grads_ce_regt
            self.Gates[l_idx] = 1
        else:
            param.grad = grads_ce_regt + grads_regi
            self.Gates[l_idx] = 2
    
    def update_proj(self, grads_ce, grads_regi, name, param):
        if "prompt_learner.proj" in name:
            l_idx = 0
        elif "prompt_learner.compound_prompt_projections" in name:
            l_idx = int(name.split("prompt_learner.compound_prompt_projections.")[1].split(".")[0]) + 1

        if self.Gates[l_idx] == 1:
            param.grad = grads_ce
        else:
            param.grad = grads_ce + grads_regi

    def forward(self, loss_ce, loss_regt, loss_regi):

        first_order = True
        second_order = not first_order

        for name, param in self.module.named_parameters():
            if param.requires_grad:
                if "prompt_learner.compound_prompts_text" in name or 'prompt_learner.ctx' in name:
                    grad_ce = grad(loss_ce, param, retain_graph=True, create_graph=second_order)[0].clone()
                    grad_regt = grad(loss_regt, param, retain_graph=True, create_graph=second_order)[0].clone()
                    grad_regi = grad(loss_regi, param, retain_graph=True, create_graph=second_order)[0].clone()
                    self.update_ctx(grad_ce, grad_regt, grad_regi, name, param)

        for name, param in self.module.named_parameters():
            if param.requires_grad:
                if "prompt_learner.compound_prompt_projections" in name or 'prompt_learner.proj' in name:
                    grad_ce = grad(loss_ce, param, retain_graph=True, create_graph=second_order)[0].clone()
                    grad_regi = grad(loss_regi, param, retain_graph=True, create_graph=second_order)[0].clone()
                    self.update_proj(grad_ce, grad_regi, name, param)

        for name, param in self.module.named_parameters():
            if param.requires_grad == False:
                param.grad = None
            elif "single_layer" in name:
                param.grad = None
            



