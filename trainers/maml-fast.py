#!/usr/bin/env python3

import traceback
from torch.autograd import grad

from learn2learn.algorithms.base_learner import BaseLearner
from learn2learn.utils import clone_module, update_module

import torch
import torch.nn as nn

class VNet(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module
        self.layer_dict = {}

    def adapt(self, loss_ce, loss_regt, loss_regi):
        first_order = True
        allow_unused = True

        second_order = not first_order

        # Compute relevant gradients
        diff_params = [p for p in self.module.parameters() if p.requires_grad]
        diff_names = [name for name, p in self.module.named_parameters() if p.requires_grad]

        grad_params_ce = grad(loss_ce,
                            diff_params,
                            retain_graph=True,
                            create_graph=second_order,
                            allow_unused=allow_unused)
        grad_params_regt = grad(loss_regt,
                            diff_params,
                            retain_graph=True,
                            create_graph=second_order,
                            allow_unused=allow_unused)
        grad_params_regi = grad(loss_regi,
                            diff_params,
                            retain_graph=True,
                            create_graph=second_order,
                            allow_unused=allow_unused)
        
        counter = 0
        for name in diff_names:
            grads_ce = grad_params_ce[counter]
            grads_regt = grad_params_regt[counter]
            grads_regi = grad_params_regi[counter]
            
            if "prompt_learner.ctx" in name:
                l_idx = 0
            elif "prompt_learner.compound_prompts_text" in name:
                l_idx = int(name.split("prompt_learner.compound_prompts_text.")[1]) + 1
            else:
                counter += 1
                continue
            grads_ce_regt = grads_ce + grads_regt
            grads_ce_regt_norm = grads_ce_regt / torch.linalg.norm(grads_ce_regt)
            grads_regi_norm = grads_regi / torch.linalg.norm(grads_regi)

            angle = torch.dot(grads_ce_regt_norm.flatten(), grads_regi_norm.flatten())
            
            if angle > 0:
                diff_params[counter].grad = grads_ce_regt
                self.layer_dict[l_idx] = 1
            else:
                diff_params[counter].grad = grads_ce_regt + grads_regi
                self.layer_dict[l_idx] = 2
            counter += 1

        counter = 0
        for name in diff_names:
            grads_ce = grad_params_ce[counter]
            grads_regi = grad_params_regi[counter]
            
            if "prompt_learner.proj" in name:
                l_idx = 0
            elif "prompt_learner.compound_prompt_projections" in name:
                l_idx = int(name.split("prompt_learner.compound_prompt_projections.")[1].split(".")[0]) + 1
            else:
                counter += 1
                continue

            if self.layer_dict[l_idx] == 1:
                diff_params[counter].grad = grads_ce
            elif self.layer_dict[l_idx] == 2:
                diff_params[counter].grad = grads_ce + grads_regi
            else:
                raise NotImplemented
            counter += 1
            
        for name, p in self.module.named_parameters():
            if p.requires_grad == False or "prompt_learner.single_layer" in name:
                p.grad = None
