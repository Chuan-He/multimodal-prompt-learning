#!/usr/bin/env python3

import traceback
from torch.autograd import grad

from learn2learn.algorithms.base_learner import BaseLearner
from learn2learn.utils import clone_module, update_module

import torch
import torch.nn as nn

class VNet(nn.Module):
    def __init__(self, cfg, device):
        super().__init__()
        self.inet = nn.ModuleList([nn.Sequential(nn.Linear(512*2, 512), nn.Sigmoid(), nn.Linear(512, 512)) for _ in range(cfg.TRAINER.KDPT_MAPLE_1_1.PROMPT_DEPTH)]).to(device)

    def adapt(self, module, loss_ce, loss_regt, loss_regi):
        first_order = True
        allow_unused = True

        second_order = not first_order

        # Compute relevant gradients
        diff_params = [p for p in module.parameters() if p.requires_grad]
        diff_names = [name for name, p in module.named_parameters() if p.requires_grad]

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
            if "prompt_learner.compound_prompts_text" in name or "prompt_learner.ctx" in name:
                if "prompt_learner.ctx" in name:
                    l_idx = 0
                elif "prompt_learner.compound_prompts_text" in name:
                    l_idx = int(name.split("prompt_learner.compound_prompts_text.")[1]) + 1
                grads_ce = grad_params_ce[counter].clone()
                grads_regt = grad_params_regt[counter].clone()
                grads_regi = grad_params_regi[counter].clone()
                grads_ce_regt = grads_ce + grads_regt
                #grads_ce_regt_norm = grads_ce_regt / (torch.linalg.norm(grads_ce_regt + 1e-8))
                #grads_regi_norm = grads_regi / (torch.linalg.norm(grads_regi + 1e-8))
                grads_ce_regt_norm = grads_ce_regt / (grads_ce_regt.norm(dim=-1, keepdim=True) + 1e-8)
                grads_regi_norm = grads_regi / (grads_regi.norm(dim=-1, keepdim=True) + 1e-8)
                # angle = torch.dot(grads_ce_regt_norm.flatten(), grads_regi_norm.flatten())
                input_grads = torch.cat([grads_ce_regt_norm, grads_regi_norm],dim=-1).float()
                output_grads = torch.sigmoid(self.inet[l_idx](input_grads)).half()
                
                diff_params[counter].grad = grads_ce + grads_regt + output_grads * grads_regi
            elif "prompt_learner.proj" in name or "prompt_learner.compound_prompt_projections" in name:
                grads_ce = grad_params_ce[counter].clone()
                grads_regi = grad_params_regi[counter].clone()
                diff_params[counter].grad = grads_ce + grads_regi
            else:
                diff_params[counter].grad = None
            counter += 1
 
        for name, p in module.named_parameters():
            if p.requires_grad == False:
                p.grad = None
