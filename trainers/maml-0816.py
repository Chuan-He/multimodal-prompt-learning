#!/usr/bin/env python3

import traceback
from torch.autograd import grad

from learn2learn.algorithms.base_learner import BaseLearner
from learn2learn.utils import clone_module, update_module

import torch
import torch.nn as nn

def adapt(module, loss_ce, loss_regt, loss_regi):
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
            grads_ce = grad_params_ce[counter]
            grads_regt = grad_params_regt[counter]
            grads_regi = grad_params_regi[counter]

            grads_ce_regt = grads_ce + grads_regt
            grads_ce_regt_norm = grads_ce_regt / (torch.linalg.norm(grads_ce_regt))
            grads_regi_norm = grads_regi / (torch.linalg.norm(grads_regi))
  
            angle = torch.dot(grads_ce_regt_norm.flatten(), grads_regi_norm.flatten())
            if angle > 0:
                diff_params[counter].grad = grads_ce + grads_regt
            else:
                diff_params[counter].grad = agreement_rand([grads_ce_regt,grads_regi])
            

        elif "prompt_learner.proj" in name or "prompt_learner.compound_prompt_projections" in name:
            grads_ce = grad_params_ce[counter]
            grads_regi = grad_params_regi[counter]
            grads_ce_norm = grads_ce / (torch.linalg.norm(grads_ce))
            grads_regi_norm = grads_regi / (torch.linalg.norm(grads_regi))
            angle = torch.dot(grads_ce_norm.flatten(), grads_regi_norm.flatten())
            if angle > 0:
                diff_params[counter].grad = grads_ce + grads_regi
            else:
                diff_params[counter].grad = grads_ce + grads_regi
        else:
            diff_params[counter].grad = None
        counter += 1

    for name, p in module.named_parameters():
        if p.requires_grad == False:
            p.grad = None

def get_grads(self):
    grads = []
    for p in self.model.parameters():
        if p.requires_grad:
            grads.append(p.grad.data.clone().flatten())
    return torch.cat(grads)
    
def agreement_rand(domain_grads):
    """ Agr-Rand consensus strategy. """

    # Compute agreement mask
    agr_mask = agreement_mask(domain_grads)

    # Sum components with same sign
    new_grads = torch.stack(domain_grads).sum(0)
    new_grads *= agr_mask

    # Get sample for components that do not agree
    sample = torch.randn((~agr_mask).sum(), device=new_grads.device, dtype=new_grads.dtype)
    scale = new_grads[agr_mask].abs().mean()
    # scale = new_grads.abs().mean()
    sample *= scale

    # Assign values to these components
    new_grads[~agr_mask] = sample

    return new_grads

def agreement_mask(domain_grads):
    """ Agreement mask. """

    grad_sign = torch.stack([torch.sign(g) for g in domain_grads])

    # True if all componentes agree, False if not
    agr_mask = torch.where(grad_sign.sum(0).abs() == len(domain_grads), 1, 0)

    return agr_mask.bool()