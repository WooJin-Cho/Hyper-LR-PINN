import torch
import torch.nn as nn
import numpy as np
from shutil import copyfile
from collections import Counter
import pandas as pd
from torch.autograd import Variable
from config import get_config

args = get_config()
device = torch.device(args.device)

def orthogonality_reg(col, row, rank):
    col_reg = torch.matmul(col, torch.transpose(col, 0, 1)) - torch.eye(rank).to(device)
    row_reg = torch.matmul(row, torch.transpose(row, 0, 1)) - torch.eye(rank).to(device)
    reg_loss = (torch.norm(col_reg ,p='fro') + torch.norm(row_reg, p='fro'))/(rank*rank)
    return reg_loss


def get_alphas(coeff, meta_layer_1_w, meta_layer_1_b, meta_layer_2_w, meta_layer_2_b, meta_layer_3_w, meta_layer_3_b,
               meta_alpha_0_w, meta_alpha_0_b, meta_alpha_1_w, meta_alpha_1_b, meta_alpha_2_w, meta_alpha_2_b):

    tanh = nn.Tanh()
    relu = nn.ReLU()
    # coeff = torch.tensor(coeff)
    coeff = coeff.type(torch.float)

    meta_vector = torch.matmul(coeff, meta_layer_1_w.T) + meta_layer_1_b
    meta_vector = tanh(meta_vector)
    meta_vector = torch.matmul(meta_vector, meta_layer_2_w.T) + meta_layer_2_b
    meta_vector = tanh(meta_vector)
    meta_vector = torch.matmul(meta_vector, meta_layer_3_w.T) + meta_layer_3_b
    meta_vector = tanh(meta_vector)

    alpha_0 = relu(torch.matmul(meta_vector, meta_alpha_0_w.T) + meta_alpha_0_b)
    alpha_1 = relu(torch.matmul(meta_vector, meta_alpha_1_w.T) + meta_alpha_1_b)
    alpha_2 = relu(torch.matmul(meta_vector, meta_alpha_2_w.T) + meta_alpha_2_b)

    alpha_0 = alpha_0.numpy()
    alpha_1 = alpha_1.numpy()
    alpha_2 = alpha_2.numpy()
    
    return alpha_0, alpha_1, alpha_2


def get_params(model):
    pp = 0
    for p in list(model.parameters()):
        if p.requires_grad == True:
            nn = 1
            for s in list(p.size()):
                nn = nn * s
            pp += nn
    return pp


def f_cal(x, t, beta, nu, rho, net, hidden_dim):
    u, col_0_f, col_1_f, col_2_f, row_0_f, row_1_f, row_2_f = net(x, t, beta, nu, rho)
    u_x = torch.autograd.grad(u.sum(), x, create_graph=True)[0]
    u_t = torch.autograd.grad(u.sum(), t, create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x.sum(), x, create_graph=True)[0]

    reg_f_0 = orthogonality_reg(col_0_f, row_0_f, hidden_dim)
    reg_f_1 = orthogonality_reg(col_1_f, row_1_f, hidden_dim)
    reg_f_2 = orthogonality_reg(col_2_f, row_2_f, hidden_dim)
    reg_f = reg_f_0 + reg_f_1 + reg_f_2

    pde = (beta * u_x) - (nu * u_xx) - (rho * u * (1-u)) + u_t
    return pde, reg_f


def f_cal_phase2(x, t, beta, nu, rho, net):
    
    u = net(x, t)
    u_x = torch.autograd.grad(u.sum(), x, create_graph=True)[0]
    u_t = torch.autograd.grad(u.sum(), t, create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x.sum(), x, create_graph=True)[0]

    pde = (beta * u_x) - (nu * u_xx) - (rho * u * (1-u)) + u_t
    
    return pde