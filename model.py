from re import T
import torch
import torch.nn as nn
from torch.autograd import Variable
import pdb
from config import get_config


class LR_PINN_phase1(nn.Module):
    def __init__(self, hidden_dim):
        super(LR_PINN_phase1, self).__init__()

        self.start_layer = nn.Linear(2, hidden_dim)
        self.end_layer = nn.Linear(hidden_dim, 1)
        self.hidden_dim = hidden_dim
        self.scale = 1/hidden_dim
        
        self.col_basis_0 = nn.Parameter(self.scale * torch.rand(self.hidden_dim, self.hidden_dim))
        self.col_basis_1 = nn.Parameter(self.scale * torch.rand(self.hidden_dim, self.hidden_dim))
        self.col_basis_2 = nn.Parameter(self.scale * torch.rand(self.hidden_dim, self.hidden_dim))

        self.row_basis_0 = nn.Parameter(self.scale * torch.rand(self.hidden_dim, self.hidden_dim))
        self.row_basis_1 = nn.Parameter(self.scale * torch.rand(self.hidden_dim, self.hidden_dim))
        self.row_basis_2 = nn.Parameter(self.scale * torch.rand(self.hidden_dim, self.hidden_dim))
        
        self.meta_layer_1 = nn.Linear(3, self.hidden_dim)
        self.meta_layer_2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.meta_layer_3 = nn.Linear(self.hidden_dim, self.hidden_dim)
        
        self.meta_alpha_0 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.meta_alpha_1 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.meta_alpha_2 = nn.Linear(self.hidden_dim, self.hidden_dim)

        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        self.softplus = nn.Softplus()
        
    def forward(self, x, t, beta, nu, rho):
        ##### meta learning #####
        meta_input = torch.cat([beta, nu, rho], dim=1)
        meta_output = self.meta_layer_1(meta_input)
        meta_output = self.tanh(meta_output)
        meta_output = self.meta_layer_2(meta_output)
        meta_output = self.tanh(meta_output)
        meta_output = self.meta_layer_3(meta_output)
        meta_output = self.tanh(meta_output)

        meta_alpha_0_output = self.relu(self.meta_alpha_0(meta_output))
        meta_alpha_1_output = self.relu(self.meta_alpha_1(meta_output))
        meta_alpha_2_output = self.relu(self.meta_alpha_2(meta_output))

        alpha_0 = torch.diag_embed(meta_alpha_0_output)
        alpha_1 = torch.diag_embed(meta_alpha_1_output)
        alpha_2 = torch.diag_embed(meta_alpha_2_output)

        ##### main neural network #####
        inputs = torch.cat([x, t], axis=1)
        weight_0 = torch.matmul(torch.matmul(self.col_basis_0, alpha_0), self.row_basis_0)
        weight_1 = torch.matmul(torch.matmul(self.col_basis_1, alpha_1), self.row_basis_1)
        weight_2 = torch.matmul(torch.matmul(self.col_basis_2, alpha_2), self.row_basis_2)

        emb_out = self.start_layer(inputs)
        emb_out = self.tanh(emb_out)
        emb_out = emb_out.unsqueeze(dim=1)

        emb_out = torch.bmm(emb_out, weight_0)
        emb_out = self.tanh(emb_out)
        
        emb_out = torch.bmm(emb_out, weight_1)
        emb_out = self.tanh(emb_out)
        
        emb_out = torch.bmm(emb_out, weight_2)
        emb_out = self.tanh(emb_out)
        
        emb_out = self.end_layer(emb_out)
        emb_out = emb_out.squeeze(dim=1)
        
        return emb_out, self.col_basis_0, self.col_basis_1, self.col_basis_2, self.row_basis_0, self.row_basis_1, self.row_basis_2
    
    

class LR_PINN_phase2(nn.Module):
    def __init__(self, hidden_dim, start_w, start_b, end_w, end_b,
                 col_0, col_1, col_2, row_0, row_1, row_2, 
                 alpha_0, alpha_1, alpha_2):
        
        super(LR_PINN_phase2, self).__init__()

        self.start_layer = nn.Linear(2, hidden_dim)
        self.end_layer = nn.Linear(hidden_dim, 1)
        
        self.start_layer.weight = nn.Parameter(start_w)
        self.start_layer.bias = nn.Parameter(start_b)
        self.end_layer.weight = nn.Parameter(end_w)
        self.end_layer.bias = nn.Parameter(end_b)
        
        self.hidden_dim = hidden_dim
        self.scale = 1/hidden_dim
        
        self.col_basis_0 = nn.Parameter(col_0, requires_grad=False)
        self.col_basis_1 = nn.Parameter(col_1, requires_grad=False)
        self.col_basis_2 = nn.Parameter(col_2, requires_grad=False)

        self.row_basis_0 = nn.Parameter(row_0, requires_grad=False)
        self.row_basis_1 = nn.Parameter(row_1, requires_grad=False)
        self.row_basis_2 = nn.Parameter(row_2, requires_grad=False)
    
        self.alpha_0 = nn.Parameter(alpha_0)
        self.alpha_1 = nn.Parameter(alpha_1)
        self.alpha_2 = nn.Parameter(alpha_2)

        self.tanh = nn.Tanh()

    def forward(self, x, t):
        
        weight_0 = torch.matmul(torch.matmul(self.col_basis_0, torch.diag(self.alpha_0)), self.row_basis_0)
        weight_1 = torch.matmul(torch.matmul(self.col_basis_1, torch.diag(self.alpha_1)), self.row_basis_1)
        weight_2 = torch.matmul(torch.matmul(self.col_basis_2, torch.diag(self.alpha_2)), self.row_basis_2)

        ##### main neural network #####
        inputs = torch.cat([x, t], axis=1)
        emb_out = self.start_layer(inputs)
        emb_out = self.tanh(emb_out)
        
        emb_out = torch.matmul(emb_out, weight_0)
        emb_out = self.tanh(emb_out)
        
        emb_out = torch.matmul(emb_out, weight_1)
        emb_out = self.tanh(emb_out)
        
        emb_out = torch.matmul(emb_out, weight_2)
        emb_out = self.tanh(emb_out)
        
        emb_out = self.end_layer(emb_out)
        return emb_out