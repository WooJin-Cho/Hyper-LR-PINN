import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from config import get_config
import torch
import random
import torch.backends.cudnn as cudnn
import pandas as pd
from model import LR_PINN_phase1, LR_PINN_phase2
from utils import orthogonality_reg, f_cal_phase2, get_params
import os
from sklearn.metrics import explained_variance_score, max_error

args = get_config()
device = torch.device(args.device)

def main():
    args = get_config()
    
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    cudnn.benchmark = False
    cudnn.deterministic = True
    random.seed(args.seed)
    
    device = torch.device(args.device)
    print("========================================")
    print("Use Device :", device)
    print("Available cuda devices :", torch.cuda.device_count())
    print("Current cuda device :", torch.cuda.current_device())
    print("Name of cuda device :", torch.cuda.get_device_name(device))
    print("========================================")

    hidden_dim = 50
    
    pde_type = args.pde_type
    initial_condition = args.init_cond
    
    start_coeff_1 = args.start_coeff_1
    end_coeff_1 = args.end_coeff_1

    target_coeff_1 = args.target_coeff_1
    target_coeff_2 = args.target_coeff_2
    target_coeff_3 = args.target_coeff_3
    
    ###################### Dataset #######################
    train_data_f    = pd.read_csv(f'./data_gen/dataset/{pde_type}/train/train_f_{target_coeff_1}_{pde_type}.csv')
    train_data_u    = pd.read_csv(f'./data_gen/dataset/{pde_type}/train/train_u_{target_coeff_1}_{pde_type}.csv')
    train_data_bd   = pd.read_csv(f'./data_gen/dataset/{pde_type}/train/train_boundary_{target_coeff_1}_{pde_type}.csv')
    test_data       = pd.read_csv(f'./data_gen/dataset/{pde_type}/test/test_{target_coeff_1}_{pde_type}.csv')
    ######################################################

    target_coeff_1 = torch.tensor(target_coeff_1).unsqueeze(dim=0)
    target_coeff_1 = target_coeff_1.type(torch.float)

    target_coeff_2 = torch.tensor(target_coeff_2).unsqueeze(dim=0)
    target_coeff_2 = target_coeff_2.type(torch.float)
    
    target_coeff_3 = torch.tensor(target_coeff_3).unsqueeze(dim=0)
    target_coeff_3 = target_coeff_3.type(torch.float)  
    
    mse_cost_function = torch.nn.MSELoss() # Mean squared error

    ############### Network Initialization ################
    net_initial = LR_PINN_phase1(hidden_dim)

    net_initial.load_state_dict(torch.load(f'./param/phase1/{pde_type}/{initial_condition}/PINN_{start_coeff_1}_{end_coeff_1}.pt'))
    
    tanh = nn.Tanh()
    relu = nn.ReLU()
    
    start_w = net_initial.state_dict()['start_layer.weight']
    start_b = net_initial.state_dict()['start_layer.bias']
    end_w = net_initial.state_dict()['end_layer.weight']
    end_b = net_initial.state_dict()['end_layer.bias']
    
    col_0 = net_initial.state_dict()['col_basis_0']
    col_1 = net_initial.state_dict()['col_basis_1']
    col_2 = net_initial.state_dict()['col_basis_2']
    row_0 = net_initial.state_dict()['row_basis_0']
    row_1 = net_initial.state_dict()['row_basis_1']
    row_2 = net_initial.state_dict()['row_basis_2']
    
    meta_layer_1_w = net_initial.state_dict()['meta_layer_1.weight']
    meta_layer_1_b = net_initial.state_dict()['meta_layer_1.bias']
    meta_layer_2_w = net_initial.state_dict()['meta_layer_2.weight']
    meta_layer_2_b = net_initial.state_dict()['meta_layer_2.bias']
    meta_layer_3_w = net_initial.state_dict()['meta_layer_3.weight']
    meta_layer_3_b = net_initial.state_dict()['meta_layer_3.bias']    
    
    meta_alpha_0_w = net_initial.state_dict()['meta_alpha_0.weight']
    meta_alpha_0_b = net_initial.state_dict()['meta_alpha_0.bias']
    meta_alpha_1_w = net_initial.state_dict()['meta_alpha_1.weight']
    meta_alpha_1_b = net_initial.state_dict()['meta_alpha_1.bias']
    meta_alpha_2_w = net_initial.state_dict()['meta_alpha_2.weight']
    meta_alpha_2_b = net_initial.state_dict()['meta_alpha_2.bias']
    
    target_coeff = torch.cat([target_coeff_1, target_coeff_2, target_coeff_3], dim=0)
    meta_vector = torch.matmul(target_coeff, meta_layer_1_w.T) + meta_layer_1_b
    meta_vector = tanh(meta_vector)
    
    meta_vector = torch.matmul(meta_vector, meta_layer_2_w.T) + meta_layer_2_b
    meta_vector = tanh(meta_vector)

    meta_vector = torch.matmul(meta_vector, meta_layer_3_w.T) + meta_layer_3_b
    meta_vector = tanh(meta_vector)

    alpha_0 = relu(torch.matmul(meta_vector, meta_alpha_0_w.T) + meta_alpha_0_b)
    alpha_1 = relu(torch.matmul(meta_vector, meta_alpha_1_w.T) + meta_alpha_1_b)
    alpha_2 = relu(torch.matmul(meta_vector, meta_alpha_2_w.T) + meta_alpha_2_b)
    ########################################################
    
    alpha_0_nonzero_index = torch.nonzero(alpha_0).squeeze()
    alpha_1_nonzero_index = torch.nonzero(alpha_1).squeeze()
    alpha_2_nonzero_index = torch.nonzero(alpha_2).squeeze()
    
    
    print(f'LR_layer_1 (rank) : {len(alpha_0_nonzero_index)}')
    print(f'LR_layer_2 (rank) : {len(alpha_1_nonzero_index)}')
    print(f'LR_layer_3 (rank) : {len(alpha_2_nonzero_index)}')
    
    
    alpha_0 = torch.gather(input=alpha_0, dim=0, index=alpha_0_nonzero_index)
    alpha_1 = torch.gather(input=alpha_1, dim=0, index=alpha_1_nonzero_index)
    alpha_2 = torch.gather(input=alpha_2, dim=0, index=alpha_2_nonzero_index)
    
    
    col_0 = torch.index_select(input=col_0, dim=1, index=alpha_0_nonzero_index)        
    col_1 = torch.index_select(input=col_1, dim=1, index=alpha_1_nonzero_index)        
    col_2 = torch.index_select(input=col_2, dim=1, index=alpha_2_nonzero_index)        

    row_0 = torch.index_select(input=row_0, dim=0, index=alpha_0_nonzero_index)        
    row_1 = torch.index_select(input=row_1, dim=0, index=alpha_1_nonzero_index)        
    row_2 = torch.index_select(input=row_2, dim=0, index=alpha_2_nonzero_index)        
    
    net = LR_PINN_phase2(hidden_dim, start_w, start_b, end_w, end_b, 
                          col_0, col_1, col_2, row_0, row_1, row_2, 
                          alpha_0, alpha_1, alpha_2)
    
    net = net.to(device)
    net.load_state_dict(torch.load(f'./param/phase2/{pde_type}/{initial_condition}/PINN_{start_coeff_1}_{end_coeff_1}_{int(target_coeff_1.item())}.pt'))
    model_size = get_params(net)
    print('Number of model parameters :', model_size)    
    
    # test point 
    x_test = Variable(torch.from_numpy(np.array(np.expand_dims(test_data['x_data'], 1))).float(), requires_grad=False).to(device)
    t_test = Variable(torch.from_numpy(np.array(np.expand_dims(test_data['t_data'], 1))).float(), requires_grad=False).to(device)
    u_test = Variable(torch.from_numpy(np.array(np.expand_dims(test_data['u_data'], 1))).float(), requires_grad=False).to(device)

    
    with torch.autograd.no_grad():
        net.eval()
        u_out_test = net(x_test, t_test)
        
        L2_error_norm = torch.linalg.norm(u_out_test-u_test, 2, dim = 0)
        L2_true_norm = torch.linalg.norm(u_test, 2, dim = 0)
        
        L2_absolute_error = torch.mean(torch.abs(u_out_test-u_test))
        L2_relative_error = L2_error_norm / L2_true_norm

        u_test_cpu = u_test.cpu()
        u_out_test_cpu = u_out_test.cpu()
        Max_err = max_error(u_test_cpu, u_out_test_cpu)
        Ex_var_score = explained_variance_score(u_test_cpu, u_out_test_cpu)

        print('L2_abs_err :', L2_absolute_error.item())
        print('L2_rel_err :', L2_relative_error.item())
        print('Max_err :', Max_err)
        print('Ex_var_score :', Ex_var_score)
        print('#########################################################################################')


if __name__ == "__main__":
    main()

