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
    
    epoch = args.epoch
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

    net_initial.load_state_dict(torch.load(f'./param/phase1/{pde_type}/{initial_condition}/PINN_{start_coeff_1}_{end_coeff_1}_20000.pt'))
    
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
    
    net = LR_PINN_phase2(hidden_dim, start_w, start_b, end_w, end_b, 
                          col_0, col_1, col_2, row_0, row_1, row_2, 
                          alpha_0, alpha_1, alpha_2)
    
    net = net.to(device)
    
    model_size = get_params(net)
    print(model_size)    
    
    optimizer = torch.optim.Adam(net.parameters(), lr=0.00025)   

    x_collocation = Variable(torch.from_numpy(np.array(np.expand_dims(train_data_f['x_data'], 1))).float(), requires_grad=True).to(device)
    t_collocation = Variable(torch.from_numpy(np.array(np.expand_dims(train_data_f['t_data'], 1))).float(), requires_grad=True).to(device)
    beta_collocation = Variable(torch.from_numpy(np.array(np.expand_dims(train_data_f['beta'], 1))).float(), requires_grad=True).to(device)
    nu_collocation = Variable(torch.from_numpy(np.array(np.expand_dims(train_data_f['nu'], 1))).float(), requires_grad=True).to(device)
    rho_collocation = Variable(torch.from_numpy(np.array(np.expand_dims(train_data_f['rho'], 1))).float(), requires_grad=True).to(device)

    all_zeros = np.zeros((len(train_data_f), 1))
    all_zeros = Variable(torch.from_numpy(all_zeros).float(), requires_grad=False).to(device)

    # initial points
    x_initial = Variable(torch.from_numpy(np.array(np.expand_dims(train_data_u['x_data'], 1))).float(), requires_grad=True).to(device)
    t_initial = Variable(torch.from_numpy(np.array(np.expand_dims(train_data_u['t_data'], 1))).float(), requires_grad=True).to(device)
    u_initial = Variable(torch.from_numpy(np.array(np.expand_dims(train_data_u['u_data'], 1))).float(), requires_grad=True).to(device)

    # boundary points (condition : upper bound = lower bound)
    x_lb = Variable(torch.from_numpy(np.array(np.expand_dims(train_data_bd['x_data_lb'], 1))).float(), requires_grad=True).to(device)
    t_lb = Variable(torch.from_numpy(np.array(np.expand_dims(train_data_bd['t_data_lb'], 1))).float(), requires_grad=True).to(device)
    x_ub = Variable(torch.from_numpy(np.array(np.expand_dims(train_data_bd['x_data_ub'], 1))).float(), requires_grad=True).to(device)
    t_ub = Variable(torch.from_numpy(np.array(np.expand_dims(train_data_bd['t_data_ub'], 1))).float(), requires_grad=True).to(device)
    
    # test point 
    x_test = Variable(torch.from_numpy(np.array(np.expand_dims(test_data['x_data'], 1))).float(), requires_grad=False).to(device)
    t_test = Variable(torch.from_numpy(np.array(np.expand_dims(test_data['t_data'], 1))).float(), requires_grad=False).to(device)
    u_test = Variable(torch.from_numpy(np.array(np.expand_dims(test_data['u_data'], 1))).float(), requires_grad=False).to(device)
    
    err_list = []
    ep_list = []
    loss_list= []
    mse_loss_list = []

    mse_u_list = []
    mse_f_list = []
    mse_bd_list = []

    L2_abs_list = []
    L2_rel_list = []
    Max_err_list = []
    Ex_var_score_list = []
    
    for ep in range(1, epoch+1):
        net.train()
        optimizer.zero_grad()
        
        net_initial_out = net(x_initial, t_initial)
        mse_u = mse_cost_function(net_initial_out, u_initial)
        
        f_out = f_cal_phase2(x_collocation, t_collocation, beta_collocation, nu_collocation, rho_collocation, net)
        mse_f = mse_cost_function(f_out, all_zeros)
        
        u_pred_lb = net(x_lb, t_lb)
        u_pred_ub = net(x_ub, t_ub)
        

        mse_bd = torch.mean((u_pred_lb - u_pred_ub) ** 2)
        
        loss = mse_u + mse_f + mse_bd 

        loss.backward()
        optimizer.step()
        
        if ep % 10 == 0:
            net.eval()
            with torch.autograd.no_grad():
                u_out_test = net(x_test, t_test)
                mse_test = mse_cost_function(u_out_test, u_test)
                
                err_list.append(mse_test.item())
                ep_list.append(ep)
                loss_list.append(loss.item())
                mse_loss_list.append((mse_u+mse_f+mse_bd).item())

                mse_u_list.append(mse_u.item())
                mse_f_list.append(mse_f.item())
                mse_bd_list.append(mse_bd.item())

                L2_error_norm = torch.linalg.norm(u_out_test-u_test, 2, dim = 0)
                L2_true_norm = torch.linalg.norm(u_test, 2, dim = 0)
                
                L2_absolute_error = torch.mean(torch.abs(u_out_test-u_test))
                L2_relative_error = L2_error_norm / L2_true_norm

                u_test_cpu = u_test.cpu()
                u_out_test_cpu = u_out_test.cpu()
                Max_err = max_error(u_test_cpu, u_out_test_cpu)
                Ex_var_score = explained_variance_score(u_test_cpu, u_out_test_cpu)

                L2_abs_list.append(L2_absolute_error.item())
                L2_rel_list.append(L2_relative_error.item())
                Max_err_list.append(Max_err)
                Ex_var_score_list.append(Ex_var_score)
                
                print('L2_abs_err :', L2_absolute_error.item())
                print('L2_rel_err :', L2_relative_error.item())
                print('Max_err :', Max_err)
                print('Ex_var_score :', Ex_var_score)
                print('Epoch :', ep, 'Error :', mse_test.item(), 'train_loss (total) :', loss.item())    
                print('mse_f :', mse_f.item(), 'mse_u :', mse_u.item(), 'mse_bd :', mse_bd.item())
                print('#########################################################################################')

        if (ep+1) % 1000 == 0:
            SAVE_PATH = f'./param/phase2/{pde_type}/{initial_condition}'
            SAVE_NAME = f'PINN_{start_coeff_1}_{end_coeff_1}_{int(target_coeff_1.item())}_{ep+1}.pt'
            
            if not os.path.isdir(SAVE_PATH): os.mkdir(SAVE_PATH)
            torch.save(net.state_dict(), SAVE_PATH + "/" + SAVE_NAME)


    err_df = pd.DataFrame(err_list)
    ep_df = pd.DataFrame(ep_list)
    loss_df = pd.DataFrame(loss_list)

    mse_loss_df = pd.DataFrame(mse_loss_list)

    mse_u_df = pd.DataFrame(mse_u_list)
    mse_f_df = pd.DataFrame(mse_f_list)
    mse_bd_df = pd.DataFrame(mse_bd_list)
    
    L2_abs_df = pd.DataFrame(L2_abs_list)
    L2_rel_df = pd.DataFrame(L2_rel_list)
    Max_err_df = pd.DataFrame(Max_err_list)
    Ex_var_score_df = pd.DataFrame(Ex_var_score_list)

    log_data = pd.concat([ep_df, loss_df, err_df, mse_loss_df, mse_u_df, mse_f_df, mse_bd_df, L2_abs_df, L2_rel_df, Max_err_df, Ex_var_score_df], axis=1)
    log_data.columns = ["epoch", "train_loss", "test_err", "mse_loss", "mse_u", "mse_f", "mse_bd", "L2_abs_err", "L2_rel_err", "Max_err", "Ex_var_score"]    

    log_path = f'./log/phase2/{pde_type}/{initial_condition}'
    log_name = f'PINN_{start_coeff_1}_{end_coeff_1}_{int(target_coeff_1.item())}_{epoch}.csv'

    if not os.path.isdir(log_path): 
        os.mkdir(log_path)
        
    log_data.to_csv(log_path+"/"+log_name, index=False)

    print('#### final ####')
    print('L2_abs_err :', L2_absolute_error.item())
    print('L2_rel_err :', L2_relative_error.item())

    print('Epoch :', ep, 'Error :', mse_test.item(), 'train_loss (total) :', loss.item())    
    print('mse_f :', mse_f.item(), 'mse_u :', mse_u.item(), 'mse_bd :', mse_bd.item())
    print('#########################################################################################')


if __name__ == "__main__":
    main()
