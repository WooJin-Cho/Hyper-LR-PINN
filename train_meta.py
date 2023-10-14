import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from config import get_config
import torch
import random
import torch.backends.cudnn as cudnn
import pandas as pd
from model import LR_PINN_phase1
from utils import orthogonality_reg, f_cal, get_params
import os
from sklearn.metrics import explained_variance_score, max_error

args = get_config()
device = torch.device(args.device)

def main():
    args=get_config()

    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    np.random.seed(0)
    cudnn.benchmark = False
    cudnn.deterministic = True
    random.seed(0)
    
    device = torch.device(args.device)

    print("=============[Deivce Info]==============")
    print("- Use Device :", device)
    print("- Available cuda devices :", torch.cuda.device_count())
    print("- Current cuda device :", torch.cuda.current_device())
    print("- Name of cuda device :", torch.cuda.get_device_name(device))
    print("========================================\n")

    mse_cost_function = torch.nn.MSELoss() # Mean squared error
    hidden_dim = 50

    net = LR_PINN_phase1(hidden_dim)
    net = net.to(device)

    model_size = get_params(net)
    
    #######################################
    ############   argparser   ############
    epoch = args.epoch

    initial_condition = args.init_cond
    pde_type = args.pde_type  
    
    start_coeff_1 = args.start_coeff_1
    end_coeff_1 = args.end_coeff_1
    
    #######################################
    #######################################

    train_data_f    = pd.read_csv(f'./data_gen/dataset/{pde_type}/train/train_f_{start_coeff_1}_{pde_type}.csv')
    train_data_u    = pd.read_csv(f'./data_gen/dataset/{pde_type}/train/train_u_{start_coeff_1}_{pde_type}.csv')
    train_data_bd   = pd.read_csv(f'./data_gen/dataset/{pde_type}/train/train_boundary_{start_coeff_1}_{pde_type}.csv')
    test_data       = pd.read_csv(f'./data_gen/dataset/{pde_type}/test/test_{start_coeff_1}_{pde_type}.csv')

    for i in range(start_coeff_1, end_coeff_1):
        f_sample    = pd.read_csv(f'./data_gen/dataset/{pde_type}/train/train_f_{i+1}_{pde_type}.csv')
        u_sample    = pd.read_csv(f'./data_gen/dataset/{pde_type}/train/train_u_{i+1}_{pde_type}.csv')
        bd_sample   = pd.read_csv(f'./data_gen/dataset/{pde_type}/train/train_boundary_{i+1}_{pde_type}.csv')
        test_sample = pd.read_csv(f'./data_gen/dataset/{pde_type}/test/test_{i+1}_{pde_type}.csv')

        train_data_f    = pd.concat([train_data_f, f_sample], ignore_index = True)
        train_data_u    = pd.concat([train_data_u, u_sample], ignore_index = True)
        train_data_bd   = pd.concat([train_data_bd, bd_sample], ignore_index = True)
        test_data       = pd.concat([test_data, test_sample], ignore_index = True)

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
    beta_initial = Variable(torch.from_numpy(np.array(np.expand_dims(train_data_u['beta'], 1))).float(), requires_grad=True).to(device)
    nu_initial = Variable(torch.from_numpy(np.array(np.expand_dims(train_data_u['nu'], 1))).float(), requires_grad=True).to(device)
    rho_initial = Variable(torch.from_numpy(np.array(np.expand_dims(train_data_u['rho'], 1))).float(), requires_grad=True).to(device)
    

    # boundary points (condition : upper bound = lower bound)
    x_lb = Variable(torch.from_numpy(np.array(np.expand_dims(train_data_bd['x_data_lb'], 1))).float(), requires_grad=True).to(device)
    t_lb = Variable(torch.from_numpy(np.array(np.expand_dims(train_data_bd['t_data_lb'], 1))).float(), requires_grad=True).to(device)
    x_ub = Variable(torch.from_numpy(np.array(np.expand_dims(train_data_bd['x_data_ub'], 1))).float(), requires_grad=True).to(device)
    t_ub = Variable(torch.from_numpy(np.array(np.expand_dims(train_data_bd['t_data_ub'], 1))).float(), requires_grad=True).to(device)
    beta_bd = Variable(torch.from_numpy(np.array(np.expand_dims(train_data_bd['beta'], 1))).float(), requires_grad=True).to(device)
    nu_bd = Variable(torch.from_numpy(np.array(np.expand_dims(train_data_bd['nu'], 1))).float(), requires_grad=True).to(device)
    rho_bd = Variable(torch.from_numpy(np.array(np.expand_dims(train_data_bd['rho'], 1))).float(), requires_grad=True).to(device)

    
    # test point 
    x_test = Variable(torch.from_numpy(np.array(np.expand_dims(test_data['x_data'], 1))).float(), requires_grad=False).to(device)
    t_test = Variable(torch.from_numpy(np.array(np.expand_dims(test_data['t_data'], 1))).float(), requires_grad=False).to(device)
    u_test = Variable(torch.from_numpy(np.array(np.expand_dims(test_data['u_data'], 1))).float(), requires_grad=False).to(device)
    beta_test = Variable(torch.from_numpy(np.array(np.expand_dims(test_data['beta'], 1))).float(), requires_grad=False).to(device)
    nu_test = Variable(torch.from_numpy(np.array(np.expand_dims(test_data['nu'], 1))).float(), requires_grad=False).to(device)
    rho_test = Variable(torch.from_numpy(np.array(np.expand_dims(test_data['rho'], 1))).float(), requires_grad=False).to(device)
         
    print("=============[Train Info]===============")
    print(f"- PDE type : {pde_type}")
    print(f"- Initial condition : {initial_condition}")
    print(f"- start_coeff_1 ~ end_coeff_1 :{start_coeff_1} ~ {end_coeff_1}")
    print(f"- Model size : {model_size}")
    print("========================================\n")

    print("=============[Model Info]===============\n")
    print(net)
    print("========================================\n")
    
    optimizer = torch.optim.Adam(net.parameters(), lr=0.00025)
    
    err_list = []
    ep_list = []
    loss_list= []

    mse_loss_list = []
    reg_loss_list = []

    mse_u_list = []
    mse_f_list = []
    mse_bd_list = []
    
    reg_u_list = []
    reg_f_list = []
    reg_bd_list = []

    L2_abs_list = []
    L2_rel_list = []
    Max_err_list = []
    Ex_var_score_list = []
    
    for ep in range(1, epoch+1):
        net.train()
        optimizer.zero_grad()
        
        net_initial_out, col_0_init, col_1_init, col_2_init, row_0_init, row_1_init, row_2_init = net(x_initial, t_initial, beta_initial, nu_initial, rho_initial)

        reg_init_0 = orthogonality_reg(col_0_init, row_0_init, hidden_dim)
        reg_init_1 = orthogonality_reg(col_1_init, row_1_init, hidden_dim)
        reg_init_2 = orthogonality_reg(col_2_init, row_2_init, hidden_dim)

        reg_init = reg_init_0 + reg_init_1 + reg_init_2
        mse_u = mse_cost_function(net_initial_out, u_initial)
        
        f_out, reg_f = f_cal(x_collocation, t_collocation, beta_collocation, nu_collocation, rho_collocation, net, hidden_dim)
        mse_f = mse_cost_function(f_out, all_zeros)
        
        u_pred_lb, col_0_lb, col_1_lb, col_2_lb, row_0_lb, row_1_lb, row_2_lb = net(x_lb, t_lb, beta_bd, nu_bd, rho_bd)
        u_pred_ub, col_0_ub, col_1_ub, col_2_ub, row_0_ub, row_1_ub, row_2_ub = net(x_ub, t_ub, beta_bd, nu_bd, rho_bd)
        
        reg_lb_0 = orthogonality_reg(col_0_lb, row_0_lb, hidden_dim)
        reg_lb_1 = orthogonality_reg(col_1_lb, row_1_lb, hidden_dim)
        reg_lb_2 = orthogonality_reg(col_2_lb, row_2_lb, hidden_dim)
        reg_ub_0 = orthogonality_reg(col_0_ub, row_0_ub, hidden_dim)
        reg_ub_1 = orthogonality_reg(col_1_ub, row_1_ub, hidden_dim)
        reg_ub_2 = orthogonality_reg(col_2_ub, row_2_ub, hidden_dim)

        reg_bd = reg_lb_0 + reg_lb_1 + reg_lb_2 + reg_ub_0 + reg_ub_1 + reg_ub_2

        mse_bd = torch.mean((u_pred_lb - u_pred_ub) ** 2)
                
        loss = mse_u + mse_f + mse_bd + reg_init + reg_f + reg_bd

        loss.backward()
        optimizer.step()
        
        if ep % 10 == 0:
            net.eval()
            with torch.autograd.no_grad():
                u_out_test, _, _, _, _, _, _ = net(x_test, t_test, beta_test, nu_test, rho_test)
                # test_loss_f = f(x_test, t_test, coefficient, net)
                mse_test = mse_cost_function(u_out_test, u_test)
                err_list.append(mse_test.item())
                ep_list.append(ep)
                loss_list.append(loss.item())
                
                mse_loss_list.append((mse_u+mse_f+mse_bd).item())
                reg_loss_list.append((reg_init+reg_f+reg_bd).item())
                
                mse_u_list.append(mse_u.item())
                mse_f_list.append(mse_f.item())
                mse_bd_list.append(mse_bd.item())
                
                reg_u_list.append(reg_init.item())
                reg_f_list.append(reg_f.item())
                reg_bd_list.append(reg_bd.item())
                
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
            SAVE_PATH = f'./param/phase1/{pde_type}/{initial_condition}'
            SAVE_NAME = f'PINN_{start_coeff_1}_{end_coeff_1}_{ep+1}.pt'
            
            if not os.path.isdir(SAVE_PATH): os.mkdir(SAVE_PATH)
            torch.save(net.state_dict(), SAVE_PATH + "/" + SAVE_NAME)

    err_df = pd.DataFrame(err_list)
    ep_df = pd.DataFrame(ep_list)
    loss_df = pd.DataFrame(loss_list)
    
    mse_loss_df = pd.DataFrame(mse_loss_list)
    reg_loss_df = pd.DataFrame(reg_loss_list)
    
    mse_u_df = pd.DataFrame(mse_u_list)
    mse_f_df = pd.DataFrame(mse_f_list)
    mse_bd_df = pd.DataFrame(mse_bd_list)
    
    reg_u_df = pd.DataFrame(reg_u_list)
    reg_f_df = pd.DataFrame(reg_f_list)
    reg_bd_df = pd.DataFrame(reg_bd_list)
    
    L2_abs_df = pd.DataFrame(L2_abs_list)
    L2_rel_df = pd.DataFrame(L2_rel_list)
    Max_err_df = pd.DataFrame(Max_err_list)
    Ex_var_score_df = pd.DataFrame(Ex_var_score_list)
    
    log_data = pd.concat([ep_df, loss_df, err_df, mse_loss_df, reg_loss_df, mse_u_df, mse_f_df, mse_bd_df, reg_u_df, reg_f_df, reg_bd_df, L2_abs_df, L2_rel_df, Max_err_df, Ex_var_score_df], axis=1)
    log_data.columns = ["epoch", "train_loss", "test_err", "mse_loss", "reg_loss", "mse_u", "mse_f", "mse_bd", "reg_u", "reg_f", "reg_bd", "L2_abs_err", "L2_rel_err", "Max_err", "Ex_var_score"]    

    log_path = f'./log/phase1/{pde_type}/{initial_condition}'
    log_name = f'PINN_{start_coeff_1}_{end_coeff_1}.csv'

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
