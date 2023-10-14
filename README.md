# Hypernetwork-based Meta-Learning for Low-Rank Physics-Informed Neural Networks

## Introduction

This is a pytorch implementation for our NeurIPS 2023 (spotlight) paper [Hypernetwork-based Meta-Learning for Low-Rank Physics-Informed Neural Networks](https://neurips.cc/virtual/2023/poster/70991)

## Experimental environment settings.

Run the following code before starting the experiment.

    conda env create -f env.yaml
    conda activate meta

## Data generation.

You can generate dataset for train / validation / test. 
Run code in the folder "data_gen".

    
         [ Code ]                   [ Description of code ]

    python gen_conv.py : Code for generating convection equation data
    python gen_diff.py : Code for generating diffusion equation data
    python gen_reac.py : Code for generating reaction equatinon data
    python gen_cd.py   : Code for generating Convection-Diffusion equation data
    python gen_rd.py   : Code for generating Reaction-Diffusion equation data
    python gen_cdr.py  : Code for generating Convection-Diffusion-Reaction data

Set the initial condition using "u0_str" parser. 
(you can select following option : sin_1, gauss, gauss_pi_2, etc...)


    [ u0_str ]

    sin_1       : 1+sin(x)
    gauss       : Gaussian distribution with STD=pi/4.
    gauss_pi_2  : Gaussian distribution with STD=pi/2.

## Train

Run the following code for Hyper-LR-PINN training / testing.

         [ Code ]              [ Description of code ]

    python train_meta.py    : Code for Hyper-LR-PINN training [phase1]
    python train_full.py    : Code for Hyper-LR-PINN (Full rank) training [phase2]
    python train_adap.py    : Code for Hyper-LR-PINN (Adaptive rank) training [phase2]

Detailed settings can be changed in config.py


For example, if you run the following code,
    python train_meta.py --epoch 20000 --pde_type convection --init_cond sin_1 --start_coeff_1 1 --end_coeff_1 20
    python train_adap.py --epoch 10000 --pde_type convection --init_cond sin_1 --start_coeff_1 1 --end_coeff_1 20 --target_coeff_1 10
    python train_full.py --epoch 10000 --pde_type convection --init_cond sin_1 --start_coeff_1 1 --end_coeff_1 20 --target_coeff_1 10

You can train/test the Hyper-LR-PINNs in the setting below.

 [ Experimental setting ] 

phase1 : 20000 epoch
phase2 : 10000 epoch
pde type : convection equation, 
initial condition : 1+sin(x), 
target equation : beta=10 (convection equation), 
meta-learning range : beta=[1, 20]

## Test

In additaon, we attach checkpoint of Hyper-LR-PINN (.pt file)
If you want to check it quickly, run the following code below.

         [ Code ]                   [ Description of code ]

    python test.py  : Code for testing Hyper-LR-PINN (Adaptive) (30~40 range, convection equation)

For example, if you run the following code,

    python test.py --pde_type convection --init_cond sin_1 --start_coeff_1 30 --end_coeff_1 40 --target_coeff_1 40

You can test the Hyper-LR-PINN (Adaptive rank) quickly. (beta=40)

## Other code

Brief description of the other code files.

        [ Code ]        [ Description of code ]
        
        model.py   :  Hyper-LR-PINN model. (phase1, phase2)
        utils.py   :  PDE residual loss
        
