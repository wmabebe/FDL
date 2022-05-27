#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6


import os
import copy
import time
import pickle
import numpy as np
from tqdm import tqdm
import random
from node import *
import datetime
import matplotlib
import matplotlib.pyplot as plt

import torch
from tensorboardX import SummaryWriter

from options import args_parser
from update import LocalUpdate, test_inference, ripple_updates
from models import MLP, CNNMnist, CNNFashion_Mnist, CNNCifar, vgg11
from utils import get_dataset, exp_details, average_gradients
from node import color_graph


if __name__ == '__main__':
    start_time = time.time()

    # define paths
    path_project = os.path.abspath('..')
    logger = SummaryWriter('../logs')

    args = args_parser()
    exp_details(args)
    print("args: ", args)

    # if args.gpu_id:
    #     torch.cuda.set_device(args.gpu_id)
    #device = 'cuda' if args.gpu != None and int(args.gpu) != 0 else 'cpu'

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device: ",device)

    # load dataset and user groups
    train_dataset, test_dataset, user_groups = get_dataset(args)

    MAX_PEERS = args.max_peers
    NODES = args.num_users
    OPPOSIT_FRAC = args.opposit_frac
    NOW = str(datetime.datetime.now()).replace(" ","--")
    OPPOSIT_STRATEGY = args.opposit_strategy
    IID = "IID" if args.iid == 1 else "NON-IID"

    #Create output directory
    dir_path = './save/{}_{}_{}_{}_nodes[{}]_maxpeers[{}]_rounds[{}]_noniidfrac[{}]_strategy[{}]_frac[{}]_local_ep[{}]_local_bs[{}]_attck_frac[{}]/'. \
        format(NOW,args.dataset, args.model, IID, NODES, MAX_PEERS, args.epochs,OPPOSIT_FRAC,OPPOSIT_STRATEGY, args.frac,args.local_ep, args.local_bs,args.attack_frac)
    os.makedirs(os.path.dirname(dir_path), exist_ok=True)

    #Initialize the p2p graph
    adj_list = [Node(i,None,None,MAX_PEERS) for i in range(NODES)]

    for idx,node in enumerate(adj_list):
        #Assign node data idxs
        node.data = user_groups[idx]
        #Add random neighbors
        while len(node.neighbors) < MAX_PEERS:
            node.add_neighbors(random.sample(adj_list,MAX_PEERS))
            if node in node.neighbors:
                node.neighbors = []

    #Randomly set attackers
    attackers = []
    if float(args.attack_frac) > 0:
        #Make train_dataset mutable for pertrubation
        for i in range(len(train_dataset)):
            train_dataset[i] = list(train_dataset[i])
        #Randomly select malicous users
        num_attackers = int(args.attack_frac * args.num_users)
        attackers = list(np.random.choice(range(args.num_users), num_attackers, replace=False))
    print ("Attackers: ", attackers)

    # BUILD MODEL
    if args.model == 'cnn':
        # Convolutional neural netork
        if args.dataset == 'mnist':
            for node in adj_list:
                node.model = CNNMnist(args=args)
        elif args.dataset == 'fmnist':
            for node in adj_list:
                node.model = CNNFashion_Mnist(args=args)
        elif args.dataset == 'cifar':
            for node in adj_list:
                node.model = CNNCifar(args=args)

    elif args.model == 'mlp':
        # Multi-layer preceptron
        img_size = train_dataset[0][0].shape
        len_in = 1
        for x in img_size:
            len_in *= x
            for node in adj_list:
                node.model = MLP(dim_in=len_in, dim_hidden=64,
                               dim_out=args.num_classes)
    elif args.model == 'vgg':
        if args.dataset == 'cifar':
            for node in adj_list:
                node.model = vgg11()
        else:
            exit('Error: We only use VGG models for cifar dataset')
    else:
        exit('Error: unrecognized model')

    # Set models to train and send to device.
    for node in adj_list:
        node.model.to(device)
        node.model.train()
        #print(node)

    # Training
    train_loss, train_accuracy = [], []
    val_acc_list, net_list = [], []
    cv_loss, cv_acc = [], []
    print_every = 2
    val_loss_pre, counter = 0, 0
    total_grads = []

    for epoch in tqdm(range(args.epochs)):
        local_weights, local_losses, local_grads = [], [], []
        print(f'\n | Global Training Round : {epoch+1} |\n')

        #global_model.train()
        m = max(int(args.frac * args.num_users), 1)
        idxs_nodes = np.random.choice(range(NODES), m, replace=False)

        for idx in idxs_nodes:
            node = adj_list[idx]
            malicous = True if idx in attackers else False
            local_model = LocalUpdate(args=args, dataset=train_dataset,
                                      idxs=node.data, logger=logger,attacker=malicous)
            w, loss, grads = local_model.update_weights(
                model=node.model, global_round=epoch)
            local_weights.append(copy.deepcopy(w))
            local_losses.append(copy.deepcopy(loss))
            #Append current user grads to local grads together with the label
            local_grads.append(copy.deepcopy(grads))

        #Check if local_grads works
        # total_grads.append(copy.deepcopy(local_grads))
        # sample_idx  = list(local_grads.keys())[0]
        # print("1 device grad size: ", len( local_grads[sample_idx] ))
        # print("Round", epoch," grads: ", len(local_grads))
        # print("Total grads: ", len(total_grads))

        # update global weights
        #global_weights = average_weights(local_weights)

        # update global weights
        # global_model.load_state_dict(global_weights)

        #Compute average global gradient for computing node colors/similarity
        global_gradient = average_gradients(local_grads)

        colors = color_graph(adj_list,global_gradient)
        
        #Update p2p nodes
        ripple_updates(adj_list,epoch,colors,dir_path,OPPOSIT_FRAC,OPPOSIT_STRATEGY)

        loss_avg = sum(local_losses) / len(local_losses)
        train_loss.append(loss_avg)

        # Calculate avg training accuracy over all non-malicous users at every epoch
        list_acc, list_loss = [], []
        
        for c in range(NODES):
            node = adj_list[c]
            node.model.eval()
            if c not in attackers:
                local_model = LocalUpdate(args=args, dataset=train_dataset,
                                        idxs=node.data, logger=logger)
                acc, loss = local_model.inference(model=node.model)
                list_acc.append(acc)
                list_loss.append(loss)
        train_accuracy.append(sum(list_acc)/len(list_acc))

        # print global training loss after every 'i' rounds
        if (epoch+1) % print_every == 0:
            print(f' \nAvg Training Stats after {epoch+1} global rounds:')
            print(f'Training Loss : {np.mean(np.array(train_loss))}')
            print('Train Accuracy: {:.2f}% \n'.format(100*train_accuracy[-1]))

    # Test inference after completion of training
    avg_test_acc, avg_test_loss = 0,0
    for node in adj_list:
        test_acc, test_loss = test_inference(args, node.model, test_dataset)
        avg_test_acc += test_acc
        avg_test_loss += test_loss
    avg_test_acc /= len(adj_list)
    avg_test_loss /= len(adj_list)

    summary = "Results after " + str(args.epochs) + " global rounds of training:" + \
             "\n|---- Avg Train Accuracy: {:.2f}%".format(100*train_accuracy[-1]) + \
            "\n|---- Avg Test Accuracy: {:.2f}%".format(100*avg_test_acc)
    
    print(summary)

    #Write summary to file
    with open(dir_path + "summary.txt","w") as f:
        f.write(summary)

    # Saving the objects train_loss and train_accuracy:
    with open(dir_path + "train-loss-accuracy.pkl", 'wb') as f:
        pickle.dump([train_loss, train_accuracy], f)
    
    # with open(dir_path + "train-loss-accuracy.pkl", 'wb') as f:
    #     pickle.dump(total_grads,f)

    print('\n Total Run Time: {0:0.4f}'.format(time.time()-start_time))

    # PLOTTING (optional)
    matplotlib.use('Agg')

    # Plot Loss curve
    plt.figure()
    plt.title('Training Loss vs Communication rounds')
    plt.plot(range(len(train_loss)), train_loss, color='r')
    plt.ylabel('Training loss')
    plt.xlabel('Communication Rounds')
    plt.savefig(dir_path + "loss.png")
    
    # Plot Average Accuracy vs Communication rounds
    plt.figure()
    plt.title('Average Accuracy vs Communication rounds')
    plt.plot(range(len(train_accuracy)), train_accuracy, color='k')
    plt.ylabel('Average Accuracy')
    plt.xlabel('Communication Rounds')
    plt.savefig(dir_path + "acc.png")
