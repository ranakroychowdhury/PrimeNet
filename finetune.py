import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import time
from random import SystemRandom
import utils
import math

from timebert import TimeBERTForClassification, TimeBERTForRegression, TimeBERTForInterpolation, TimeBERTConfig

parser = argparse.ArgumentParser()
parser.add_argument('--niters', type=int, default=2000, help='Maximum number of iterations to run.')
parser.add_argument('--lr', type=float, default=0.01, help='Learning Rate.')
parser.add_argument('--rec-hidden', type=int, default=32, help='Model Hidden Size for Dense Layers.')
parser.add_argument('--embed-time', type=int, default=128, help='Size of Time Embedding Layer.')
parser.add_argument('--save', type=int, default=0, help='Non-zero: Save the finetuned model. Zero: Do not save the finetuned model.')
parser.add_argument('--fname', type=str, default=None, help='Filename of pretrained checkpoint.')
parser.add_argument('--seed', type=int, default=0, help='Setting Random Seed.')
parser.add_argument('--split', type=int, default=0)
parser.add_argument('--n', type=int, default=8000)
parser.add_argument('--batch-size', type=int, default=50, help='Batch Size.')
parser.add_argument('--quantization', type=float, default=0.1, 
                    help="Quantization on the physionet dataset.")
parser.add_argument('--classif', action='store_true', 
                    help="Include binary classification loss")
parser.add_argument('--learn-emb', action='store_true', help='True: Use Learnable Time Embedding, linear layer for time embedding followed by sinusoidal activation. False: Fixed Positional Encoding.')
parser.add_argument('--num-heads', type=int, default=1, help='Number of Attention Heads.')
parser.add_argument('--freq', type=float, default=10., help='Positional Encoding Parameter.')
parser.add_argument('--dataset', type=str, default='physionet', help='Name of the Dataset.')
parser.add_argument('--old-split', type=int, default=1)
parser.add_argument('--nonormalize', action='store_true')
parser.add_argument('--classify-pertp', action='store_true', help='Whether to do a per timestep classification.')

parser.add_argument('--dev', type=str, default='0', help='GPU Device Number.')
parser.add_argument('--task', type=str, default='classification', help='[classification, regression, interpolation]: Name of the Finetuning Task')
parser.add_argument('--pooling', type=str, default='bert', help='[ave, att, bert]: What pooling to use to aggregate the model output sequence representation for different tasks.')
parser.add_argument('--pretrain_model', type=str, default='0.15', help='[full, full2, cl, interp, att, bert, 0.15]')
parser.add_argument('--path', type=str, default='./data/finetune/', help='Base path where all datasets are located.')
args = parser.parse_args()


if __name__ == '__main__':
    args.path = './data/finetune/'
    all_mse_loss, all_mae_loss, best_mse_epochs = [], [], []
    all_best_auc, all_best_acc, all_lowest_loss_auc, all_lowest_loss_acc = [], [], [], []

    experiment_id = int(SystemRandom().random()*100000)
    # print(args, experiment_id)
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)  
    gpu_id = 'cuda:' + args.dev  
    args.device = torch.device(#'cpu')
        gpu_id if torch.cuda.is_available() else 'cpu')
        
    data_obj = utils.get_finetune_data(args)
    train_loader = data_obj["train_dataloader"]
    test_loader = data_obj["test_dataloader"]
    val_loader = data_obj["val_dataloader"]
    dim = data_obj["input_dim"]

    # model
    
    config = TimeBERTConfig(dataset=args.dataset,
                            input_dim=dim,
                            cls_query=torch.linspace(0, 1., 128),
                            hidden_size=args.rec_hidden,
                            embed_time=args.embed_time,
                            num_heads=args.num_heads,
                            learn_emb=args.learn_emb,
                            freq=args.freq,
                            pooling=args.pooling,
                            classify_pertp=args.classify_pertp,
                            max_length=512,
                            dropout=0.3,
                            temp=0.05)
    
    if args.task == 'classification':
        model = TimeBERTForClassification(config).to(args.device)
    elif args.task == 'regression':
        model = TimeBERTForRegression(config).to(args.device)
    elif args.task == 'interpolation':
        model = TimeBERTForInterpolation(config).to(args.device)

    if args.pretrain_model is not None:
        print('Pretrained Model: ' + args.pretrain_model)
        model.bert.load_state_dict(torch.load('models/' + args.pretrain_model + '.h5')['model_state_dict'])
        print('Load successfully.')
    else:
        print('Model training from scratch')

    params = (list(model.parameters()))
    print('parameters:', utils.count_parameters(model))
    optimizer = optim.Adam(params, lr=args.lr)

    if args.task == 'classification':
        criterion = nn.CrossEntropyLoss()
    elif args.task == 'regression' or args.task == 'interpolation':
        criterion = nn.MSELoss()
    
    if args.fname is not None:
        checkpoint = torch.load(args.fname)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print('loading saved weights', checkpoint['epoch'])
    
    best_val_loss = float('inf')
    total_time = 0.
    best_mse_loss = float('inf')
    best_mae_loss = float('inf')
    best_mse_epoch = 0
    results = []

    best_acc = 0.
    best_auc = 0.
    lowest_loss_acc = 0.
    lowest_loss_auc = 0.

    for itr in range(1, args.niters + 1):
        train_loss = 0
        train_n = 0
        train_acc = 0
        total_values = 0
        #avg_reconst, avg_kl, mse = 0, 0, 0
        start_time = time.time()
        for train_batch, label in train_loader:
            train_batch, label = train_batch.to(args.device), label.to(args.device)
            batch_len  = train_batch.shape[0]
            observed_data, observed_mask, observed_tp \
                = train_batch[:, :, :dim], train_batch[:, :, dim:2*dim], train_batch[:, :, -1]
            out = model(torch.cat((observed_data, observed_mask), 2), observed_tp)
            if args.task == 'classification' and args.classify_pertp:
                N = label.size(-1)
                out = out.view(-1, N)
                label = label.view(-1, N)
                _, label = label.max(-1)
                loss = criterion(out, label.long())
            else:
                if args.task == 'classification': loss = criterion(out, label)
                elif args.task == 'regression': loss = criterion(out[ : , 0], label)
                elif args.task == 'interpolation':
                    target_data, target_mask = label[:, :, :dim], label[:, :, dim:2*dim].bool()
                    num_values = torch.sum(target_mask).item()
                    loss = criterion(out[target_mask], target_data[target_mask])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if args.task == 'classification':
                train_loss += loss.item() * batch_len
                train_acc += torch.mean((out.argmax(1) == label).float()).item() * batch_len
                train_n += batch_len
            elif args.task == 'regression':
                train_loss += loss.item() * batch_len
                train_n += batch_len
            elif args.task == 'interpolation':
                train_loss += loss.item() * num_values
                total_values += num_values
        total_time += time.time() - start_time


        if args.task == 'classification':
            val_loss, val_acc, val_auc = utils.evaluate_classifier(model, val_loader, args=args, dim=dim)
            test_loss, test_acc, test_auc = utils.evaluate_classifier(model, test_loader, args=args, dim=dim)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                lowest_loss_acc = test_acc
                lowest_loss_auc = test_auc


        elif args.task == 'regression':
            val_mse_loss, val_mae_loss = utils.evaluate_regressor(model, val_loader, args=args, dim=dim)
            test_mse_loss, test_mae_loss = utils.evaluate_regressor(model, test_loader, args=args, dim=dim)
            best_val_loss = min(best_val_loss, val_mse_loss)


        elif args.task == 'interpolation':
            val_mse_loss, val_mae_loss = utils.evaluate_interpolator(model, val_loader, args=args, dim=dim)
            test_mse_loss, test_mae_loss = utils.evaluate_interpolator(model, test_loader, args=args, dim=dim)
            best_val_loss = min(best_val_loss, val_mse_loss)
        
        
        if args.task == 'classification':
            results.append([train_loss/train_n, train_acc/train_n, val_loss, val_acc, val_auc, test_loss, test_acc, test_auc])

            print('Iter: {}, loss: {:.4f}, acc: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}, test_acc: {:.4f}, test_auc: {:.4f}'
                .format(itr, train_loss/train_n, train_acc/train_n, val_loss, val_acc, test_acc, test_auc), end = '\r')
            
            best_acc = max(test_acc, best_acc)
            best_auc = max(test_auc, best_auc)


        elif args.task == 'regression':
            results.append([train_loss/train_n, val_mse_loss, val_mae_loss, test_mse_loss, test_mae_loss])

            print('Iter: {}, train_loss: {:.6f}, val_mse_loss: {:.6f}, val_mae_loss: {:.6f}, test_mse_loss: {:.6f}, test_mae_loss: {:.6f}'
                .format(itr, train_loss/train_n, val_mse_loss, val_mae_loss, test_mse_loss, test_mae_loss), end = '\r')
            
            if test_mse_loss < best_mse_loss:
                best_mse_loss = min(test_mse_loss, best_mse_loss)
                best_mse_epoch = itr
            best_mae_loss = min(test_mae_loss, best_mae_loss)


        elif args.task == 'interpolation':
            results.append([train_loss/total_values, val_mse_loss, val_mae_loss, test_mse_loss, test_mae_loss])

            print('Iter: {}, train_loss: {:.6f}, val_mse_loss: {:.6f}, val_mae_loss: {:.6f}, test_mse_loss: {:.6f}, test_mae_loss: {:.6f}'
                .format(itr, train_loss/total_values, val_mse_loss, val_mae_loss, test_mse_loss, test_mae_loss), end = '\r')
            
            if test_mse_loss < best_mse_loss:
                best_mse_loss = min(test_mse_loss, best_mse_loss)
                best_mse_epoch = itr
            best_mae_loss = min(test_mae_loss, best_mae_loss)
        

        if itr % 100 == 0 and args.save:
            torch.save({
                'args': args,
                'epoch': itr,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': -loss,
            }, 'models/' + args.pretrain_model + '_finetuned.h5')
    

    if args.task == 'classification':
        print('Best ACC:', best_acc)
        print('Best AUC:', best_auc)
        print('Lowest Loss ACC:', lowest_loss_acc)
        print('Lowest Loss AUC:', lowest_loss_auc)

        all_best_acc.append(best_acc)
        all_best_auc.append(best_auc)
        all_lowest_loss_acc.append(lowest_loss_acc)
        all_lowest_loss_auc.append(lowest_loss_auc)


    elif args.task == 'regression' or args.task == 'interpolation':
        print('Best MSE Loss:', best_mse_loss)
        print('Best MAE Loss:', best_mae_loss)

        all_mse_loss.append(best_mse_loss)
        all_mae_loss.append(best_mae_loss)

        best_mse_epochs.append(best_mse_epoch)


    results_path = 'results/' + args.pretrain_model + '_finetuned.npy'
    with open(results_path, 'wb') as f:
        np.save(f, np.array(results))
        

if args.task == 'classification':
    all_best_acc_round = [round(num, 3) for num in all_best_acc]
    print('Best Accuracy: ' + str(all_best_acc_round))

    all_best_auc_round = [round(num, 3) for num in all_best_auc]
    print('Best AUC: ' + str(all_best_auc_round))

    all_lowest_loss_acc_round = [round(num, 3) for num in all_lowest_loss_acc]
    print('Lowest Loss Accuracy: ' + str(all_lowest_loss_acc_round))

    all_lowest_loss_auc_round = [round(num, 3) for num in all_lowest_loss_auc]
    print('Lowest Loss AUC: ' + str(all_lowest_loss_auc_round))

    print('Mean Best Acc, Std Best Acc: ' + str(np.mean(all_best_acc)) + ', ' + str(np.std(all_best_acc)))
    print('Mean Best Auc, Std Best Auc: ' + str(np.mean(all_best_auc)) + ', ' + str(np.std(all_best_auc)))

    print('Mean Lowest Loss Acc, Std Lowest Loss Acc: ' + str(np.mean(all_lowest_loss_acc)) + ', ' + str(np.std(all_lowest_loss_acc)))
    print('Mean Lowest Loss Auc, Std Lowest Loss Auc: ' + str(np.mean(all_lowest_loss_auc)) + ', ' + str(np.std(all_lowest_loss_auc)))


elif args.task == 'regression' or args.task == 'interpolation':
    all_rmse_loss = [math.sqrt(num) for num in all_mse_loss]
    print('MSE Loss: ' + str(all_mse_loss))
    print('MAE Loss: ' + str(all_mae_loss))

    print('Best MSE epochs: ' + str(best_mse_epochs))
    print('Mean MSE Loss, Std MSE Loss: ' + str(np.mean(all_mse_loss)) + ', ' + str(np.std(all_mse_loss)))
    print('Mean RMSE Loss, Std RMSE Loss: ' + str(np.mean(all_rmse_loss)) + ', ' + str(np.std(all_rmse_loss)))
    print('Mean MAE Loss, Std MAE Loss: ' + str(np.mean(all_mae_loss)) + ', ' + str(np.std(all_mae_loss)))