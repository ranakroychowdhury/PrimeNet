import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import argparse
import numpy as np
import torch
import torch.optim as optim
import sys
from random import SystemRandom
import utils

from timebert import TimeBERTForPretraining, TimeBERTConfig, TimeBERTForPretrainingV2

parser = argparse.ArgumentParser()
parser.add_argument('--niters', type=int, default=2000, help='Maximum number of iterations to run.')
parser.add_argument('--lr', type=float, default=0.01, help='Learning Rate.')
parser.add_argument('--rec-hidden', type=int, default=32, help='Model Hidden Size for Dense Layers.')
parser.add_argument('--embed-time', type=int, default=128, help='Size of Time Embedding Layer.')
parser.add_argument('--save', type=int, default=1)
parser.add_argument('--enc', type=str, default='mtan_enc')
parser.add_argument('--fname', type=str, default=None, help='Filename of pretrained checkpoint that may be loaded for further pretraining, if training stopped midway.')
parser.add_argument('--seed', type=int, default=0, help='Setting Random Seed.')
parser.add_argument('--split', type=int, default=0)
parser.add_argument('--n', type=int, default=8000)
parser.add_argument('--batch-size', type=int, default=50, help='Batch Size.')
parser.add_argument('--quantization', type=float, default=0.1, 
                    help='Quantization on the physionet dataset.')
parser.add_argument('--classif', action='store_true', 
                    help='Include binary classification loss')
parser.add_argument('--learn-emb', action='store_true', help='True: Use Learnable Time Embedding, linear layer for time embedding followed by sinusoidal activation. False: Fixed Positional Encoding.')
parser.add_argument('--num-heads', type=int, default=1, help='Number of Attention Heads.')
parser.add_argument('--freq', type=float, default=10., help='Positional Encoding Parameter.')
parser.add_argument('--dataset', type=str, default='physionet', help='Name of the Dataset.')
parser.add_argument('--old-split', type=int, default=1)
parser.add_argument('--nonormalize', action='store_true')
parser.add_argument('--classify-pertp', action='store_true')
parser.add_argument('--dev', type=str, default='0', help='GPU Device Number.')
parser.add_argument('--add_pos', action='store_true')
parser.add_argument('--transformer', action='store_true')
parser.add_argument('--pooling', type=str, default='bert', help='[ave, att, bert]: What pooling to use to aggregate the model output sequence representation for different tasks.')
parser.add_argument('--path', type=str, default='./data/pretrain/', help='Base path where all datasets are located.')
# parser.add_argument('--training', type=str, default='pretrain')
parser.add_argument('--pretrain_tasks', type=str, default='full2', help='[full, cl, interp, full2]: cl will only pretrain using TimeCL. interp will only pretrain using TimeReco. full2 will pretrain using both TimeCL and TimeReco.')
parser.add_argument('--patience', type=int, default=20, help='Early Stopping Criterion: How may iterations to wait for the validation accuracy at current epoch to exceed the best validation accuracy so far before early stopping training. Accuracy refers to Contrastive Learning Accuracy')
parser.add_argument('--segment_num', type=int, default=3, help='number of time interval segment to mask, default: 3 time intervals')
parser.add_argument('--mask_ratio_per_seg', type=float, default=0.05, help='fraction of the sequence length to mask for each time interval, deafult: 0.05 * seq_len to be masked for each of the time interval')
# parser.add_argument('--variable_name', type=str, default='segment_num', help='[pretrain_tasks, pooling, segment_num, mask_ratio_per_seg, lr]')

args = parser.parse_args()



def train(args, model, train_loader, optimizer):
    model.train()

    cl_loss_list = []
    mse_loss_list = []
    correct_list = []
    total_list = []
    for train_batch in train_loader:

        value_batch = train_batch['value'].to(args.device)
        time_batch = train_batch['time'].to(args.device)
        mask_batch = train_batch['mask'].to(args.device)

        # print(value_batch.shape, time_batch.shape, mask_batch.shape)
        x_batch = torch.cat([value_batch, mask_batch], dim=-1)
        
        out = model(x_batch, time_batch)

        cl_loss_list.append(out['cl_loss'])
        mse_loss_list.append(out['mse_loss'])
        correct_list.append(out['correct_num'])
        total_list.append(out['total_num'])

        loss = out['loss']
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    acc = sum(correct_list)/sum(total_list)
    cl_loss = sum(cl_loss_list)/len(cl_loss_list)
    mse_loss = sum(mse_loss_list)/len(mse_loss_list)

    return cl_loss, mse_loss, acc



def eval(args, model, eval_loader):
    model.eval()

    cl_loss_list = []
    mse_loss_list = []
    correct_list = []
    total_list = []

    with torch.no_grad():
        for train_batch in eval_loader:

            value_batch = train_batch['value'].to(args.device)
            time_batch = train_batch['time'].to(args.device)
            mask_batch = train_batch['mask'].to(args.device)

            x_batch = torch.cat([value_batch, mask_batch], dim=-1)
            
            out = model(x_batch, time_batch)

            cl_loss_list.append(out['cl_loss'])
            mse_loss_list.append(out['mse_loss'])
            correct_list.append(out['correct_num'])
            total_list.append(out['total_num'])

    acc = sum(correct_list)/sum(total_list)
    cl_loss = sum(cl_loss_list)/len(cl_loss_list)
    mse_loss = sum(mse_loss_list)/len(mse_loss_list)

    return cl_loss, mse_loss, acc



if __name__ == '__main__':
    args.path = './data/pretrain/'
    experiment_id = int(SystemRandom().random()*100000)
    print(args, experiment_id)
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)  
    gpu_id = 'cuda:' + args.dev  
    args.device = torch.device(#'cpu')
        gpu_id if torch.cuda.is_available() else 'cpu')

    '''
    if args.variable_name == 'pretrain_tasks': args.variable_value = str(args.pretrain_tasks)
    elif args.variable_name == 'pooling': args.variable_value = str(args.pooling)
    elif args.variable_name == 'segment_num': args.variable_value = str(args.segment_num)
    elif args.variable_name == 'mask_ratio_per_seg': args.variable_value = str(args.mask_ratio_per_seg)
    elif args.variable_name == 'lr': args.variable_value = str(args.lr)
    '''

    data_obj = utils.get_unlabeled_pretrain_data(args)
    train_loader = data_obj["train_dataloader"]
    val_loader = data_obj["val_dataloader"]
    dim = data_obj["input_dim"]
    max_len = data_obj["max_len"]
    max_len = max(max_len, 512)

    # model
    config = TimeBERTConfig(input_dim=dim,
                            pretrain_tasks=args.pretrain_tasks,
                            cls_query=torch.linspace(0, 1., 128),
                            hidden_size=args.rec_hidden,
                            embed_time=args.embed_time,
                            num_heads=args.num_heads,
                            learn_emb=args.learn_emb,
                            freq=args.freq,
                            pooling=args.pooling,
                            max_length=max_len,
                            dropout=0.3,
                            temp=0.05)
    if args.pretrain_tasks == 'full2':
        model = TimeBERTForPretrainingV2(config).to(args.device)
    else:
        model = TimeBERTForPretraining(config).to(args.device)

    params = (list(model.parameters()))
    print('parameters:', utils.count_parameters(model))
    optimizer = optim.Adam(params, lr=args.lr)
    
    if args.fname is not None:
        checkpoint = torch.load(args.fname)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print('loading saved weights', checkpoint['epoch'])

    best_acc = 0
    patience = args.patience

    results = []
    for itr in range(1, args.niters + 1):
        train_cl_loss, train_mse_loss, train_acc = train(args, model, train_loader, optimizer)
        val_cl_loss, val_mse_loss, val_acc = eval(args, model, val_loader)

        # if validation accuracy is less than best accuracy for patience number of epochs, stop training
        # save the model with best validation accuracy
        if args.pretrain_tasks == 'full' or args.pretrain_tasks == 'cl' or args.pretrain_tasks == 'full2':
            if val_acc > best_acc:
                torch.save({
                    'args': args,
                    'epoch': itr,
                    'model_state_dict': model.bert.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }, 'models/' + str(experiment_id) +
                    '.h5')
                
                best_acc = val_acc
                patience = args.patience
            else:
                patience-=1
                if patience < 0:
                    break

        results.append([train_cl_loss.item(), train_mse_loss.item(), train_acc, val_cl_loss.item(), val_mse_loss.item(), val_acc])

        sys.stdout.write('Iter: {}, train_cl_loss: {:.4f}, train_mse_loss: {:.4f}, train_acc: {:.4f}, dev_acc: {:.4f}, best_acc: {:.4f}\r'
              .format(itr, train_cl_loss, train_mse_loss, train_acc, val_acc, best_acc))
        sys.stdout.flush()

    if args.pretrain_tasks == 'interp':
        torch.save({
            'args': args,
            'epoch': itr,
            'model_state_dict': model.bert.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, 'models/' + str(experiment_id) +
                    '.h5')

    print('Training complete!')

    results_path = 'results/' + str(experiment_id) + '.npy'
    with open(results_path, 'wb') as f:
        np.save(f, np.array(results))
    print('Results saved!')