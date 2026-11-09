import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import time, random
from tqdm import tqdm
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler, QuantileTransformer
import datetime
import argparse
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms, utils, datasets
from torch.utils.data import Dataset, DataLoader

from ember_utils import *
from ember_model import *
from ember_pjr_utils import *


def get_only_av_class_labeled_samples(all_X, all_Y, Y_Family):
    
    new_X = []
    new_Y = []
    new_Y_family = []
    
    for ind, y in enumerate(all_Y):
        if y == 1:
            if Y_Family[ind] != '':
                new_Y.append(y)
                new_X.append(all_X[ind])
                new_Y_family.append(Y_Family[ind])
        else:
            new_Y.append(y)
            new_X.append(all_X[ind])
            new_Y_family.append(Y_Family[ind])
            
            
    print(len(all_Y==1) == (len(new_Y) + (len(all_Y) - len(new_Y))))  
    print(len(new_X) == len(new_Y))
    return new_X, new_Y, new_Y_family

def get_family_labeled_month_data(data_dir, month, train=True):
    
    if train:
        data_dir = data_dir + str(month) + '/'
        XY_train = np.load(data_dir + 'XY_train.npz')
        X_tr, Y_tr, Y_tr_family = XY_train['X_train'], XY_train['Y_train'], XY_train['Y_family_train']

        #print(f'X_train {X_tr.shape} Y_train {Y_tr.shape} Y_tr_family {Y_tr_family.shape}')
        
        return X_tr, Y_tr, Y_tr_family
    else:
        data_dir = data_dir + str(month) + '/'
        XY_test = np.load(data_dir + 'XY_test.npz')
        X_test, Y_test, Y_test_family = XY_test['X_test'], XY_test['Y_test'], XY_test['Y_family_test']

        return X_test, Y_test, Y_test_family

    
    
def get_family_labeled_task_test_data(data_dir, task_months, mlp_net=False):
    
    X_te, Y_te, Y_te_family = get_family_labeled_month_data(data_dir, task_months[-1], train=False)
    
    for month in task_months[:-1]:
        pre_X_te, pre_Y_te, pre_Y_te_family = get_family_labeled_month_data(data_dir, month, train=False)
        X_te, Y_te, Y_te_family = np.concatenate((X_te, pre_X_te)), np.concatenate((Y_te, pre_Y_te)),\
                                np.concatenate((Y_te_family, pre_Y_te_family))
        

    X_test, Y_test, Y_test_family = X_te, Y_te, Y_te_family
    
    #print(f' X_test {X_test.shape} Y_test {Y_test.shape} Y_te_family {Y_test_family.shape}')
    
    return X_test, Y_test, Y_test_family



def get_month_data(data_dir, month, train=True):
    
    if train:
        data_dir = data_dir + str(month) + '/'
        XY_train = np.load(data_dir + 'XY_train.npz')
        X_tr, Y_tr = XY_train['X_train'], XY_train['Y_train']
        
        return X_tr, Y_tr
    else:
        data_dir = data_dir + str(month) + '/'
        XY_test = np.load(data_dir + 'XY_test.npz')
        X_test, Y_test = XY_test['X_test'], XY_test['Y_test']

        return X_test, Y_test 

    
    
def get_MemoryData(X, Y, memory_budget):
    indx = [i for i in range(len(Y))]
    random.shuffle(indx)
    
    replay_index = indx[:memory_budget]
    X_train = X[replay_index]
    Y_train = Y[replay_index]
    
    return X_train, Y_train



def get_GRS_data(data_dir, task_months, memory_budget, train=True, joint=True):
    
    if train:
        X_tr, Y_tr = get_month_data(data_dir, task_months[-1])
        #print(f'Current Task month {task_months[-1]} data X {X_tr.shape} Y {Y_tr.shape}')
        
        if len(task_months) != 1:
            previous_Xs, previous_Ys = [], []
            for month in task_months[:-1]:
                #pre_X_tr, pre_Y_tr, pre_X_val, pre_Y_val = get_month_data(data_dir, month)
                pre_X_tr, pre_Y_tr = get_month_data(data_dir, month)


                pre_X_tr, pre_Y_tr = np.array(pre_X_tr), np.array(pre_Y_tr)
                #print(f'pre_X_tr {pre_X_tr.shape}  pre_Y_tr {pre_Y_tr.shape}')

                for idx, prevSample in enumerate(pre_X_tr):
                    previous_Xs.append(prevSample)
                    previous_Ys.append(pre_Y_tr[idx])


            previous_Xs, previous_Ys = np.array(previous_Xs), np.array(previous_Ys)  

            #print(f'Y_tr {Y_tr.shape}  previous_Ys {previous_Ys.shape}')
            if joint:
                X_tr, Y_tr = np.concatenate((X_tr, previous_Xs)), np.concatenate((Y_tr, previous_Ys))
            else:
                if memory_budget >= len(previous_Ys):
                    X_tr, Y_tr = np.concatenate((X_tr, previous_Xs)), np.concatenate((Y_tr, previous_Ys))
                else:
                    previous_Xs, previous_Ys = get_MemoryData(previous_Xs, previous_Ys, memory_budget)
                    X_tr, Y_tr = np.concatenate((X_tr, previous_Xs)), np.concatenate((Y_tr, previous_Ys))
                    #print(f'memory_budget {memory_budget}  Y_tr {previous_Ys.shape} ')
                    assert memory_budget == len(previous_Ys)
            

        X_train, Y_train  = X_tr, Y_tr
        print(f'X_train {X_train.shape} Y_train {Y_train.shape}\n')
        return X_train, Y_train
    else:
        X_te, Y_te = get_month_data(data_dir, task_months[-1], train=False)
        for month in task_months[:-1]:
            pre_X_te, pre_Y_te = get_month_data(data_dir, month, train=False)
            
            X_te, Y_te = np.concatenate((X_te, pre_X_te)), np.concatenate((Y_te, pre_Y_te))

        X_test, Y_test  = X_te, Y_te
        print(f'X_test {X_test.shape} Y_test {Y_test.shape}')
        return X_test, Y_test





parser = argparse.ArgumentParser()
parser.add_argument('--num_exps', type=int, default=1, required=False, help='Number of Experiments to Run.')
parser.add_argument('--num_epoch', type=int, default=100, required=False)
parser.add_argument('--batch_size', type=int, default=6000, required=False)
parser.add_argument('--grs_joint',  action="store_true", required=False)
parser.add_argument('--memory_budget', type=int, required=False)
parser.add_argument('--data_dir', type=str,\
                    default='../../../month_based_processing_with_family_labels/', required=False)
parser.add_argument('--max_tasks', type=int, default=12, required=False,
                    help='SMOKE: only run the first K monthly tasks.')
parser.add_argument('--seed', type=int, default=42, required=False,
                    help='SMOKE: fixed RNG seed (official code used random seeds).')
parser.add_argument('--out_dir', type=str, default='./smoke_out', required=False,
                    help='SMOKE: redirect all model/result writes here (official code uses ../GRS_SavedModel etc.).')

args = parser.parse_args()


all_task_months = ['2018-01', '2018-02', '2018-03', '2018-04', '2018-05', '2018-06',
                   '2018-07', '2018-08', '2018-09', '2018-10', '2018-11', '2018-12']

data_dir = args.data_dir
num_exps = args.num_exps
num_epoch = args.num_epoch
batch_size = args.batch_size
memory_budget = args.memory_budget
patience = 5



if args.grs_joint:
    memory_budget = 'joint'
    
    
replay_type = 'joint_partial'

exp_seeds = [args.seed + i for i in range(num_exps)]  # SMOKE: fixed seed; official used random.randint
task_month_range = range(min(len(all_task_months), args.max_tasks))  # SMOKE: 2-task smoke


def testing_ap(model, X_test, Y_test, batch_size, device):
    """SMOKE: average-precision + ROC-AUC from raw sigmoid outputs (official
    testing_aucscore only reports metrics on rounded predictions)."""
    from sklearn.metrics import average_precision_score, roc_auc_score
    model.eval()
    Xt = torch.from_numpy(np.array(X_test, np.float32)).type(torch.FloatTensor)
    yt = torch.from_numpy(np.array(Y_test)).type(torch.FloatTensor)
    dl = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(Xt, yt),
                                     batch_size=batch_size, shuffle=False)
    scores, trues = [], []
    with torch.no_grad():
        for xb, yb in dl:
            xb = xb.to(device)
            scores += model(xb).squeeze(1).cpu().tolist()
            trues += yb.cpu().tolist()
    scores, trues = np.array(scores), np.array(trues)
    return float(average_precision_score(trues, scores)), float(roc_auc_score(trues, scores))




cnt =  1    
for exp in exp_seeds:
    start_time = time.time()
    use_cuda = True
    print('Torch', torch.__version__, 'CUDA', torch.version.cuda)
    use_cuda = use_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    torch.manual_seed(exp)

    random.seed(exp); np.random.seed(exp)  # SMOKE: seed stdlib/numpy RNGs too (official left them unseeded)
    model = Ember_MLP_Net()
    #optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=0.000001)
       
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)
    
    model = model.to(device)
    print(f'Model has {count_parameters(model)/1000000}m parameters')    
    criterion = nn.BCELoss()    
    
    
    standardization = StandardScaler()
    standard_scaler = None
    for task_month in task_month_range:
                
        print(f'\n{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")} Round {cnt} ...')
        task_start = time.time()
        current_task = all_task_months[task_month]
        task_months = all_task_months[:task_month+1]
        print(f'Current Task {current_task} with Budget {memory_budget}')


        model_save_dir = args.out_dir + '/models/GRSModel_' + str(memory_budget) + '/' + str(current_task) + '/'
        create_parent_folder(model_save_dir)

        opt_save_path = args.out_dir + '/models/GRSOpt_' + str(memory_budget) + '/' + str(current_task) + '/'
        create_parent_folder(opt_save_path)

        results_save_dir = args.out_dir + '/results_meta/GRS_' + str(memory_budget) + '/'
        create_parent_folder(results_save_dir)
        
        
        if args.grs_joint:
            X_train, Y_train = get_GRS_data(data_dir, task_months, memory_budget, train=True, joint=True)
            X_test, Y_test = get_GRS_data(data_dir, task_months, memory_budget, train=False, joint=True)
        else:
            X_train, Y_train = get_GRS_data(data_dir, task_months, memory_budget, train=True, joint=False)
            X_test, Y_test = get_GRS_data(data_dir, task_months, memory_budget, train=False, joint=False)
    
        
        # to debug
        #X_train = X_train[:500]
        #Y_train = Y_train [:500]
        #X_valid, Y_valid = X_valid[:500], Y_valid[:500]
        
        print(f'{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")} Standardizing ...')
        
        if args.grs_joint:
                standardization = StandardScaler()
                standard_scaler = None
                standard_scaler = standardization.fit(X_train)
        else:        
                standard_scaler = standardization.partial_fit(X_train)

        X_train = standard_scaler.transform(X_train)
        #X_valid = standard_scaler.transform(X_valid)
        X_test = standard_scaler.transform(X_test)


        X_train, Y_train = np.array(X_train, np.float32), np.array(Y_train, np.int32)
        #X_valid, Y_valid = np.array(X_valid, np.float32), np.array(Y_valid, np.int32)
        X_test, Y_test = np.array(X_test, np.float32), np.array(Y_test, np.int32)        
        
        print(f'{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")} Training ...')
        
        
        
        task_training_time, epoch_ran, training_loss, validation_loss  = training_early_stopping(\
                             model, model_save_dir, opt_save_path, X_train, Y_train,\
                             X_test, Y_test, patience, batch_size, device, optimizer, num_epoch,\
                             criterion, replay_type, current_task, exp, earlystopping=True)

        
        #model = Ember_MLP_Net()
        #model = model.to(device)
        best_model_path = model_save_dir + os.listdir(model_save_dir)[0]
        print(f'loading best model {best_model_path}')
        model.load_state_dict(torch.load(best_model_path))
        
        #optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.000001)
        best_optimizer = opt_save_path + os.listdir(opt_save_path)[0]
        print(f'loading best optimizer {best_optimizer}')
        optimizer.load_state_dict(torch.load(best_optimizer))
        
        
        
        
        acc, precision, recall, f1score = testing_aucscore(model, X_test, Y_test, batch_size, device)
        ap_score, roc_auc = testing_ap(model, X_test, Y_test, batch_size, device)
        if use_cuda:
            peak_vram_mb = float(torch.cuda.max_memory_allocated(device) / 1024**2)
        else:
            peak_vram_mb = None


        end_time = time.time()

        print(f'Elapsed time {(end_time - start_time)/60} mins.')


        task_end = time.time()
        task_run_time = (task_end - task_start)/60

        import resource, json
        peak_rss_mb = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)

        os.makedirs(args.out_dir + '/Results_Domain', exist_ok=True)  # SMOKE: official assumed ./Results_Domain exists
        results_f = open(os.path.join(args.out_dir + '/Results_Domain/', 'grs_' + str(memory_budget) + '_results.txt'), 'a')
        result_string = '{}\t{}\t{}\t{}\t{}\t\n'.format(current_task, acc, precision, recall, f1score)
        
        results_f.write(result_string)
        results_f.flush()
        results_f.close()

        # SMOKE: per-task machine-readable record (official only wrote the .txt line above)
        smoke_rec = {
            'method': 'joint' if args.grs_joint else ('none' if str(memory_budget) == '0' else 'grs'),
            'memory_budget': str(memory_budget), 'task': current_task, 'seed': exp,
            'num_epoch': num_epoch, 'batch_size': batch_size,
            'task_wallclock_min': float(task_run_time),
            'train_time_min': float(task_training_time),
            'peak_rss_mb': peak_rss_mb, 'peak_vram_mb': peak_vram_mb,
            'cuda_available': bool(use_cuda),
            'acc': float(acc), 'precision': float(precision),
            'recall': float(recall), 'f1': float(f1score),
            'ap': ap_score, 'roc_auc': roc_auc,
            'n_train': int(len(Y_train)), 'n_test': int(len(Y_test)),
        }
        with open(os.path.join(args.out_dir, f"smoke_grs_{memory_budget}_{current_task}_seed{exp}.json"), 'w') as jf:
            json.dump(smoke_rec, jf, indent=2)

    end_time = time.time()
    cnt += 1
    print(f'Elapsed time {(end_time - start_time)/60} mins.')
    
    del model_save_dir
    del opt_save_path
    del results_save_dir