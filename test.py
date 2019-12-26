import time
import os
import math
import argparse
from glob import glob
from collections import OrderedDict
import random
import warnings
from datetime import datetime
import pdb

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
import joblib

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn import metrics
from skimage.io import imread

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import torchvision
from torchvision import datasets, models, transforms

from lib.dataset import Dataset
from lib.models.model_factory import get_model, get_gcn_model
from lib.utils import *
from lib.metrics import *
from lib.losses import *
from lib.preprocess import preprocess

from lib.kmeans.kmeans import lloyd

os.environ["CUDA_VISIBLE_DEVICES"] = "3"

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', default=None,
                        help='model name: (default: arch+timestamp)')

    args = parser.parse_args()

    return args

def main():
    test_args = parse_args()

    args = joblib.load('models/%s/args.pkl' %test_args.name)

    print('Config -----')
    for arg in vars(args):
        print('%s: %s' % (arg, getattr(args, arg)))
    print('------------')

    if args.pred_type == 'classification':
        num_outputs = 5
    elif args.pred_type == 'regression':
        num_outputs = 1
    elif args.pred_type == 'multitask':
        num_outputs = 6
    else:
        raise NotImplementedError

    # cudnn.benchmark = True

    cudnn.deterministic = True

    test_transform = transforms.Compose([
        transforms.Resize((args.input_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    # # data loading code
    test_dir = preprocess(
        'aptos2019',
        args.img_size,
        scale=args.scale_radius,
        norm=args.normalize,
        pad=args.padding,
        remove=args.remove)
    test_df = pd.read_csv('inputs/train.csv')
    test_img_paths = test_dir + '/' + test_df['id_code'].values + '.png'
    test_labels = test_df['diagnosis'].values

    # diabetic_retinopathy_dir = preprocess(
    #     'diabetic_retinopathy',
    #     args.img_size,
    #     scale=args.scale_radius,
    #     norm=args.normalize,
    #     pad=args.padding,
    #     remove=args.remove)
    # diabetic_retinopathy_df = pd.read_csv('inputs/diabetic-retinopathy-resized/trainLabels.csv')
    # diabetic_retinopathy_img_paths = \
    #     diabetic_retinopathy_dir + '/' + diabetic_retinopathy_df['image'].values + '.jpeg'
    # diabetic_retinopathy_labels = diabetic_retinopathy_df['level'].values

    # test_img_paths = np.hstack((test_img_paths, diabetic_retinopathy_img_paths))
    # test_labels = np.hstack((test_labels, diabetic_retinopathy_labels))

    test_set = Dataset(
        test_img_paths,
        test_labels,
        transform=test_transform)
    test_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=1,
        shuffle=False,
        num_workers=4)

    preds = []
    losses = []
    for fold in range(args.n_splits):
        print('Fold [%d/%d]' %(fold+1, args.n_splits))
        print('Fold [%d/%d]' %(fold+1, args.n_splits))
        # create model
        model_path = 'models/%s/model_%d.pth' % (args.name, fold+1)
        if not os.path.exists(model_path):
            print('%s is not exists.' %model_path)
        model = get_gcn_model(model_name=args.arch,
                          num_outputs=num_outputs,
                          freeze_bn=args.freeze_bn,
                          dropout_p=args.dropout_p)
        device = torch.device('cuda')

        criterion = nn.CrossEntropyLoss().cuda()

        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
        model.to(device)
        model.load_state_dict(torch.load(model_path))
        model.eval()        
        losses_fold = []
        with torch.no_grad():
            for i, (input, target) in tqdm(enumerate(test_loader), total=len(test_loader)):
                input = input.cuda()
                output = model(input)

                losses_fold.extend(criterion(output, target).data.cpu().numpy())

                if args.pred_type == 'classification':                
                    preds.extend(output.data.cpu().numpy())
                else:
                    preds.extend(output.data.cpu().numpy()[:, 0])
        losses_fold = np.array(losses_fold)
        losses.append(losses_fold)
    pdb.set_trace()

    def clustering(n_clusters=10, tol=1e-4):
        
        clusters_index, centers = lloyd(features, n_clusters, device=0, tol=tol)
        _, pred_counts = np.unique(preds, return_counts=True)
        _, cluster_counts = np.unique(clusters_index, return_counts=True)
        _, true_counts = np.unique(test_labels, return_counts=True)

        print('True count:' ,true_counts)
        print('Pred count:' ,pred_counts)
        print('Clus count:' ,cluster_counts)        
        true_class=[]
        print('Class count per cluster:')
        for i in range(n_clusters):
            label_count=np.zeros(5,dtype=np.int16)            
            for j in range(len(clusters_index)):            
                if clusters_index[j] == i:
                    label_count[test_labels[j]]+=1
            print(label_count)
            true_class.append(np.argmax(label_count))
        print('Label per cluster:', true_class)
        
        acc = metrics.accuracy_score(preds, test_labels, normalize=True)

        print('Accuracy: %f' %acc)

        torch.cuda.empty_cache()
    
    pdb.set_trace()
    clustering()

    # test_df['diagnosis'] = preds
    # test_df.to_csv('submissions/%s.csv' %args.name, index=False)

    


if __name__ == '__main__':
    main()