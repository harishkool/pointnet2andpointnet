import argparse
import numpy as np
import sys
import open3d
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import functional, optim
import os
import importlib
# from torch.utils.tensorboard import SummaryWriter
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'models'))
sys.path.append(os.path.join(BASE_DIR, 'utils'))
import util
from pointnet_cls import *
import pdb
from transforms_alt import *
from tensorboard_logger import configure, log_value

TRAIN_FILES = util.getDataFiles( \
    os.path.join(BASE_DIR, 'data/modelnet40/train_files.txt'))
TEST_FILES = util.getDataFiles(\
    os.path.join(BASE_DIR, 'data/modelnet40/test_files.txt'))

parser = argparse.ArgumentParser()
parser.add_argument('--model', default='pointnet_cls', help='Model you want to train pointnet_cls for classification')
parser.add_argument('--batch_size', default=32, type=int, help='Batch size for the network')
parser.add_argument('--print_every', type=int, default=4, help='Print loss values')
parser.add_argument('--num_epochs', type=int, default=10, help='Number of epochs')
parser.add_argument('--evaluation_epoch', type=int, default=10, help='Run evaluation for every this argument value of epochs')
parser.add_argument('--checkpoint_dir',default='./checkpoints')
parser.add_argument('--continue_latest',type=int, default=0)
parser.add_argument('--trial_name',default='run1')
args = parser.parse_args()
CUDA_FLAG = torch.cuda.is_available()
NUM_CLASSES = 40
num_epochs = args.num_epochs
evaluation_epoch = args.evaluation_epoch
trail_name ="runs"+args.trial_name
configure(trail_name,flush_secs=5)

if not os.path.isdir(args.checkpoint_dir):
    os.mkdir(args.checkpoint_dir)

def get_model():
    if args.model=='pointnet_cls':
        model = PointNetcls_conv1d(NUM_CLASSES)
        input_transform = input_transform_net_alt()
        feature_transform = feature_transform_net_alt()
    if CUDA_FLAG:
        print('Model moved to cuda')
        model = model.cuda()
        # input_transform.cuda()
        # feature_transform.cuda()
    return model

def save_model(experiment_directory, filename, model, epoch):
    path = os.path.join(experiment_directory,'model')
    if not os.path.isdir(path):
        os.mkdir(path)
    torch.save(
        {"epoch": epoch, "model_state_dict": model.state_dict()},
        os.path.join(path, filename),
    )


def save_optimizer(experiment_directory, filename, optimizer, epoch):
    path = os.path.join(experiment_directory,'optimizer')
    if not os.path.isdir(path):
        os.mkdir(path)
    torch.save(
        {"epoch": epoch, "optimizer_state_dict": optimizer.state_dict()},
        os.path.join(path, filename),
    )


def load_optimizer(experiment_directory, filename, optimizer):

    full_filename = os.path.join(experiment_directory, 'optimizer',filename)

    if not os.path.isfile(full_filename):
        raise Exception(
            'optimizer state dict "{}" does not exist'.format(full_filename)
        )

    data = torch.load(full_filename)

    optimizer.load_state_dict(data["optimizer_state_dict"])
    print('loaded the optimizer')
    return optimizer


def load_model(checkpoints_dir, filename, model):
    full_file = os.path.join(checkpoints_dir,'model', filename)
    if not os.path.isfile(full_file):
        raise Exception('model state dict {} doesnot exists'.format(filename))
    data = torch.load(full_file)

    model.load_state_dict(data["model_state_dict"])
    print('loaded the model')
    return model

def train():

    train_file_idxs = np.arange(0, len(TRAIN_FILES))
    test_file_idxs = np.arange(0, len(TEST_FILES))
    np.random.shuffle(train_file_idxs)
    model = get_model()
    print(model)
    dtype = torch.FloatTensor
    criterion = nn.CrossEntropyLoss()
    # optimizer = optim.SGD(model.parameters(), lr=0.001)
    optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))
    losses = []
    # writer = SummaryWriter()
    global_step =0
    # pdb.set_trace()
    # for i in range(len(TEST_FILES)):
    #     test_data, gt = util.loadDataFile(TEST_FILES[test_file_idxs[i]])
    #     print(test_data.shape)
    #     print(gt[0])
    if args.continue_latest==1:
        model = load_model(args.checkpoint_dir,'latest.pth',model)
        optimizer = load_optimizer(args.checkpoint_dir,'latest.pth',optimizer)
    for epoch in range(num_epochs):
        running_loss = 0
        num_total_btches=0
        total_correct =0
        total_training_samples=0
        for i in range(len(TRAIN_FILES)):
            train_data, current_labels = util.loadDataFile(TRAIN_FILES[train_file_idxs[i]])
            train_data = util.to_var(torch.from_numpy(train_data))
            current_labels = util.to_var(torch.from_numpy(current_labels))
            num_batches = train_data.shape[0] // args.batch_size
            print('Training file: {:5d} |num of batches: {:5d}'.format(i ,num_batches))
            for btch in range(num_batches):
                optimizer.zero_grad()
                start_idx = btch*args.batch_size
                end_idx = (btch+1)*args.batch_size
                current_train = train_data[start_idx:end_idx, :, :]
                btch_label = current_labels[start_idx:end_idx,:].type(torch.long)
                # pdb.set_trace()
                logits = model(current_train)
                # pdb.set_trace()
                loss = criterion(logits, btch_label.view(-1))
                loss.backward()
                optimizer.step()
                preds = F.log_softmax(logits, 1)
                pred_choice = preds.data.max(1)[1]
                correct = pred_choice.eq(btch_label.view(-1).data).cpu().sum()
                total_correct+=correct.item()
                running_loss+= loss.item()*args.batch_size  
                losses.append(loss.item())
                total_training_samples+=btch_label.shape[0]
                # writer.add_scalar('loss',loss.item(), global_step)
                # writer.add_graph(model,current_train)
                # pdb.set_trace()
                if btch % args.print_every==0:
                    print('Epoch [{:5d}/{:5d}] | loss: {:6.4f} | accuracy:{:6.4f}'.format(epoch+1, num_epochs, loss.item(),
                                                                    correct.item()/float(args.batch_size)))
                global_step+=1
                num_total_btches+=1
        print(num_total_btches*args.batch_size, total_training_samples)      
        print("Epoch {} : Total training loss {:6.4f} and accuracy {:6.4f}".format(epoch, running_loss/total_training_samples, total_correct/total_training_samples))
        log_value('training_loss',running_loss/total_training_samples,epoch)
        log_value('accuracy',total_correct/total_training_samples,epoch)
        if (epoch % evaluation_epoch==0 and epoch!=0):
            model.eval()
            pred_score = 0
            test_loss+ = 0
            total_test_samples=0
            num_test_batches=0
            with torch.no_grad():
                for i in range(len(TEST_FILES)):
                    test_data, gt = util.loadDataFile(TEST_FILES[test_file_idxs[i]])
                    test_data = util.to_var(torch.from_numpy(test_data))
                    gt = util.to_var(torch.from_numpy(gt)).type(torch.long)
                    num_batches = test_data.shape[0] // args.batch_size
                    for btch in range(num_batches):
                        start_indx = btch*args.batch_size
                        end_indx = (btch+1)*args.batch_size
                        current_test = test_data[start_indx:end_indx, :, :]
                        logits = model(current_test)
                        gt_btch = gt[start_indx:end_indx,:]
                        loss = criterion(logits, gt_btch.view(-1))
                        test_loss+=loss.item()*args.batch_size
                        preds = F.log_softmax(logits, 1)
                        predictions = preds.data.max(1)[1]
                        actuals = predictions.eq(gt_btch.view(-1).data).cpu().sum()
                        pred_score+=actuals.item()
                        num_test_batches+=1
                        total_test_samples+=gt_btch.shape[0]
            # pdb.set_trace()
            # print(num_test_batches*args.batch_size, total_test_samples)
            print('Evaluation loss {:6.4f} | Accuracy {:6.4f}'.format(test_loss/total_test_samples,pred_score/total_test_samples))
            log_value('evaluation_accuracy',pred_score/total_test_samples, epoch)      
            model.train()   
        
    # writer.close()
    model.eval()
    pred_score = 0
    test_loss = 0
    total_test_samples=0
    num_test_batches=0
    with torch.no_grad():
        for i in range(len(TEST_FILES)):
            test_data, gt = util.loadDataFile(TEST_FILES[test_file_idxs[i]])
            test_data = util.to_var(torch.from_numpy(test_data))
            gt = util.to_var(torch.from_numpy(gt)).type(torch.long)
            num_batches = test_data.shape[0] // args.batch_size
            for btch in range(num_batches):
                start_indx = btch*args.batch_size
                end_indx = (btch+1)*args.batch_size
                current_test = test_data[start_indx:end_indx, :, :]
                logits = model(current_test)
                gt_btch = gt[start_indx:end_indx,:]
                loss = criterion(logits, gt_btch.view(-1))
                test_loss+=loss.item()*args.batch_size
                preds = F.log_softmax(logits, 1)
                predictions = preds.data.max(1)[1]
                actuals = predictions.eq(gt_btch.view(-1).data).cpu().sum()
                pred_score+=actuals.item()
                num_test_batches+=1
                total_test_samples+=gt_btch.shape[0]
    print('Final test loss {:6.4f} | accuracy {:6.4f}'.format(test_loss/total_test_samples,pred_score/total_test_samples))
    save_model(args.checkpoint_dir,'latest.pth',model, num_epochs) 
    save_optimizer(args.checkpoint_dir,'latest.pth',optimizer,num_epochs)


def test():
    pass

if __name__=='__main__':
    train()
    test()