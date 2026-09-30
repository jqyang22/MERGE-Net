import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import argparse
import json
import os
import time
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import classification_report, cohen_kappa_score, accuracy_score
from thop import profile

from models.MERGE_Net import MERGE_Net, ETF_Classifier
from loadData import data_pipe


parser = argparse.ArgumentParser(description='MERGE-Net')
parser.add_argument('--device', type=str, default='cuda:0' if torch.cuda.is_available() else 'cpu')
parser.add_argument('--path-config', type=str, default='config/config_Houston2013.yaml')
parser.add_argument('--runs', type=int, default=10)
parser.add_argument('--seed', type=int, default=666)
args = parser.parse_args()
args.results_dir = datetime.now().strftime("%Y%m%d-%H%M")

config = yaml.load(open(args.path_config, "r"), Loader=yaml.FullLoader)
dataset_name = config["data_input"]["dataset_name"]
classes = config["data_input"]["classes"]
patch_size = config["data_input"]["patch_size"]
num_components = config["data_transforms"]["num_components"]
max_epoch = config["network_config"]["max_epoch"]
learning_rate = config["network_config"]["learning_rate"]
weight_decay = config["network_config"]["weight_decay"]
num_nodes = config["network_config"]["num_nodes"]
log_interval = config["result_output"]["log_interval"]
path_weight = config["result_output"]["path_weight"]
path_result = config["result_output"]["path_result"]
os.makedirs(path_result, exist_ok=True)
os.makedirs(path_weight, exist_ok=True)

w = 0.1
print("dataset: ", dataset_name, " patch_size: ", patch_size, '\n')


def train(net, classifier, train_loader, criterion, optimizer, scheduler):
    best_loss = 9999
    train_losses = []
    net.train()
    classifier.to(args.device)
    classifier.train()

    for epoch in range(1, max_epoch + 1):
        correct = 0
        for data, data_LiDAR, target in train_loader:
            data = data.to(args.device)
            data_LiDAR = data_LiDAR.to(args.device)
            target = target.to(args.device)

            optimizer.zero_grad()
            output_net = net(data, data_LiDAR, w)

            feat = classifier(output_net)
            output = output_net - torch.matmul(feat, classifier.ori_M.to(args.device))

            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()
        scheduler.step()
        train_losses.append(loss.cpu().detach().item())

        if epoch % log_interval == 0:
            print('Train Epoch: {}\tLoss: {:.6f} \tAccuracy: {:.6f}'.format(epoch, loss.item(), correct / len(train_loader.dataset)))
        if loss.item() < best_loss:
            best_loss = loss.item()
            torch.save(net.state_dict(), path_weight + 'model.pth')
            torch.save(optimizer.state_dict(), path_weight + 'optimizer.pth')
    return train_losses


def test(net, classifier, test_loader, criterion):
    net.eval()
    classifier.to(args.device)
    classifier.eval()

    test_preds = []
    test_loss = 0
    correct = 0
    net.load_state_dict(torch.load(path_weight + 'model.pth'))

    with torch.no_grad():
        for data, data_LiDAR, target in test_loader:
            data = data.to(args.device)
            data_LiDAR = data_LiDAR.to(args.device)
            target = target.to(args.device)

            output_net = net(data, data_LiDAR, w)
            feat = classifier(output_net)
            output = output_net - torch.matmul(feat, classifier.ori_M.to(args.device))

            test_loss += criterion(output, target).item()
            test_pred = output.data.max(1, keepdim=True)[1]
            correct += test_pred.eq(target.data.view_as(test_pred)).sum()
            test_preds.append(torch.argmax(output, dim=1).cpu().numpy().tolist())

    print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset), 100. * correct / len(test_loader.dataset)))
    return test_preds


seed = args.seed
Experiment_num = args.runs

Experiment_result = np.zeros([classes + 6 + 3, Experiment_num + 2])

for count in range(Experiment_num):
    tic0 = time.time()
    data_pipe.set_deterministic(seed=seed)

    train_loader, test_loader, train_label, test_label = data_pipe.get_data(path_config=args.path_config)

    net = MERGE_Net(input_channels=num_components, num_nodes=(np.max(test_label) + 1) * num_nodes,
                    num_classes=np.max(test_label) + 1, patch_size=patch_size, dataset_name=dataset_name).to(args.device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[max_epoch // 2, (5 * max_epoch) // 6], gamma=0.1)

    for data, data_LiDAR, target in train_loader:
        profile(net, (data.float().to(args.device), data_LiDAR.float().to(args.device), 0.9), verbose=False)
    for data, data_LiDAR, target in train_loader:
        macs, params = profile(net, inputs=(data.float().to(args.device), data_LiDAR.float().to(args.device), 0.9), verbose=False)
        g_macs = macs / 1e9
        g_flops = g_macs * 2
        m_params = params / 1e6
        break

    classifier = ETF_Classifier(feat_in=np.max(test_label) + 1, num_classes=np.max(test_label) + 1)

    tic1 = time.time()
    train(net, classifier, train_loader, criterion, optimizer, scheduler)
    toc1 = time.time()

    tic2 = time.time()
    test_preds = test(net, classifier, test_loader, criterion)
    toc2 = time.time()

    y_pred_test = [j for i in test_preds for j in i]

    num_tes = np.zeros([classes])
    num_tes_pred = np.zeros([classes])
    y_pre = np.array(y_pred_test)
    y_tes = np.array(test_label)
    for k in y_tes:
        num_tes[k - 1] += 1
    for j in range(y_tes.shape[0]):
        if y_tes[j] == y_pre[j]:
            num_tes_pred[y_tes[j] - 1] += 1
    Acc = num_tes_pred / (num_tes + 1e-5) * 100

    classification = classification_report(test_label, y_pred_test, digits=4)
    OA = 100. * accuracy_score(test_label, y_pred_test)
    Kappa = 100. * cohen_kappa_score(test_label, y_pred_test)
    toc0 = time.time()
    training_time = toc1 - tic1
    testing_time = toc2 - tic2
    runtime = toc0 - tic0

    print(classification)
    print("OA: ", OA, " Kappa: ", Kappa)

    Experiment_result[0, count] = OA
    Experiment_result[1, count] = np.mean(Acc)
    Experiment_result[2, count] = Kappa
    Experiment_result[3, count] = training_time
    Experiment_result[4, count] = testing_time
    Experiment_result[5, count] = runtime
    Experiment_result[6, count] = g_macs
    Experiment_result[7, count] = g_flops
    Experiment_result[8, count] = m_params
    Experiment_result[9:, count] = Acc

    with pd.ExcelWriter(path_result + 'seed' + str(seed) + '_w' + str(w) + '_' + str(int(OA * 100)) + '.xlsx') as writer:
        pd.DataFrame(Experiment_result).to_excel(writer, sheet_name='Acc&Time')

    end_result = {"classification": classification, "kappa": Kappa,
                  "training_time": training_time, "testing_time": testing_time}
    with open(path_result + args.results_dir + "-" + dataset_name + '.json', 'w') as fid:
        config.update(args.__dict__)
        config.update(end_result)
        json.dump(config, fid, indent=2)

    seed += 1


Experiment_result[:, -2] = np.mean(Experiment_result[:, 0:-2], axis=1)
Experiment_result[:, -1] = np.std(Experiment_result[:, 0:-2], axis=1)
with pd.ExcelWriter(path_result + dataset_name + '_w' + str(w) + '_AllMeanStd_' + repr(int(Experiment_result[0, -2] * 100))
                    + '_patch_' + repr(patch_size) + '.xlsx') as writer:
    pd.DataFrame(Experiment_result).to_excel(writer, sheet_name='All&Mean&Std')
print("OA mean/std: {:.2f} / {:.2f}".format(Experiment_result[0, -2], Experiment_result[0, -1]))
