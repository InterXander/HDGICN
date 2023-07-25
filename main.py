import argparse
import numpy as np
from sklearn import metrics
from utils.early_stoping import EarlyStopping
from utils.tools import get_dirs_path
import os
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear

from torch_geometric.data import Data
from sklearn.preprocessing import StandardScaler
import torch_geometric.transforms as T
from torch_geometric.nn import ChebConv, SAGEConv, DeepGraphInfomax, GCNConv
from torch_geometric.utils import dropout_adj


class Encoder(nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels, normalize=True)
        self.conv2 = ChebConv(hidden_channels, hidden_channels, K=2,normalization='sym')
        self.prelu = nn.PReLU(hidden_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = self.prelu(x)
        x = self.conv2(x, edge_index)
        x = self.prelu(x)
        return x

def corruption(x, edge_index):
    return x[torch.randperm(x.size(0))], edge_index

def summary(z, *args, **kwargs):
    return torch.sigmoid(z.mean(dim=0))

def get_DGI_model(num_features,output_channels,device):
    model = DeepGraphInfomax(
            hidden_channels=output_channels, encoder=Encoder(num_features, output_channels),
            summary=summary,
            corruption=corruption).to(device)
    return model

class Net(torch.nn.Module):
    def __init__(self, args, w=[0.99, 0.95, 0.30, 0.15]):
        super().__init__()
        self.args = args
        in_channels = self.args.in_channels
        hidden_channels = self.args.hidden_channels1
        self.fc1 = Linear(in_channels, hidden_channels)
        self.conv_1 = SAGEConv(hidden_channels, hidden_channels)
        self.conv_2 = SAGEConv(2 * hidden_channels, hidden_channels)
        self.conv_3 = SAGEConv(2 * hidden_channels, hidden_channels)
        self.conv_aux_1 = GCNConv(hidden_channels, hidden_channels)
        self.conv_aux_2 = GCNConv(2 * hidden_channels, hidden_channels)
        self.conv_aux_3 = GCNConv(2 * hidden_channels, hidden_channels)

        self.lin1 = Linear(hidden_channels, 1)
        self.lin2 = Linear(2*hidden_channels, 1)
        self.lin3 = Linear(2*hidden_channels, 1)
        self.lin4 = Linear(2*hidden_channels, 1)

        self.w1 = torch.nn.Parameter(torch.Tensor([w[0]]), requires_grad=True)
        self.w2 = torch.nn.Parameter(torch.Tensor([w[1]]), requires_grad=True)
        self.w3 = torch.nn.Parameter(torch.Tensor([w[2]]), requires_grad=True)
        self.w4 = torch.nn.Parameter(torch.Tensor([w[3]]), requires_grad=True)

    def forward(self, data):
        x_input = data.x
        edge_index = data.edge_index
        edge_index_aux = data.edge_index_aux

        edge_index, _ = dropout_adj(edge_index, p=0.5,
                                      force_undirected=True,
                                      num_nodes=x_input.shape[0],
                                      training=self.training)
        
        edge_index_aux, _ = dropout_adj(edge_index_aux, p=0.5,
                                      force_undirected=True,
                                      num_nodes=x_input.shape[0],
                                      training=self.training)

        x_input = F.dropout(x_input, p=0.5, training=self.training)

        out_1 = torch.relu(self.fc1(x_input))
        x0 = out_1
        
        out_c_1 = self.conv_1(out_1, edge_index)
        out_c_1 = out_c_1+x0 # residual connection x0
        out_aux_c_1 = self.conv_aux_1(out_1, edge_index_aux)
        out_aux_c_1 = out_aux_c_1+x0 # residual connection x0
        
        out_2 = torch.cat((out_c_1, out_aux_c_1), 1)
        x1 = out_2

        out_c_2 = self.conv_2(out_2, edge_index)
        out_c_2 = out_c_2+x0 # residual connection x0
        out_aux_c_2 = self.conv_aux_2(out_2, edge_index_aux)
        out_aux_c_2 = out_aux_c_2+x0 # residual connection x0

        out_3 = torch.cat((out_c_2, out_aux_c_2), 1)
        out_3 = out_3+x1 # residual connection x1

        out_c_3 = self.conv_3(out_3, edge_index)
        out_c_3 = out_c_3 + x0 # residual connection x0
        out_aux_c_3 = self.conv_aux_3(out_3, edge_index_aux)
        out_aux_c_3 = out_aux_c_3 + x0 # residual connection x0

        out_4 = torch.cat((out_c_3, out_aux_c_3), 1)
        out_4 = out_4+x1 # residual connection x1

        out_1 = F.dropout(out_1, p=0.5, training=self.training)
        res1 = self.lin1(out_1)
        out_2 = F.dropout(out_2, p=0.5, training=self.training)
        res2 = self.lin2(out_2)
        out_3 = F.dropout(out_3, p=0.5, training=self.training)
        res3 = self.lin3(out_3)
        out_4 = F.dropout(out_4, p=0.5, training=self.training)
        res4 = self.lin4(out_4)

        out = res1 * self.w1 + res2 * self.w2 + res3 * self.w3 + res4 * self.w4
        return out
    
args = argparse.Namespace()
data_name = 'GGNet'

params = {
    'graph_diffusion': 'ppr',
    'ppr_alpha': 0.9,
    'ppr_eps': 0.0001,
    'net_avg_deg': 50,
    'is_5_CV_test': True,
    'dataset_file': f'./data/{data_name}/dataset_{data_name}.pkl',
    'net_file': f'./data/{data_name}/{data_name}.txt',
    'epochs': 300,
    'mode':0,
    'lr': 0.001,
    'w_decay': 0.00005,
    'in_channels': 58,
    'hidden_channels1': 200,
    'device': 0
}

params["data_name" ] = data_name
if data_name == 'GGNet':
    params["net_avg_deg"]=111
elif data_name == 'PathNet':
    params["net_avg_deg"]=24
else:
    params["net_avg_deg"]=50

if params["is_5_CV_test"] == True:
    params["dataset_file"] = f'./data/{data_name}/dataset_{data_name}_ten_5CV.pkl'

args.__dict__.update(params)

device = torch.device('cuda:%d' % args.device if torch.cuda.is_available() else 'cpu')

with open(args.dataset_file, 'rb') as f:
    dataset = pickle.load(f)
mask = dataset['split_set']
std = StandardScaler()
features = std.fit_transform(dataset['feature'].detach().numpy())
features = torch.FloatTensor(features)

data_aux = Data(x=dataset['feature'], y=dataset['label'], edge_index=dataset['edge_index'])
gdc = T.GDC(self_loop_weight=None, normalization_in='sym',
                    normalization_out='col',
                    diffusion_kwargs=dict(method=args.graph_diffusion, alpha=args.ppr_alpha, eps=args.ppr_eps),
                    sparsification_kwargs=dict(method='threshold', avg_degree=args.net_avg_deg),
                    exact=True)
edge_index_aux = gdc(data_aux).edge_index

data = Data(x=features, y=dataset['label'], edge_index=dataset['edge_index'], edge_index_aux=edge_index_aux, mask=mask, node_names=dataset['node_name'])

data = data.to(device)

features_folder = f'./data/features/{args.data_name}/'
features_folder = get_dirs_path(features_folder)
features_path =  os.path.join(features_folder,f'features_{args.data_name}.pkl')

def get_dgi_data(x, edge_index, type=None):
    model = get_DGI_model(num_features=args.in_channels,output_channels=args.in_channels,device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    def train(epoch):
        model.train()
        optimizer.zero_grad()
        pos_z, neg_z, summary = model(x, edge_index)
        loss = model.loss(pos_z, neg_z, summary)
        
        loss.backward()
        optimizer.step()
        return loss.item()

    es = EarlyStopping(os.path.join(features_folder,f'best_features_model_{args.data_name}_{type}.pth'), 10, verbose=True)
    for epoch in range(1, 1000):
        loss = train(epoch=epoch)
        es(loss, model)
        print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')

        if es.early_stop:
            break
    model.eval()
    z, _, _ = model(x, edge_index)
    return z
if not os.path.exists( features_path):
    z1 = get_dgi_data(data.x, data.edge_index, type="origin")
    z2 = get_dgi_data(data.x, data.edge_index_aux, type="auxiliary")
    torch.save(torch.div(z1+z2,2), features_path)

# 获取增强表示
datas = torch.load(features_path)
data.x = data.x+datas
data = data.to(device)

def compute_auc_with_threshold(y_true, pred, thresholds):
    accuracys = []
    for t in thresholds:
        binary_predictions = (pred >= t).astype(int)  # 将概率分数转换为二进制类别预测
        accuracy = metrics.accuracy_score(y_true, binary_predictions)
        accuracys.append(accuracy)
    return accuracys, thresholds

@torch.no_grad()
def test(model,data,mask):
    model.eval()
    x = model(data)
    pred = torch.sigmoid(x[mask])
    precision, recall, aupr_thresholds = metrics.precision_recall_curve(data.y[mask].cpu().numpy(), pred.cpu().detach().numpy())
    aupr = metrics.auc(recall, precision)
    accuracys, acc_thresholds = compute_auc_with_threshold(data.y[mask].cpu().numpy(), pred.cpu().detach().numpy(), aupr_thresholds)
    fpr, tpr, auc_thresholds = metrics.roc_curve(data.y[mask].cpu().numpy(), pred.cpu().detach().numpy())
    roc_auc = metrics.auc(fpr, tpr)
    return roc_auc, fpr, tpr, auc_thresholds, aupr, precision, recall, aupr_thresholds, accuracys, acc_thresholds

import time
start_time = time.time()
if args.mode == 0:
    # Ten times of 5_CV
    AUC = np.zeros(shape=(10, 5))
    AUPR = np.zeros(shape=(10, 5))

    for i in range(10):
        for cv_run in range(5):
            tr_mask, te_mask = data.mask[i][cv_run]
            model = Net(args).to(device)

            optimizer = torch.optim.Adam([
                dict(params=model.fc1.parameters(), weight_decay=args.w_decay),
                dict(params=model.lin1.parameters(), weight_decay=args.w_decay),
                dict(params=model.lin2.parameters(), weight_decay=args.w_decay),
                dict(params=model.lin3.parameters(), weight_decay=args.w_decay),
                dict(params=model.lin4.parameters(), weight_decay=args.w_decay),
                dict(params=model.w1, lr=args.lr * 0.1),
                dict(params=model.w2, lr=args.lr * 0.1),
                dict(params=model.w3, lr=args.lr * 0.1),
                dict(params=model.w4, lr=args.lr * 0.1)
            ], lr=args.lr)

            for epoch in range(1, args.epochs + 1):
                model.train()
                optimizer.zero_grad()
                pred = model(data) 
                loss = F.binary_cross_entropy_with_logits(pred[tr_mask], data.y[tr_mask].view(-1, 1))
                loss.backward()             
                optimizer.step()
                if epoch % 100 == 0:
                    print(f'Training epoch: {epoch:03d}')
           
            # roc_auc, fpr, tpr, auc_thresholds, aupr, precision, recall, aupr_thresholds, accuracys, acc_thresholds
            AUC[i][cv_run], fpr, tpr, auc_threshold, AUPR[i][cv_run], precision, recall, aupr_threshold, accuracy, acc_threshold = test(model, data, te_mask)

            print('Round--%d CV--%d  AUROC: %.5f, AUPRC: %.5f' % (i, cv_run+1, AUC[i][cv_run], AUPR[i][cv_run]))
        print('Round--%d Mean AUROC: %.5f, Mean AUPRC: %.5f' % (i, np.mean(AUC[i, :]), np.mean(AUPR[i, :])))
end_time = time.time()

execution_time = end_time - start_time

if args.mode == 0:    
    print('10 rounds for 5CV--:')
    minutes, seconds = divmod(execution_time, 60)
    data_result = {'MeanAUROC': [AUC.mean()], 'MeanAUPRC': [AUPR.mean()],'TotalTime':[f'{minutes}m{seconds}s']}
    print(data_result)