import os
import time
import glob
import argparse

import torch
import torch.nn.functional as F
from models import NodeClassificationModel
from torch_geometric.datasets import Coauthor, Planetoid
from utils import index_to_mask, random_splits

parser = argparse.ArgumentParser()

parser.add_argument('--seed', type=int, default=777, help='random seed')
parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
parser.add_argument('--weight_decay', type=float, default=0.01, help='weight decay')
parser.add_argument('--nhid', type=int, default=64, help='hidden size')
parser.add_argument('--depth', type=int, default=4, help='number of encoder layers')
parser.add_argument('--sample_neighbor', type=bool, default=True, help='whether sample neighbors within h-hops')
parser.add_argument('--sparse_attention', type=bool, default=True, help='whether use sparse attention')
parser.add_argument('--structure_learning', type=bool, default=True, help='whether perform structure learning')
parser.add_argument('--hop_connection', type=bool, default=False, help='whether directly connect node within h-hops')
parser.add_argument('--hop', type=int, default=2, help='h-hops')
parser.add_argument('--lamb', type=float, default=0.0, help='trade-off parameter')
parser.add_argument('--dataset', type=str, default='CS', help='Cora/Citeseer/Pubmed/Physics')
parser.add_argument('--device', type=str, default='cuda:0', help='specify cuda devices')
parser.add_argument('--epochs', type=int, default=200, help='maximum number of epochs')
parser.add_argument('--pool1', type=float, default=0.05, help='pool1 parameter')
parser.add_argument('--pool2', type=float, default=0.5, help='pool2 parameter')
parser.add_argument('--pool3', type=float, default=0.5, help='pool3 parameter')
parser.add_argument('--pool4', type=float, default=0.5, help='pool4 parameter')
parser.add_argument('--pool5', type=float, default=0.8, help='pool5 parameter')

args = parser.parse_args()
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)

if args.dataset == 'Physics' or args.dataset == 'CS':
    dataset = Coauthor(os.path.join('data', args.dataset), args.dataset)
    data = dataset.data
    data = random_splits(data, dataset.num_classes)
else:
    dataset = Planetoid(os.path.join('data', args.dataset), args.dataset)
    data = dataset.data

args.num_nodes = data.x.size(0)
args.num_features = data.x.size(1)
args.num_classes = dataset.num_classes

print(args)

model = NodeClassificationModel(args).to(args.device)
data = data.to(args.device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)


def train():
    best_test_acc = 0
    val_acc_values = []
    val_loss_values = []
    best_epoch = 0
    min_loss = 1e10

    t = time.time()
    for epoch in range(args.epochs):
        loss_train = 0.0
        model.train()
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        out = F.log_softmax(out, dim=1)
        loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()
        loss_train += loss.item()
        pred = out[data.train_mask].max(dim=1)[1]
        correct = pred.eq(data.y[data.train_mask]).sum().item()
        acc_train = correct / data.train_mask.sum().item()
        acc_val, loss_val = compute_test(data.val_mask)
        acc_test, loss_test = compute_test(data.test_mask)

        if acc_test > best_test_acc:
            best_test_acc = acc_test

        print('Epoch: {:04d}'.format(epoch + 1), 'loss_train: {:.4f}'.format(loss_train),
              'acc_train: {:.4f}'.format(acc_train), 'loss_val: {:.4f}'.format(loss_val),
              'acc_val: {:.4f}'.format(acc_val), 'acc_test: {:.4f}'.format(best_test_acc),
              'time: {:.4f}s'.format(time.time() - t))

        val_acc_values.append(acc_val)
        val_loss_values.append(loss_val)
        torch.save(model.state_dict(), '{}.pth'.format(epoch))

        if val_loss_values[-1] < min_loss:
            min_loss = val_loss_values[-1]
            best_epoch = epoch

        files = glob.glob('*.pth')
        for f in files:
            epoch_nb = int(f.split('.')[0])
            if epoch_nb < best_epoch:
                os.remove(f)

    files = glob.glob('*.pth')
    for f in files:
        epoch_nb = int(f.split('.')[0])
        if epoch_nb > best_epoch:
            os.remove(f)
    print('Optimization Finished! Total time elapsed: {:.4f}'.format(time.time() - t))

    return best_epoch


def compute_test(mask):
    model.eval()
    with torch.no_grad():
        correct = 0.0
        loss_test = 0.0
        out = model(data.x, data.edge_index)
        out = F.log_softmax(out, dim=1)
        pred = out[mask].max(dim=1)[1]
        correct += pred.eq(data.y[mask]).sum().item()
        loss_test += F.nll_loss(out[mask], data.y[mask]).item()
        return correct / mask.sum().item(), loss_test


def save_embedding(inputfile):
    model.eval()
    f = open(inputfile, 'w')
    with torch.no_grad():
        embeddings = model.gen_embedding(data.x, data.edge_index)
        embeddings = embeddings.cpu().detach().numpy()
        gt = data.y.cpu().detach().numpy()
        num_nodes, num_dims = embeddings.shape
        for i in range(num_nodes):
            write_string = str(gt[i])
            for j in range(num_dims):
                write_string += ' ' + str(embeddings[i, j])
            write_string += '\n'
            f.writelines(write_string)
    f.close()


if __name__ == '__main__':
    # Model training
    best_model = train()
    # Restore best model for test set
    model.load_state_dict(torch.load('{}.pth'.format(best_model)))
    test_acc, test_loss = compute_test(data.test_mask)
    print('Test set results, loss = {:.4f}, accuracy = {:.4f}'.format(test_loss, test_acc))
