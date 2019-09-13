import argparse

import torch
from scipy import io
from torch import optim
from utils import metric_map, metric_recall, sort2query, csr2test, Evaluator
from torch.utils.data import DataLoader
from tqdm import tqdm
from vae import VAE
import numpy as np
from sklearn.preprocessing import normalize


def Guassian_loss(recon_x, x):
    recon_x = torch.sigmoid(recon_x)
    weights = x * args.alpha + (1 - x)
    loss = x - recon_x
    loss = torch.sum(weights * (loss ** 2))
    return loss


def BCE_loss(recon_x, x):
    recon_x = torch.sigmoid(recon_x)
    eps = 1e-8
    loss = -torch.sum(args.alpha * torch.log(recon_x + eps) * x + torch.log(1 - recon_x + eps) * (1 - x))
    return loss


def regularization(mu, logvar):
    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())


def train(epoch):
    model.train()
    loss_value = 0
    for batch_idx, data in enumerate(tqdm(train_loader, unit='epoch')):
        data = data.to(args.device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = loss_function(recon_batch, data) + regularization(mu, logvar) * args.beta
        loss.backward()
        loss_value += loss.item()
        optimizer.step()
        # if args.log != 0 and batch_idx % args.log == 0:
        #     print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        #         epoch, batch_idx * len(data), len(train_loader.dataset),
        #                100. * batch_idx / len(train_loader),
        #                loss.item() / len(data)))

    print('====> Epoch: {} Average loss: {:.4f}'.format(epoch, loss_value / len(train_loader.dataset)))


def evaluate(split='valid'):
    y_true = eval(split + '_data')
    # Ttensor = eval(split + '_tensor')
    model.eval()
    y_score, _, _ = model(train_tensor)
    y_score.detach_()
    y_score = y_score.squeeze(0)
    y_score[train_data.row, train_data.col] = 0
    _, rec_items = torch.topk(y_score, args.N, dim=1)
    # y_pred = torch.gather(Ttensor, 1, rec_items).cpu().numpy()
    run = sort2query(rec_items[:, 0:args.N])
    test = csr2test(y_true.tocsr())
    evaluator = Evaluator({'recall', 'map_cut'})
    evaluator.evaluate(run, test)
    result = evaluator.show(
        ['recall_5', 'recall_10', 'recall_15', 'recall_20', 'map_cut_5', 'map_cut_10', 'map_cut_15', 'map_cut_20'])
    print(result)


# def evaluate(split='valid'):
#     # Ttensor = torch.from_numpy(T.A.astype('float32'))
#     # test_loader = DataLoader(Ttensor, args.batch, shuffle=True)
#     y_true = eval(split + '_data')
#     Ttensor = eval(split + '_tensor')
#     model.eval()
#     y_score, _, _ = model(train_tensor)
#     y_score.detach_()
#     y_score = y_score.squeeze(0)
#     y_score[train_data.row, train_data.col] = 0
#     y_score, rec_items = torch.topk(y_score, args.N, dim=1)
#     y_pred = torch.gather(Ttensor, 1, rec_items).cpu().numpy()
#     # y_score = y_score.cpu().numpy()
#     map_res = metric_map(y_pred, y_true)
#     recall_res = metric_recall(y_pred, y_true)
#     print('{}: Recall@{}={}, MAP@{}={}'.format(split, args.N, np.mean(recall_res), args.N, np.mean(map_res)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Variational Auto Encoder')
    parser.add_argument('--batch', type=int, default=100, help='input batch size for training (default: 100)')
    parser.add_argument('-m', '--maxiter', type=int, default=5, help='number of epochs to train (default: 10)')
    parser.add_argument('--gpu', action='store_true', default=False, help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
    parser.add_argument('--log', type=int, default=1, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--dir', help='dataset directory', default='/Users/chenyifan/jianguo/dataset')
    parser.add_argument('--data', help='specify dataset', default='test')
    parser.add_argument('--layer', nargs='+', help='number of neurals in each layer', type=int, default=[20])
    parser.add_argument('-N', help='number of recommended items', type=int, default=10)
    parser.add_argument('--lr', help='learning rate', type=float, default=1e-3)
    parser.add_argument('-a', '--alpha', help='parameter alpha', type=float, default=1)
    parser.add_argument('-b', '--beta', help='parameter beta', type=float, default=1)
    parser.add_argument('--rating', help='feed input as rating', action='store_true')
    parser.add_argument('--save', help='save model', action='store_true')
    parser.add_argument('--load', help='load model', type=int, default=0)

    args = parser.parse_args()
    torch.manual_seed(args.seed)
    args.device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")

    print('dataset directory: ' + args.dir)
    directory = args.dir + '/' + args.data

    path = '{}/split/train.txt'.format(directory)
    print('train data path: ' + path)
    train_data = io.mmread(path)
    train_tensor = torch.from_numpy(train_data.A.astype('float32')).to(args.device)

    path = '{}/split/valid.txt'.format(directory)
    valid_data = io.mmread(path)
    # print(np.sum(valid_data, axis=1))
    valid_tensor = torch.from_numpy(valid_data.A.astype('float32')).to(args.device)

    path = '{}/split/test.txt'.format(directory)
    test_data = io.mmread(path)
    test_tensor = torch.from_numpy(test_data.A.astype('float32')).to(args.device)

    if args.rating:
        args.d = train_data.shape[1]
        train_loader = DataLoader(train_tensor, args.batch, shuffle=True)
        loss_function = BCE_loss
    else:
        path = directory + '/feature.txt'
        print('feature data path: ' + path)
        X = io.mmread(path).A.transpose()
        X[X > 0] = 1
        args.d = X.shape[1]
        # X = normalize(X, axis=1)
        X = torch.from_numpy(X.astype('float32')).to(args.device)
        train_loader = DataLoader(X, args.batch, shuffle=True)
        loss_function = Guassian_loss

    model = VAE(args).to(args.device)
    if args.load > 0:
        name = 'cvae' if args.load == 2 else 'fvae'
        path = directory + '/model/' + name
        for l in args.layer:
            path += '_' + str(l)
        print('load model from path: ' + path)
        model.load_state_dict(torch.load(path))

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    evaluate()
    for epoch in range(1, args.maxiter + 1):
        train(epoch)
        evaluate()
        evaluate('test')
    # test()

    # _, idx = torch.sort(score, 1, True)

    # y_score = idx[:, :args.N]
    # test = T.tocsr()
    # evaluator = Evaluator({'recall'})

    # evaluator.evaluate(run, test)
    # result = evaluator.show(
    #     ['recall_5', 'recall_10', 'recall_15', 'recall_20'])
    # print(result)
    # line = 'cVAE\t{}\t{}\t{}\t{}\t0'.format(args.data, args.alpha, args.beta, len(args.layer))
    # for _, value in result.items():
    #     line += '\t{:.5f}'.format(value)
    # line += '\r\n'
    # file = open('result', 'a')
    # file.write(line)
    # file.close()
    #
    if args.save:
        name = 'cvae' if args.rating else 'fvae'
        path = directory + '/model/' + name
        for l in args.layer:
            path += '_' + str(l)
        model.cpu()
        torch.save(model.state_dict(), path)
