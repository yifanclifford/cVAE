import argparse

import torch
from scipy import io
from torch import optim
from torch.nn import functional
from torch.utils.data import DataLoader

from utils import Evaluator
from utils import sort2query, csr2test
from vae import VAE


def Guassian_loss(recon_x, x):
    recon_x = functional.sigmoid(recon_x)
    weights = x * args.alpha + (1 - x)
    loss = x - recon_x
    loss = torch.sum(weights * loss * loss)
    return loss


def BCE_loss(recon_x, x):
    recon_x = functional.sigmoid(recon_x)
    eps = 1e-8
    loss = -torch.sum(args.alpha * torch.log(recon_x + eps) * x + torch.log(1 - recon_x + eps) * (1 - x))
    return loss


def regularization(mu, logvar):
    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())


def train(epoch):
    model.train()
    loss_value = 0
    for batch_idx, data in enumerate(train_loader):

        data = data.to(args.device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)

        loss = loss_function(recon_batch, data) + regularization(mu, logvar) * args.beta
        loss.backward()
        loss_value += loss.item()
        optimizer.step()
        if args.log != 0 and batch_idx % args.log == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader),
                       loss.item() / len(data)))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
        epoch, loss_value / len(train_loader.dataset)))


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Variational Auto Encoder')
    parser.add_argument('--batch', type=int, default=100, help='input batch size for training (default: 100)')
    parser.add_argument('-m', '--maxiter', type=int, default=5, help='number of epochs to train (default: 10)')
    parser.add_argument('--gpu', action='store_true', default=False, help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
    parser.add_argument('--log', type=int, default=1, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--cold', help='evaluate under cold-start setting')
    parser.add_argument('--fold', help='specify the fold', type=int, default=1)
    parser.add_argument('--dir', help='dataset directory', default='/Users/chenyifan/jianguo/dataset')
    parser.add_argument('-d', '--data', help='specify dataset', default='test')
    parser.add_argument('--layer', nargs='+', help='number of neurals in each layer', type=int, default=[20])
    parser.add_argument('-N', help='number of recommended items', type=int, default=20)
    parser.add_argument('--lr', help='learning rate', type=float, default=1e-3)
    parser.add_argument('-a', '--alpha', help='parameter alpha', type=float, default=0)
    parser.add_argument('-b', '--beta', help='parameter beta', type=float, default=1)
    parser.add_argument('--rating', help='feed input as rating', action='store_true')
    parser.add_argument('--save', help='save model', action='store_true')
    parser.add_argument('--load', help='load model', type=int, default=0)

    args = parser.parse_args()
    torch.manual_seed(args.seed)
    args.device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")

    print('dataset directory: ' + args.dir)
    directory = args.dir + '/' + args.data

    path = '{}/split/train{}.txt'.format(directory, args.fold)
    print('train data path: ' + path)
    R = io.mmread(path)
    Rtensor = R.A
    Rtensor = torch.from_numpy(Rtensor.astype('float32')).to(args.device)
    if args.rating:
        args.d = R.shape[1]
        train_loader = DataLoader(Rtensor, args.batch, shuffle=True)
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

    testfile = 'valid'
    path = '{}/split/{}{}.txt'.format(directory, testfile, args.fold)
    print('test file path: {}'.format(path))
    T = io.mmread(path)
    Ttensor = torch.from_numpy(T.A.astype('float32')).to(args.device)
    test_loader = DataLoader(Ttensor, args.batch, shuffle=True)

    model = VAE(args).to(args.device)
    if args.load > 0:
        name = 'cvae' if args.load == 2 else 'fvae'
        path = directory + '/model/' + name
        for l in args.layer:
            path += '_' + str(l)
        print('load model from path: ' + path)
        model.load_state_dict(torch.load(path))

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(1, args.maxiter + 1):
        train(epoch)
        # test()

    model.eval()
    score, _, _ = model(Rtensor)
    score = score.squeeze(0)
    score[R.row, R.col] = 0
    _, idx = torch.sort(score, 1, True)

    run = sort2query(idx[:, 0:args.N])
    test = csr2test(T.tocsr())
    evaluator = Evaluator({'recall'})
    evaluator.evaluate(run, test)
    result = evaluator.show(
        ['recall_5', 'recall_10', 'recall_15', 'recall_20'])
    print(result)
    line = 'cVAE\t{}\t{}\t{}\t{}\t0'.format(args.data, args.alpha, args.beta, len(args.layer))
    for _, value in result.items():
        line += '\t{:.5f}'.format(value)
    line += '\r\n'
    file = open('result', 'a')
    file.write(line)
    file.close()

    if args.save:
        name = 'cvae' if args.rating else 'fvae'
        path = directory + '/model/' + name
        for l in args.layer:
            path += '_' + str(l)
        model.cpu()
        torch.save(model.state_dict(), path)
