import torch
from torch.nn import functional
from torch import nn
from utils import trace

class VAE(nn.Module):
    def __init__(self, args):
        super(VAE, self).__init__()
        self.l = len(args.layer)
        self.device = args.device
        self.inet = nn.ModuleList()
        darray = [args.d] + args.layer
        for i in range(self.l - 1):
            self.inet.append(nn.Linear(darray[i], darray[i + 1]))
        self.mu = nn.Linear(darray[self.l - 1], darray[self.l])
        self.sigma = nn.Linear(darray[self.l - 1], darray[self.l])
        self.gnet = nn.ModuleList()
        for i in range(self.l):
            self.gnet.append(nn.Linear(darray[self.l - i], darray[self.l - i - 1]))

    def encode(self, x):
        h = x
        for i in range(self.l - 1):
            h = functional.relu(self.inet[i](h))
        return self.mu(h), self.sigma(h)

    def decode(self, z):
        h = z
        for i in range(self.l - 1):
            h = functional.relu(self.gnet[i](h))
        return self.gnet[self.l - 1](h)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    def infer_reg(self):
        reg = 0
        for infer in self.inet:
            for param in infer.parameters():
                reg += trace(param)
        return reg

    def gen_reg(self):
        reg = 0
        for infer in self.gnet:
            for param in infer.parameters():
                reg += trace(param)
        return reg
