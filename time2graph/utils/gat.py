import torch
import torch.nn as nn
import torch.nn.functional as F
from .gat_utils import GraphAttentionLayer


class GAT(nn.Module):
    def __init__(self, nfeat, nhid, nnodes, nclass, dropout, alpha, nheads, aggregate):
        """Dense version of GAT."""
        super(GAT, self).__init__()
        self.dropout = dropout
        # sum or aggregate flag
        self.aggregate = aggregate
        self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        # self.out_att = GraphAttentionLayer(nhid * nheads, nclass, dropout=dropout,
        # alpha=alpha, concat=False, reshape=True)
        # self.add_module('attention_out', self.out_att)

        self.hidden_size = nnodes * nhid * nheads if self.aggregate else nnodes * nhid
        self.output = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size * 2),
            nn.ReLU(),
            nn.Linear(self.hidden_size * 2, nclass)
        )

    def forward(self, x, adj, feat_flag=False):
        x = F.dropout(x, self.dropout, training=self.training)
        x_head = [att(x, adj) for att in self.attentions]
        if self.aggregate:
            x = torch.cat(x_head, dim=2).view(x.size()[0], -1)
        else:
            x = torch.sum(torch.stack(x_head, dim=2), dim=2).view(x.size()[0], -1)
        x = F.dropout(x, self.dropout, training=self.training)
        if feat_flag:
            return F.elu(x)
        else:
            x = self.output(F.elu(x))
            return F.log_softmax(x, dim=1)


def accuracy_torch(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def label_np(output, cuda):
    if cuda:
        return output.max(1)[1].cpu().numpy()
    else:
        return output.max(1)[1].numpy()


def output_np(output, cuda):
    if cuda:
        return output.detach().cpu().numpy()
    else:
        return output.detach().numpy()
