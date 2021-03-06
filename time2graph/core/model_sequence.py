# -*- coding: utf-8 -*-
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from time2graph.utils.deep_models import LSTMClassifier, GRUClassifier
from time2graph.utils.deep_utils import DeepDataloader, DeepDataset, train_RNNs
from .shapelet_utils import shapelet_distance
from .time_aware_shapelets import learn_time_aware_shapelets
from .static_shapelets import learn_static_shapelets
from ..utils.base_utils import Debugger
import torch.nn.functional as F


class Time2GraphSequence(object):
    """
        Sequence version of Time2Graph: Shapelet-Seq
    """
    def __init__(self, K=100, C=1000, seg_length=30, warp=2, tflag=True,
                 hidden_size=64, output_size=64, dropout=0.1, gpu_enable=True,
                 model='lstm', batch_size=100, **kwargs):
        super(Time2GraphSequence, self).__init__()
        self.K = K
        self.C = C
        self.seg_length = seg_length
        self.warp = warp
        self.tflag = tflag
        self.model = model
        self.batch_size = batch_size
        self.gpu_enable = gpu_enable
        self.shapelets = None
        self.rnns = None
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout = dropout
        self.lr = kwargs.pop('lr', 1e-2)
        self.p = kwargs.pop('p', 2)
        self.alpha = kwargs.pop('alpha', 10.0)
        self.beta = kwargs.pop('beta', 5.0)
        self.debug = kwargs.pop('debug', True)
        self.measurement = kwargs.pop('measurement', 'gdtw')
        self.niter = kwargs.pop('niter', 200)
        self.n_sequences = kwargs.pop('n_sequences', 1)
        self.kwargs = kwargs
        assert self.n_sequences == 1
        Debugger.info_print('initialize t2g model with {}'.format(self.__dict__))

    def learn_shapelets(self, x, y, num_segment, data_size, num_batch):
        assert x.shape[1] == num_segment * self.seg_length
        if self.tflag:
            self.shapelets = learn_time_aware_shapelets(
                time_series_set=x, label=y, K=self.K, C=self.C, p=self.p,
                num_segment=num_segment, seg_length=self.seg_length, data_size=data_size,
                lr=self.lr, alpha=self.alpha, beta=self.beta, num_batch=num_batch,
                measurement=self.measurement, gpu_enable=self.gpu_enable, **self.kwargs)
        else:
            self.shapelets = learn_static_shapelets(
                time_series_set=x, label=y, K=self.K, C=self.C, warp=self.warp, num_segment=num_segment,
                seg_length=self.seg_length, measurement=self.measurement,  **self.kwargs)

    def retrieve_sequence(self, x, init):
        assert self.shapelets is not None
        if len(x.shape) == 2:
            x = x.reshape(x.shape[0], x.shape[1], 1)
        data_length = x.shape[1]
        shapelet_dist = shapelet_distance(
            time_series_set=x, shapelets=self.shapelets, seg_length=self.seg_length, tflag=self.tflag,
            tanh=self.kwargs.get('tanh', False), debug=self.debug, init=init, warp=self.warp,
            measurement=self.measurement)
        ret = []
        for k in range(shapelet_dist.shape[0]):
            sdist, sequences = shapelet_dist[k], []
            for i in range(self.n_sequences):
                tmp = []
                for j in range(sdist.shape[0]):
                    min_s = np.argsort(sdist[j, :]).reshape(-1)[i]
                    tmp.append(self.shapelets[min_s][0])
                sequences.append(np.concatenate(tmp, axis=0))
            ret.append(np.array(sequences).reshape(self.n_sequences, data_length, -1))
        return np.array(ret)

    def fit(self, x, y, init=0):
        sequences = self.retrieve_sequence(x=x, init=init).reshape(x.shape[0], x.shape[1], -1)
        if self.model == 'lstm':
            self.rnns = LSTMClassifier(data_size=x.shape[-1], hidden_size=self.hidden_size,
                                       output_size=self.output_size, dropout=self.dropout, gpu_enable=self.gpu_enable)
        elif self.model == 'gru':
            self.rnns = GRUClassifier(data_size=x.shape[-1], hidden_size=self.hidden_size,
                                      output_size=self.output_size, dropout=self.dropout, gpu_enable=self.gpu_enable)
        else:
            raise NotImplementedError()
        self.rnns.double()
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.rnns.parameters(), lr=self.lr)
        if self.gpu_enable:
            self.rnns.cuda()
            criterion.cuda()
        train_dataset = DeepDataset(x=sequences, y=y)
        train_dataloader = DeepDataloader(train_dataset, batch_size=self.batch_size, shuffle=True,
                                          num_workers=2)
        for epoch in range(self.niter):
            train_RNNs(epoch=epoch, dataloader=train_dataloader, rnn=self.rnns, criterion=criterion,
                       optimizer=optimizer, debug=self.debug, gpu_enable=self.gpu_enable)

    def predict(self, x, init=0):
        assert self.shapelets is not None, 'shapelets has not been learnt yet...'
        assert self.rnns is not None, 'classifier has not been learnt yet...'
        x_seq = torch.from_numpy(self.retrieve_sequence(x=x, init=init).reshape(x.shape[0], x.shape[1], -1))
        if self.gpu_enable:
            return F.softmax(self.rnns(x_seq.cuda())).data.max(1, keepdim=True)[1].cpu().numpy()
        else:
            return F.softmax(self.rnns(x_seq)).data.max(1, keepdim=True)[1].numpy()

    def save_model(self, fpath):
        ret = {}
        for key, val in self.__dict__.items():
            ret[key] = val
        torch.save(ret, fpath)

    def load_model(self, fpath):
        cache = torch.load(fpath)
        for key, val in cache.items():
            self.__dict__[key] = val

    def save_shapelets(self, fpath):
        torch.save(self.shapelets, fpath)

    def load_shapelets(self, fpath):
        self.shapelets = torch.load(fpath)
