# -*- coding: utf-8 -*-
import torch
import torch.optim as optim
import torch.nn.functional as F
from config import *
from scipy.special import softmax
from .shapelet_utils import shapelet_distance, adjacent_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from .time_aware_shapelets import learn_time_aware_shapelets
from .static_shapelets import learn_static_shapelets
from ..utils.gat import GAT, accuracy_torch
from ..utils.gat_utils import GATDataloader, GATDataset
from ..utils.base_utils import ModelUtils, evaluate_performance
from baselines.feature_based import FeatureModel


class Time2GraphGAT(ModelUtils):
    """
        Time2GraphGAT model
        Hyper-parameters:
            K: number of learned shapelets
            C: number of candidates
            A: number of shapelets assigned to each segment
            tflag: timing flag
    """
    def __init__(self, K, seg_length, num_segment, warp=2, tflag=True, gpu_enable=True, optimizer='Adam', n_hidden=8,
                 n_heads=8, dropout=0.2, lk_relu=0.2, data_size=1, out_clf=True, softmax=False,  percentile=80,
                 dataset='Unspecified', append=False, sort=False, ft_xgb=False,  opt_metric='f1', feat_flag=True,
                 feat_norm=True, aggregate=True, standard_scale=False, diff=False, **kwargs):
        super(Time2GraphGAT, self).__init__(kernel='xgb')
        self.K = K
        self.C = kwargs.pop('C', K * 10)
        self.seg_length = seg_length
        self.num_segment = num_segment
        self.data_size = data_size
        self.warp = warp
        self.tflag = tflag
        self.gpu_enable = gpu_enable
        self.cuda = self.gpu_enable and torch.cuda.is_available()
        # Debugger.info_print('torch.cuda: {}, self.cuda: {}'.format(torch.cuda.is_available(), self.cuda))
        self.shapelets = None
        self.append = append
        self.percentile = percentile
        self.threshold = None
        self.sort = sort
        self.aggregate = aggregate
        if self.append:
            self.n_features = self.num_segment + self.seg_length * self.data_size
        else:
            self.n_features = self.num_segment
        self.n_hidden = n_hidden
        self.n_heads = n_heads
        self.dropout = dropout
        self.lk_relu = lk_relu
        self.out_clf = out_clf
        self.softmax = softmax
        self.dataset = dataset
        self.diff = diff
        self.standard_scale = standard_scale
        self.opt_metric = opt_metric
        self.xgb = self.clf__(use_label_encoder=False,  eval_metric="logloss", verbosity=0)
        self.ft_xgb = ft_xgb
        self.fm = FeatureModel(seg_length=self.seg_length, kernel=self.kernel)
        self.fm_scaler = MinMaxScaler()
        self.feat_flag = feat_flag
        self.feat_norm = feat_norm
        self.pretrain = kwargs.pop('pretrain', None)

        self.gat = GAT(
            nfeat=self.n_features, nhid=self.n_hidden, nclass=kwargs.get('n_class', 2),
            dropout=self.dropout, nheads=self.n_heads, alpha=self.lk_relu, nnodes=self.K, aggregate=self.aggregate
        )
        self.gat = self.gat.double()
        self.lr = kwargs.pop('lr', 1e-3)
        self.p = kwargs.pop('p', 2)
        self.alpha = kwargs.pop('alpha', 0.1)
        self.beta = kwargs.pop('beta', 0.05)
        self.debug = kwargs.pop('debug', False)
        self.optimizer = optimizer
        self.measurement = kwargs.pop('measurement', 'gdtw')
        self.batch_size = kwargs.pop('batch_size', 200)
        self.init = kwargs.pop('init', 0)
        self.niter = kwargs.pop('niter', 1000)
        self.fastmode = kwargs.pop('fastmode', False)
        self.tol = kwargs.pop('tol', 1e-4)
        self.cuda = self.gpu_enable and torch.cuda.is_available()
        self.kwargs = kwargs
        Debugger.info_print('initialize time2graph+ model with {}'.format(self.__dict__))

    def learn_shapelets(self, x, y, num_segment, data_size):
        assert x.shape[1] == num_segment * self.seg_length
        Debugger.info_print('basic statistics before learn shapelets: max {:.4f}, min {:.4f}'.format(np.max(x), np.min(x)))
        if self.tflag:
            self.shapelets = learn_time_aware_shapelets(
                time_series_set=x, label=y, K=self.K, C=self.C, p=self.p,
                num_segment=num_segment, seg_length=self.seg_length, data_size=data_size,
                lr=self.lr, alpha=self.alpha, beta=self.beta, num_batch=int(x.shape[0] / self.batch_size),
                measurement=self.measurement, gpu_enable=self.gpu_enable, **self.kwargs)
        else:
            self.shapelets = learn_static_shapelets(
                time_series_set=x, label=y, K=self.K, C=self.C, warp=self.warp,
                num_segment=num_segment, seg_length=self.seg_length, measurement=self.measurement, **self.kwargs)

    def __gat_features__(self, X, train=False):
        __shapelet_distance = shapelet_distance(
            time_series_set=X, shapelets=self.shapelets, seg_length=self.seg_length,
            tflag=self.tflag, tanh=self.kwargs.get('tanh', False), debug=self.debug,
            init=self.init, warp=self.warp, measurement=self.measurement)
        threshold = None if train else self.threshold
        adj_matrix, self.threshold = adjacent_matrix(
            sdist=__shapelet_distance, num_time_series=X.shape[0], num_segment=int(X.shape[1] / self.seg_length),
            num_shapelet=self.K, percentile=self.percentile, threshold=threshold, debug=self.debug)
        __shapelet_distance = np.transpose(__shapelet_distance, axes=(0, 2, 1))
        if self.sort:
            __shapelet_distance = softmax(-1 * np.sort(__shapelet_distance, axis=1), axis=1)
        if self.softmax and not self.sort:
            __shapelet_distance = softmax(__shapelet_distance, axis=1)
        if self.append:
            origin = np.array([v[0].reshape(-1) for v in self.shapelets], dtype=np.float).reshape(1, self.K, -1)
            return np.concatenate((__shapelet_distance, np.tile(origin, (__shapelet_distance.shape[0], 1, 1))),
                                  axis=2).astype(np.float), adj_matrix
        else:
            return __shapelet_distance.astype(np.float), adj_matrix

    def __fit_gat(self, X, Y):
        feats, adj = self.__gat_features__(X=X, train=True)
        optimizer = optim.Adam(self.gat.parameters(), lr=self.lr, weight_decay=5e-4)
        weight = torch.DoubleTensor([float(sum(Y) / len(Y)), 1 - float(sum(Y) / len(Y))])
        if self.cuda:
            self.gat = self.gat.cuda()
            weight = weight.cuda()

        for epoch in range(self.niter):
            dataset = GATDataset(feat=feats, adj=adj, y=Y)
            dataloader = GATDataloader(dataset, batch_size=self.batch_size, shuffle=True)
            for i, (feat_batch, adj_batch, lb_batch) in enumerate(dataloader, 0):
                self.gat.train()
                optimizer.zero_grad()
                if self.cuda:
                    feat_batch = feat_batch.cuda()
                    adj_batch = adj_batch.cuda()
                    lb_batch = lb_batch.cuda()
                output_batch = self.gat(feat_batch, adj_batch)
                loss_train = F.nll_loss(output_batch, lb_batch, weight=weight)
                acc_train = accuracy_torch(output_batch, lb_batch)
                loss_train.backward()
                optimizer.step()
                if not self.fastmode:
                    # Evaluate validation set performance separately,
                    # deactivates dropout during validation run.
                    self.gat.eval()
                    output_batch = self.gat(feat_batch, adj_batch)
                loss_val = F.nll_loss(output_batch, lb_batch)
                acc_val = accuracy_torch(output_batch, lb_batch)
                Debugger.debug_print(
                    msg='Epoch: {:04d}-{:02d}, train loss: {:.4f} accu: {:.4f}, val loss: {:.4f}, accu: {:.4f}'.format(
                        epoch + 1, i + 1, loss_train.data.item(), acc_train.data.item(), loss_val.data.item(),
                        acc_val.data.item()), debug=self.debug)
        y_pred = self.___gat_predict(feat=feats, adj=adj)
        accu, prec, recall, f1 = evaluate_performance(y_pred=y_pred, y_true=Y)
        Debugger.info_print('fitting gat: accu {:.4f}, prec {:.4f}, recall {:.4f}, f1 {:.4f}'.format(
            accu, prec, recall, f1))

    def ___gat_predict(self, feat, adj):
        y_batch_list = []
        dataset = GATDataset(feat=feat, adj=adj)
        dataloader = GATDataloader(dataset, batch_size=self.batch_size, shuffle=False)
        for i, (feat_batch, adj_batch) in enumerate(dataloader, 0):
            self.gat.eval()
            if self.cuda:
                feat_batch = feat_batch.cuda()
                adj_batch = adj_batch.cuda()
            output_batch = self.gat(feat_batch, adj_batch)
            y_batch = output_batch.max(1)[1].type(torch.IntTensor)
            if self.cuda:
                y_batch_list.append(y_batch.data.cpu().numpy())
            else:
                y_batch_list.append(y_batch.data.numpy())
        return np.concatenate(y_batch_list, axis=0)

    def __gat_hidden_feature(self, feat, adj, X=None, train=False):
        feat_batch_list = []
        dataset = GATDataset(feat=feat, adj=adj)
        dataloader = GATDataloader(dataset, batch_size=self.batch_size, shuffle=False)
        for i, (feat_batch, adj_batch) in enumerate(dataloader, 0):
            self.gat.eval()
            if self.cuda:
                feat_batch = feat_batch.cuda()
                adj_batch = adj_batch.cuda()
            output_batch = self.gat(feat_batch, adj_batch, feat_flag=True)
            if self.cuda:
                feat_batch_list.append(output_batch.data.cpu().numpy())
            else:
                feat_batch_list.append(output_batch.data.numpy())
        feat_batch = np.concatenate(feat_batch_list, axis=0)
        if self.feat_flag:
            assert X is not None, 'time series data not provided when feat_flag is set as True'
            feat = self.fm.extract_features(samples=X)
            if train and self.feat_norm:
                feat = self.fm_scaler.fit_transform(feat)
            elif self.feat_norm:
                feat = self.fm_scaler.transform(feat)
            return np.concatenate((feat_batch, feat), axis=1)
        else:
            return feat_batch

    def __preprocess_input_data(self, X):
        X_scale = X.copy()
        if self.diff:
            X_scale[:, : -1, :] = X[:, 1:, :] - X[:, :-1, :]
            X_scale[:, -1, :] = 0
            Debugger.debug_print('conduct time differing...')
        if self.standard_scale:
            for i in range(self.data_size):
                X_std = np.std(X_scale[:, :, i], axis=1).reshape(X.shape[0], -1)
                X_std[X_std == 0] = 1.0
                X_scale[:, :, i] = (X_scale[:, :, i] - np.mean(X_scale[:, :, i], axis=1).reshape(X.shape[0], -1)) / X_std
                Debugger.debug_print('conduct standard scaling on data-{}, with mean {:.2f} and var {:.2f}'.format(i, np.mean(X_scale[0, :, i]), np.std(X_scale[0, :, i])))
        return X_scale

    def fit(self, X, Y, n_splits=5, reset=False):
        num_segment, data_size = int(X.shape[1] / self.seg_length), X.shape[-1]
        assert self.data_size == X.shape[-1]
        X_scale = self.__preprocess_input_data(X)

        if reset or self.shapelets is None:
            self.learn_shapelets(x=X_scale, y=Y, num_segment=num_segment, data_size=data_size)
        self.__fit_gat(X=X_scale, Y=Y)
        X_feat, X_adj = self.__gat_features__(X_scale)

        if self.out_clf:
            if not self.ft_xgb:
                if self.pretrain is not None:
                    assert isinstance(self.pretrain, dict)
                    Debugger.info_print('setting pretrained parameters: {}'.format(self.pretrain))
                    # set default parameter explicitly
                    self.xgb = self.clf__(
                        **self.pretrain, verbosity=0, use_label_encoder=False, eval_metric="logloss")
                else:
                    Debugger.info_print('using default xgboost parameters')
                self.xgb.fit(self.__gat_hidden_feature(feat=X_feat, adj=X_adj, X=X, train=True), Y)
            else:
                skf = StratifiedKFold(n_splits=n_splits, shuffle=True)
                x_train, y_train, x_val, y_val = [], [], [], []
                for train_idx, test_idx in skf.split(X, Y):
                    x_train.append(X[train_idx, :])
                    y_train.append(Y[train_idx])
                    x_val.append(X[test_idx, :])
                    y_val.append(Y[test_idx])
                self.__fine_tune_xgboost_on_validation__(
                    x_train=x_train, y_train=y_train, x_val=x_val, y_val=y_val, nested=True)

        y_pred = self.___gat_predict(feat=X_feat, adj=X_adj)
        accu, prec, recall, f1 = evaluate_performance(y_pred=y_pred, y_true=Y)
        Debugger.info_print('fully-connected-layer res on training set: {:.4f}, {:.4f}, {:.4f}, {:.4f}'.format(
            accu, prec, recall, f1))
        if self.out_clf:
            y_pred = self.xgb.predict(self.__gat_hidden_feature(feat=X_feat, adj=X_adj, X=X))
            accu, prec, recall, f1 = evaluate_performance(y_pred=y_pred, y_true=Y)
            Debugger.info_print('out-classifier on training set: {:.4f}, {:.4f}, {:.4f}, {:.4f}'.format(
                accu, prec, recall, f1))

    def __fine_tune_xgboost_on_validation__(self, x_train, y_train, x_val, y_val, nested=True):
        assert self.out_clf, 'outer-classifier not exist: out_clf = {}'.format(self.out_clf)
        if not nested:
            x_train, y_train, x_val, y_val = [x_train], [y_train], [x_val], [y_val]
        feat_train, adj_train, feat_val, adj_val = [], [], [], []
        num_split = len(x_train)
        for idx in range(num_split):
            x_train_scale = self.__preprocess_input_data(x_train[idx])
            x_val_scale = self.__preprocess_input_data(x_val[idx])
            feat_train_, adj_train_ = self.__gat_features__(x_train_scale)
            feat_val_, adj_val_ = self.__gat_features__(x_val_scale)
            feat_train.append(feat_train_)
            adj_train.append(adj_train_)
            feat_val.append(feat_val_)
            adj_val.append(adj_val_)
        num_args = len(list(self.clf_paras(balanced=True)))
        arguments = self.clf_paras(balanced=True)
        max_res, max_args, cnt = -1, None, 0
        metric_measure = self.return_metric_method(opt_metric=self.opt_metric)
        for args in arguments:
            cnt += 1
            Debugger.debug_print('[{}]/[{}] fine-tune for xgboost-args: {} ######'.format(cnt, num_args, args), debug=self.debug)
            self.xgb = self.clf__(
                **args, verbosity=0, use_label_encoder=False, eval_metric="logloss"
            )
            tmp_metric_res = 0
            for idx in range(num_split):
                self.xgb.fit(
                    self.__gat_hidden_feature(feat=feat_train[idx], adj=adj_train[idx], X=x_train[idx], train=True),
                    y_train[idx])
                y_pred = self.xgb.predict(self.__gat_hidden_feature(feat=feat_val[idx], adj=adj_val[idx], X=x_val[idx]))
                tmp_metric_res += metric_measure(y_pred=y_pred, y_true=y_val[idx])
            if tmp_metric_res > max_res:
                max_res = tmp_metric_res
                max_args = args
            Debugger.debug_print('[{}]/[{}] fine-tune for xgboost-args on {}: cur {:.4f}, best {:.4f}'.format(
                cnt, num_args, self.opt_metric, tmp_metric_res, max_res))
        Debugger.info_print('fine-tuning on xgb({}):{:.4f}, with best arguments: {}'.format(
            self.opt_metric, max_res / num_split, max_args))
        self.xgb = self.clf__(
            **max_args, verbosity=0, use_label_encoder=False, eval_metric="logloss"
        )
        x_train_all, y_train_all = np.concatenate(x_train, axis=0), np.concatenate(y_train, axis=0)
        x_train_all_scale = self.__preprocess_input_data(x_train_all)
        feat_train_all, adj_train_all = self.__gat_features__(x_train_all_scale)
        self.xgb.fit(self.__gat_hidden_feature(
            feat=feat_train_all, adj=adj_train_all, X=x_train_all, train=True), y_train_all)

    # @NOTE: for debugging
    def __fit_gat__(self, X, Y):
        self.__fit_gat(self.__preprocess_input_data(X), Y)

    def predict(self, X):
        assert self.shapelets is not None, 'shapelets has not been learnt yet...'
        X_scale = self.__preprocess_input_data(X)
        x, adj = self.__gat_features__(X_scale)
        if self.out_clf:
            return self.xgb.predict(self.__gat_hidden_feature(feat=x, adj=adj, X=X))
        else:
            return self.___gat_predict(feat=x, adj=adj)

    def save_model(self, fpath):
        ret = {}
        for key, val in self.__dict__.items():
            if key != 'xgb':
                ret[key] = val
        self.xgb.save_model('{}.xgboost'.format(fpath))
        torch.save(ret, fpath)

    def load_model(self, fpath, map_location='cuda:0'):
        # @TODO: specify map_location
        cache = torch.load(fpath, map_location=map_location)
        for key, val in cache.items():
            self.__dict__[key] = val
        self.xgb.load_model('{}.xgboost'.format(fpath))

    def save_shapelets(self, fpath):
        torch.save(self.shapelets, fpath)

    def load_shapelets(self, fpath, map_location='cuda:0'):
        self.shapelets = torch.load(fpath, map_location=map_location)
