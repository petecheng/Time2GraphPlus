import warnings
import argparse
from config import *
from baselines.feature_based import FeatureModel, ModelUtils
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from time2graph.utils.deep_utils import *
from time2graph.utils.base_utils import evaluate_performance
from time2graph.core.model_sequence import Time2GraphSequence


class RawModel(ModelUtils):
    def __init__(self, kernel='xgb', opt_metric='f1', **kwargs):
        super(RawModel, self).__init__(kernel=kernel, **kwargs)
        self.opt_metric = opt_metric
        self.clf = self.clf__()

    def fit(self, x, lb, n_splits=5, balanced=True):
        max_accu, max_prec, max_recall, max_f1, max_metric = -1, -1, -1, -1, -1
        arguments, opt_args = self.clf_paras(balanced=balanced), None
        metric_measure = self.return_metric_method(opt_metric=self.opt_metric)
        for args in arguments:
            self.clf.set_params(**args)
            skf = StratifiedKFold(n_splits=n_splits, shuffle=True)
            tmp = np.zeros(5, dtype=np.float32).reshape(-1)
            measure_vector = [metric_measure, accuracy_score, precision_score, recall_score, f1_score]
            for train_idx, test_idx in skf.split(x, lb):
                self.clf.fit(x[train_idx], lb[train_idx])
                y_pred, y_true = self.clf.predict(x[test_idx]), lb[test_idx]
                for k in range(5):
                    tmp[k] += measure_vector[k](y_true=y_true, y_pred=y_pred)
            tmp /= n_splits
            if max_metric < tmp[0]:
                max_metric = tmp[0]
                opt_args = args
                max_accu, max_prec, max_recall, max_f1 = tmp[1:]
        Debugger.info_print('args {} for clf {}, performance: {:.4f}, {:.4f}, {:.4f}, {:.4f}'.format(
            opt_args, self.kernel, max_accu, max_prec, max_recall, max_f1))
        self.clf.set_params(**opt_args)

    def predict(self, x):
        return self.clf.predict(x)


if __name__ == '__main__':
    warnings.filterwarnings(module='sklearn*', action='ignore', category=DeprecationWarning)
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dataset', type=str, default='stealing')
    parser.add_argument('--cache', action='store_true', default=False)
    parser.add_argument('--features', action='store_true', default=False)
    parser.add_argument('--sequence', action='store_true', default=False)
    parser.add_argument('--K', type=int, default=100)
    parser.add_argument('--seg_length', type=int, default=30)
    parser.add_argument('--num_segment', type=int, default=12)

    args = parser.parse_args()
    Debugger.info_print('run with options: {}'.format(args.__dict__))
    if args.cache and path.isfile('{}/scripts/cache/{}_x_train.npy'.format(module_path, args.dataset)):
        x_train = np.load('{}/scripts/cache/{}_x_train.npy'.format(module_path, args.dataset))
        y_train = np.load('{}/scripts/cache/{}_y_train.npy'.format(module_path, args.dataset))
        x_test = np.load('{}/scripts/cache/{}_x_test.npy'.format(module_path, args.dataset))
        y_test = np.load('{}/scripts/cache/{}_y_test.npy'.format(module_path, args.dataset))
    else:
        raise NotImplementedError()

    x_train = x_train[:, :args.seg_length * args.num_segment, :]
    x_test = x_test[:, :args.seg_length * args.num_segment, :]
    Debugger.info_print('input training sequence shape: {}'.format(x_train.shape))

    if args.sequence:
        model = Time2GraphSequence(K=args.K, C=args.K * 10, seg_length=args.seg_length)
        model.load_shapelets(fpath='{}/scripts/cache/{}_greedy_{}_{}_shapelets.cache'.format(
            module_path, args.dataset, args.K, args.seg_length))
        Debugger.info_print('load shapelets from {}/scripts/cache/{}_greedy_{}_{}_shapelets.cache'.format(
            module_path, args.dataset, args.K, args.seg_length))
    elif not args.features:
        model = RawModel()
        x_train = x_train.reshape(x_train.shape[0], -1)
        x_test = x_test.reshape(x_test.shape[0], -1)
    else:
        model = FeatureModel(seg_length=30)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    accu, prec, recall, f1 = evaluate_performance(y_true=y_test, y_pred=y_pred)
    Debugger.info_print('res: accu {:.4f}, prec {:.4f}, recall {:.4f}, f1 {:.4f}'.format(accu, prec, recall, f1))
