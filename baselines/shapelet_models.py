import warnings
import argparse
from config import *
from time2graph.utils.base_utils import evaluate_performance
from tslearn.shapelets import ShapeletModel
"""
    Baseline: Learning Shapelets using tslearn.
"""

if __name__ == '__main__':
    warnings.filterwarnings(module='sklearn*', action='ignore', category=DeprecationWarning)
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dataset', type=str, default='stealing')
    parser.add_argument('--model', type=str, default='LS')
    parser.add_argument('--K', type=int, default=100)
    parser.add_argument('--num_segment', type=int, default=12)
    parser.add_argument('--seg_length', type=int, default=30)
    parser.add_argument('--cache', action='store_true', default=False)

    args = parser.parse_args()
    origin_seg_length = args.seg_length
    Debugger.info_print('run with options: {}'.format(args.__dict__))
    if args.cache and path.isfile('{}/scripts/cache/{}_x_train.npy'.format(module_path, args.dataset)):
        x_train = np.load('{}/scripts/cache/{}_x_train.npy'.format(module_path, args.dataset))
        y_train = np.load('{}/scripts/cache/{}_y_train.npy'.format(module_path, args.dataset))
        x_test = np.load('{}/scripts/cache/{}_x_test.npy'.format(module_path, args.dataset))
        y_test = np.load('{}/scripts/cache/{}_y_test.npy'.format(module_path, args.dataset))
    else:
        raise NotImplementedError()

    if args.model == 'LS':
        Debugger.info_print('begin to train LS model...')
        model = ShapeletModel(n_shapelets_per_size={origin_seg_length: args.K}, max_iter=100)
        model.fit(x_train, y_train)
        Debugger.info_print('LS model training done.')
        y_pred = model.predict(x_test)
        accu, prec, recall, f1 = evaluate_performance(y_true=y_test, y_pred=y_pred)
        Debugger.info_print('res: accu {:.4f}, prec {:.4f}, recall {:.4f}, f1 {:.4f}'.format(accu, prec, recall, f1))
    else:
        raise NotImplementedError()
