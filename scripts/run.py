# -*- coding: utf-8 -*-
import argparse
import warnings
from pathos.helpers import mp
from time2graph.core.model_gat import Time2GraphGAT
from config import *
from archive.load_usr_dataset import load_usr_dataset_by_name
from time2graph.utils.base_utils import Debugger, evaluate_performance
warnings.filterwarnings(action='ignore', category=DeprecationWarning)
warnings.filterwarnings(action='ignore', category=FutureWarning)
warnings.filterwarnings(action='ignore', category=RuntimeWarning)
warnings.filterwarnings(action='ignore', category=FutureWarning)


if __name__ == '__main__':
    mp.set_start_method('spawn')
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dataset', type=str, default='ucr-Earthquakes')
    parser.add_argument('--n_splits', type=int, default=5)
    parser.add_argument('--nhidden', type=int, default=8)
    parser.add_argument('--nheads', type=int, default=8)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--relu', type=float, default=0.2)
    parser.add_argument('--data_size', type=int, default=1)
    parser.add_argument('--opt_metric', type=str, default='f1')

    parser.add_argument('--niter', type=int, default=1000)
    parser.add_argument('--njobs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--percentile', type=int, default=80)

    parser.add_argument('--diff', action='store_true', default=False)
    parser.add_argument('--standard_scale', action='store_true', default=False)

    parser.add_argument('--softmax', action='store_true', default=False)
    parser.add_argument('--append', action='store_true', default=False)
    parser.add_argument('--sort', action='store_true', default=False)
    parser.add_argument('--ft_xgb', action='store_true', default=False)
    parser.add_argument('--aggregate', action='store_true', default=False)
    parser.add_argument('--feat_flag', action='store_true', default=False)
    parser.add_argument('--feat_norm', action='store_true', default=False)
    parser.add_argument('--debug', action='store_true', default=False)

    parser.add_argument('--model_cache', action='store_true', default=False)
    parser.add_argument('--shapelet_cache', action='store_true', default=False)
    parser.add_argument('--gpu_enable', action='store_true', default=False)
    parser.add_argument('--pretrain', action='store_true', default=False)
    parser.add_argument('--finetune', action='store_true', default=False)

    args = parser.parse_args()
    Debugger.info_print('run with options: {}'.format(args.__dict__))
    if path.isfile('{}/scripts/cache/{}_x_train.npy'.format(module_path, args.dataset)):
        x_train = np.load('{}/scripts/cache/{}_x_train.npy'.format(module_path, args.dataset))
        y_train = np.load('{}/scripts/cache/{}_y_train.npy'.format(module_path, args.dataset))
        x_test = np.load('{}/scripts/cache/{}_x_test.npy'.format(module_path, args.dataset))
        y_test = np.load('{}/scripts/cache/{}_y_test.npy'.format(module_path, args.dataset))
    elif args.dataset.startswith('ucr'):
        dataset = args.dataset.rstrip('\n\r').split('-')[-1]
        x_train, y_train, x_test, y_test = load_usr_dataset_by_name(
            fname=dataset, length=args.seg_length * args.num_segment)
    else:
        raise NotImplementedError()
    Debugger.info_print('original training shape {}'.format(x_train.shape))
    Debugger.info_print('basic statistics: max {:.4f}, min {:.4f}'.format(np.max(x_train), np.min(x_train)))
    general_options = {
        'kernel': 'xgb',
        'opt_metric': args.opt_metric,
        'init': 0,
        'warp': 2,
        'tflag': True,
        'mode': 'embedding',
        'candidate_method': 'greedy'
    }
    model_options = model_args[args.dataset]
    xgb_options = xgb_args.get(args.dataset, {})
    xgb_options['n_jobs'] = njobs
    pretrain = None if not args.pretrain else xgb_options
    model = Time2GraphGAT(
        gpu_enable=args.gpu_enable, n_hidden=args.nhidden, n_heads=args.nheads, dropout=args.dropout, lk_relu=args.relu,
        out_clf=True, softmax=args.softmax, data_size=args.data_size, dataset=args.dataset, njobs=args.njobs,
        niter=args.niter, batch_size=args.batch_size, append=args.append, sort=args.sort, ft_xgb=args.ft_xgb,
        ggregate=args.aggregate, feat_flag=args.feat_flag, feat_norm=args.feat_norm, pretrain=pretrain, diff=args.diff,
        standard_scale=args.standard_scale, percentile=args.percentile, **general_options, **model_options, debug=args.debug
    )
    res = np.zeros(4, dtype=np.float32)
    Debugger.info_print('in this split: training {} samples, with {} positive'.format(
        len(x_train), sum(y_train)))
    tflag_str = 'time_aware' if general_options['tflag'] else 'static'
    shapelet_cache = '{}/scripts/cache/{}_{}_{}_{}_shapelets.cache'.format(
        module_path, args.dataset, model_options['K'], model_options['seg_length'], tflag_str)

    if path.isfile(shapelet_cache) and args.shapelet_cache:
        model.load_shapelets(fpath=shapelet_cache)
        Debugger.info_print('load shapelets from {}'.format(shapelet_cache))
    else:
        Debugger.info_print('train_size {}, label size {}'.format(x_train.shape, y_train.shape))
        model.learn_shapelets(x=x_train, y=y_train, num_segment=model_options['num_segment'], data_size=args.data_size)
        Debugger.info_print('learn {} shapelets done...'.format(tflag_str))
        model.save_shapelets(shapelet_cache)

    Debugger.info_print('training: {:.2f} positive ratio with {}'.format(float(sum(y_train) / len(y_train)),
                                                                         len(y_train)))
    Debugger.info_print('test: {:.2f} positive ratio with {}'.format(float(sum(y_test) / len(y_test)),
                                                                     len(y_test)))

    model.fit(X=x_train, Y=y_train, n_splits=args.n_splits)
    accu, prec, recall, f1 = evaluate_performance(y_true=y_test, y_pred=model.predict(X=x_test))
    Debugger.info_print('res: accu {:.4f}, prec {:.4f}, recall {:.4f}, f1 {:.4f}'.format(accu, prec, recall, f1))
    model_cache = '{}/scripts/cache/{}_ttgp_{}.cache'.format(module_path, args.dataset, tflag_str)
    model.save_model(model_cache)


