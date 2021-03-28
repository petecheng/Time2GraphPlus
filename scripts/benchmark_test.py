# -*- coding: utf-8 -*-
"""
    test scripts on three benchmark datasets: EQS, WTC, STB
"""
import argparse
import warnings
from pathos.helpers import mp
from config import *
from archive.load_usr_dataset import load_usr_dataset_by_name
from time2graph.utils.base_utils import Debugger, evaluate_performance
from time2graph.core.model_gat import Time2GraphGAT


if __name__ == '__main__':
    mp.set_start_method('spawn')
    warnings.filterwarnings(module='sklearn*', action='ignore', category=DeprecationWarning)
    warnings.filterwarnings(action='ignore', category=RuntimeWarning)
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dataset', type=str, default='ucr-Earthquakes')
    parser.add_argument('--n_splits', type=int, default=5)
    parser.add_argument('--nhidden', type=int, default=8)
    parser.add_argument('--nheads', type=int, default=8)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--relu', type=float, default=0.2)
    parser.add_argument('--data_size', type=int, default=1)

    parser.add_argument('--niter', type=int, default=1000)
    parser.add_argument('--njobs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=100)

    parser.add_argument('--diff', action='store_true', default=False)
    parser.add_argument('--standard_scale', action='store_true', default=False)

    parser.add_argument('--softmax', action='store_true', default=False)
    parser.add_argument('--append', action='store_true', default=False)
    parser.add_argument('--sort', action='store_true', default=False)
    parser.add_argument('--ft_xgb', action='store_true', default=False)
    parser.add_argument('--aggregate', action='store_true', default=False)
    parser.add_argument('--feat_flag', action='store_true', default=False)
    parser.add_argument('--feat_norm', action='store_true', default=False)

    parser.add_argument('--model_cache', action='store_true', default=False)
    parser.add_argument('--shapelet_cache', action='store_true', default=False)
    parser.add_argument('--gpu_enable', action='store_true', default=False)
    parser.add_argument('--pretrain', action='store_true', default=False)
    parser.add_argument('--finetune', action='store_true', default=False)

    args = parser.parse_args()
    Debugger.info_print('running with {}'.format(args.__dict__))

    # set default options
    general_options = {
        'kernel': 'xgb',
        'opt_metric': 'accuracy',
        'init': 0,
        'warp': 2,
        'tflag': True,
        'mode': 'embedding',
        'candidate_method': 'greedy'
    }
    model_options = model_args[args.dataset]
    xgb_options = xgb_args[args.dataset]
    xgb_options['n_jobs'] = njobs

    # load benchmark dataset
    if args.dataset.startswith('ucr'):
        dataset = args.dataset.rstrip('\n\r').split('-')[-1]
        x_train, y_train, x_test, y_test = load_usr_dataset_by_name(
            fname=dataset, length=model_options['seg_length'] * model_options['num_segment'])
    else:
        raise NotImplementedError()
    Debugger.info_print('training: {:.2f} positive ratio with {}'.format(
        float(sum(y_train) / len(y_train)), len(y_train)))
    Debugger.info_print('test: {:.2f} positive ratio with {}'.format(
        float(sum(y_test) / len(y_test)), len(y_test)))

    # initialize Time2Graph+ model
    pretrain = None if not args.pretrain else xgb_options
    m = Time2GraphGAT(
        gpu_enable=args.gpu_enable, n_hidden=args.nhidden, n_heads=args.nheads, dropout=args.dropout, lk_relu=args.relu,
        out_clf=True, softmax=args.softmax, data_size=args.data_size, dataset=args.dataset, njobs=args.njobs,
        niter=args.niter, batch_size=args.batch_size, append=args.append, sort=args.sort, ft_xgb=args.ft_xgb,
        ggregate=args.aggregate, feat_flag=args.feat_flag, feat_norm=args.feat_norm, pretrain=pretrain,
        diff=args.diff, standard_scale=args.standard_scale, **general_options, **model_options
    )
    shapelet_cache_path = '{}/scripts/cache/{}_time_aware_shapelets_{}.cache'.format(
            module_path, args.dataset, model_options['seg_length'])
    model_cache_path = '{}/scripts/cache/{}_time2gat.cache'.format(module_path, args.dataset)
    if args.shapelet_cache and path.isfile(shapelet_cache_path):
        m.load_shapelets(fpath=shapelet_cache_path)
    else:
        m.learn_shapelets(x=x_train, y=y_train, num_segment=model_options['num_segment'], data_size=args.data_size)
        m.save_shapelets(fpath=shapelet_cache_path)

    if args.model_cache and path.isfile(model_cache_path):
        m.load_model(fpath=model_cache_path)
    elif args.finetune:
        Debugger.info_print('fine-tuning on the dataset: {}'.format(args.dataset))
        m.__fit_gat__(x_train, y_train)
        m.__fine_tune_xgboost_on_validation__(x_train=x_train, y_train=y_train, x_val=x_test, y_val=y_test, nested=False)
        m.save_model(fpath=model_cache_path)
    else:
        Debugger.info_print('training time2graph+ model on {}'.format(args.dataset))
        m.fit(X=x_train, Y=y_train, n_splits=args.n_splits)
        m.save_model(fpath=model_cache_path)

    Debugger.info_print('evaluating time2graph+ model on {}'.format(args.dataset))
    accu, prec, recall, f1 = evaluate_performance(y_true=y_test, y_pred=m.predict(X=x_test))
    Debugger.info_print('classification result: accuracy {:.4f}, precision {:.4f}, recall {:.4f}, F1 {:.4f}'.format(
            accu, prec, recall, f1
        ))

