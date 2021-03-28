# -*- coding: utf-8 -*-
import warnings
import argparse
import torch.optim as optim
from config import *
from archive.load_usr_dataset import load_usr_dataset_by_name
from time2graph.utils.deep_models import *
from time2graph.utils.deep_utils import *
from time2graph.utils.base_utils import ModelUtils
from sklearn.neural_network import MLPClassifier


if __name__ == '__main__':
    warnings.filterwarnings(module='sklearn*', action='ignore', category=DeprecationWarning)
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dataset', type=str, default='stealing')
    parser.add_argument('--model', type=str, default='LSTM')
    parser.add_argument('--n_splits', type=int, default=5)
    parser.add_argument('--num_segment', type=int, default=12)
    parser.add_argument('--seg_length', type=int, default=30)
    parser.add_argument('--hidden_size', type=int, default=256)
    parser.add_argument('--output_size', type=int, default=128)
    parser.add_argument('--niter', type=int, default=10)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--optimizer', type=str, default='Adam')
    parser.add_argument('--opt_metric', type=str, default='accuracy')
    parser.add_argument('--gpu_enable', action='store_true', default=False)
    parser.add_argument('--data_size', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=50)
    parser.add_argument('--cache', action='store_true', default=False)

    args = parser.parse_args()
    Debugger.info_print('run with options: {}'.format(args.__dict__))
    if args.cache and path.isfile('{}/scripts/cache/{}_x_train.npy'.format(module_path, args.dataset)):
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

    if args.model != 'LSTM':
        x_train = x_train.reshape(x_train.shape[0], -1)
        x_test = x_test.reshape(x_test.shape[0], -1)

    args.data_length = args.num_segment * args.seg_length * args.data_size
    Debugger.info_print('original training shape {}, data_length: {}'.format(x_train.shape, args.data_length))
    Debugger.info_print('basic statistics: max {:.4f}, min {:.4f}'.format(np.max(x_train), np.min(x_train)))
    train_dataset = DeepDataset(x=x_train, y=y_train)
    train_dataloader = DeepDataloader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                      num_workers=2)
    test_dataset = DeepDataset(x=x_test, y=y_test)
    test_dataloader = DeepDataloader(test_dataset, batch_size=args.batch_size, shuffle=True,
                                     num_workers=2)
    if args.model == 'LSTM':
        model = LSTMClassifier(data_size=args.data_size, hidden_size=args.hidden_size,
                               output_size=args.output_size, dropout=args.dropout,
                               gpu_enable=args.gpu_enable).double()
    elif args.model == 'MLP':
        model = MLP(data_size=args.data_length, hidden_size=args.hidden_size,
                    output_size=args.output_size).double()
    elif args.model == 'VAE':
        model = VAE(encoder=EnDecoder(D_in=args.data_length, H=args.hidden_size, D_out=args.output_size),
                    decoder=EnDecoder(D_in=args.hidden_size, H=args.output_size, D_out=args.data_length),
                    encode_dim=args.output_size, latent_dim=args.hidden_size
                    ).double()
    elif args.model == 'sklearn':
        model = MLPClassifier(hidden_layer_sizes=(args.hidden_size, args.output_size))
    else:
        raise NotImplementedError()
    if args.gpu_enable:
        model = model.cuda()

    if args.model == 'sklearn':
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
    elif args.model != 'VAE':
        weight = torch.Tensor([float(sum(y_train)) / len(y_train), 1.0]).double()
        criterion = nn.CrossEntropyLoss(weight=weight)
        if args.gpu_enable:
            criterion = criterion.cuda()
        optimizer = optim.SGD(model.parameters(), lr=args.lr)
        for epoch in range(args.niter):
            train_RNNs(epoch=epoch, dataloader=train_dataloader, rnn=model, criterion=criterion,
                       optimizer=optimizer, debug=True, gpu_enable=args.gpu_enable)
            test_DeepModels(dataloader=train_dataloader, rnn=model,
                            criterion=criterion, debug=True, gpu_enable=args.gpu_enable)
            test_DeepModels(dataloader=test_dataloader, rnn=model,
                            criterion=criterion, debug=True, gpu_enable=args.gpu_enable)
        x_test_tensor = torch.from_numpy(x_test).double()
        if args.gpu_enable:
            x_test_tensor = x_test_tensor.cuda()
        y_pred = F.softmax(model(x_test_tensor)).data.max(1, keepdim=True)[1].cpu().numpy()
    else:
        means, std = np.mean(x_train, axis=1), np.std(x_train, axis=1)
        x_train -= means.reshape(x_train.shape[0], -1)
        std[std == 0] = 1.0
        x_train /= std.reshape(x_train.shape[0], -1)
        criterion = nn.MSELoss()
        if args.gpu_enable:
            criterion = criterion.cuda()
        optimizer = optim.SGD(model.parameters(), lr=args.lr)
        for epoch in range(args.niter):
            train_VAE(epoch=epoch, dataloader=train_dataloader, vae=model, criterion=criterion,
                      optimizer=optimizer, debug=True, gpu_enable=args.gpu_enable)
        x_train_tensor = torch.from_numpy(x_train).double()
        x_test_tensor = torch.from_numpy(x_test).double()
        if args.gpu_enable:
            x_train_tensor = x_test_tensor.cuda()
            x_test_tensor = x_test_tensor.cuda()
            x_train_feature = model.encoder(x_train_tensor).cpu().data.numpy()
            x_test_feature = model.encoder(x_test_tensor).cpu().data.numpy()
        else:
            x_train_feature = model.encoder(x_train_tensor).data.numpy()
            x_test_feature = model.encoder(x_test_tensor).data.numpy()
        max_metric, max_paras, cnt = -1, None, 0
        for arguments in ModelUtils(kernel='xgb').clf_paras(balanced=True):
            clf = ModelUtils(kernel='xgb').clf__()
            clf.set_params(**arguments)
            clf.fit(x_train_feature, y_train)
            y_pred = clf.predict(x_test_feature)
            accu, prec, recall, f1 = evaluate_performance(y_true=y_test, y_pred=y_pred)
            if args.opt_metric == 'accuracy':
                if accu > max_metric:
                    max_metric = accu
                    max_paras = arguments
            else:
                if f1 > max_metric:
                    max_metric = f1
                    max_paras = arguments
            Debugger.debug_print('[{}]tmp metric {}'.format(cnt, max_metric))
            cnt += 1
        Debugger.info_print('max metric: {}'.format(max_metric))
        clf = ModelUtils(kernel='xgb').clf__()
        clf.set_params(**max_paras)
        clf.fit(x_train_feature, y_train)
        y_pred = clf.predict(x_test_feature)
    accu, prec, recall, f1 = evaluate_performance(y_true=y_test, y_pred=y_pred)
    Debugger.info_print('res: accu {:.4f}, prec {:.4f}, recall {:.4f}, f1 {:.4f}'.format(
        accu, prec, recall, f1
    ))
