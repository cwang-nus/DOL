import argparse
import os
import torch
import numpy as np
import uuid
import datetime
import importlib
from utils.logging import get_logger
from utils.tools import init_dl_program, StandardScaler

parser = argparse.ArgumentParser(description='[DOL] Distribution-aware Online Learning Framework')

parser.add_argument('--data', type=str, default='chicago-t', help='data')
parser.add_argument('--method', type=str, default='dol')
parser.add_argument('--mode', type=str, default='train', help='online or train')

parser.add_argument('--root_path', type=str, default='./datasets/', help='root path of the data file')
parser.add_argument('--save_path', type=str, default='./results/', help='the path to save the output')
parser.add_argument('--adj_filename', type=str, default='', help='the adj file path')
parser.add_argument('--checkpoint_path', type=str, default='', help='pretrain checkpoint path')
parser.add_argument('--features', type=str, default='M',
                    help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
parser.add_argument('--freq', type=str, default='t',
                    help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')

parser.add_argument('--seq_len', type=int, default=12, help='input sequence length of Informer encoder')
parser.add_argument('--pred_len', type=int, default=12, help='prediction sequence length')
parser.add_argument('--dropout', type=float, default=0.05, help='dropout')
parser.add_argument('--embed', type=str, default='timeF',
                    help='time features encoding, options:[timeF, fixed, learned]')
parser.add_argument('--activation', type=str, default='gelu', help='activation')
parser.add_argument('--num_workers', type=int, default=0, help='data loader num workers')
parser.add_argument('--n_inner', type=int, default=1)
parser.add_argument('--itr', type=int, default=1, help='experiments times')
parser.add_argument('--seed', type=int, default=42, help='seed')
parser.add_argument('--train_epochs', type=int, default=150, help='train epochs')
parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
parser.add_argument('--patience', type=int, default=10, help='early stopping patience')

parser.add_argument('--opt', type=str, default='adamw')
parser.add_argument('--learning_rate', type=float, default=0.001, help='optimizer learning rate')
parser.add_argument('--weight_decay', type=float, default=1e-3, help='optimizer learning rate')
parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)
parser.add_argument('--aug', type=int, default=0, help='Training with augmentation data aug iterations')
parser.add_argument('--lr_test', type=float, default=1e-3, help='learning rate during test')
parser.add_argument('--use_adbfgs', action='store_true', help='use the Adbfgs optimizer', default=True)
parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
parser.add_argument('--gpu', type=int, default=0, help='gpu')
parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')
parser.add_argument('--test_bsz', type=int, default=1)
parser.add_argument('--update_type', type=str, default='adapter')
parser.add_argument('--loss_type', type=str, default='maskedmae')
parser.add_argument('--awake_week_num', type=int, default=1)
parser.add_argument('--hib_week_num', type=int, default=1)

parser.add_argument('--awake_data', action='store_false', help='inverse output data', default=True)
parser.add_argument('--buffer_size', type=int, default=1000, help='the slots in streaming memory buffer')
parser.add_argument('--lsa_dim', type=int, default=4)
parser.add_argument('--lsa_num', type=int, default=2)
parser.add_argument('--mem_size', type=int, default=8)

parser.add_argument('--exp_name', type=str, default='')

args = parser.parse_args()

args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False
args.test_bsz = args.batch_size if args.test_bsz == -1 else args.test_bsz
if args.use_gpu and args.use_multi_gpu:
    args.devices = args.devices.replace(' ', '')
    device_ids = args.devices.split(',')
    args.device_ids = [int(id_) for id_ in device_ids]
    args.gpu = args.device_ids[0]

data_parser = {
    'chicago-t': {'data': 'chicago-t/chicago20_23.npz', 'adj_filename': 'chicago-t/adj_chicago.npy', 'node_num': 77, 'one_week_interval': 672, 'time_interval': 15},
    'singapore-t': {'data': 'singapore-t/15mins_0206_0806.npz', 'adj_filename': 'singapore-t/adj.npy', 'node_num': 87, 'one_week_interval': 672, 'time_interval': 15},
    'metr-la': {'data': 'metr-la/metr-la.h5', 'node_num': 207, 'adj_filename': 'metr-la/adj_mx_la.pkl', 'one_week_interval': 2016, 'time_interval': 5},
    'pems-bay': {'data': 'pems-bay/pems-bay.h5', 'node_num': 325, 'adj_filename': 'pems-bay/adj_mx_bay.pkl', 'one_week_interval': 2016, 'time_interval': 5},
}


if args.data in data_parser.keys():
    data_info = data_parser[args.data]
    args.data_path = data_info['data']
    args.node_num = data_info['node_num']
    args.adj_filename = data_info['adj_filename']
    args.one_week_interval = data_info['one_week_interval']
    args.time_interval = data_info['time_interval']

Exp = getattr(importlib.import_module('exp.exp_{}'.format(args.method)), 'Exp')

metrics, preds, true, mae, mse = [], [], [], [], []

for ii in range(args.itr):
    print('\n ====== Run {} ====='.format(ii))
    # setting record of experiments
    method_name = args.method
    uid = uuid.uuid4().hex[:4]
    suffix = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M") + "_" + uid
    setting = '{}_{}_{}_{}_{}_{}_{}_{}_lsa{}_{}_{}_{}_{}'. \
        format(args.data, args.exp_name, args.awake_data, method_name, args.update_type,
               args.pred_len, args.opt,
               args.test_bsz, suffix,
               args.lsa_num, args.lsa_dim, args.mem_size, args.buffer_size)

    seed = args.seed + ii

    if args.use_gpu:
        init_dl_program(args.gpu, seed=seed)
    args.finetune_model_seed = seed

    scaler = StandardScaler()

    args.log_dir = args.save_path + setting

    args.checkpoints = args.save_path

    logger = get_logger(
        args.log_dir, __name__, 'info.log')

    logger.info(args)

    exp = Exp(args, scaler)  # set experiments
    print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
    logger.info(
        '{},{}'.format('Total parameters ', str(sum(p.numel() for p in exp.model.parameters() if p.requires_grad))))
    if args.mode != 'online':
        exp.train(setting, logger)

    print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
    if args.mode != 'train_only':
        exp_time, preds, trues = exp.test(setting, logger)
        torch.cuda.empty_cache()
