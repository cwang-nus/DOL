import os
import time
import warnings

import numpy as np
import torch
from torch import optim
from tqdm import tqdm

from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from models.dol import DOL
from utils.buffer import Buffer
from utils.criterion import select_criterion
from utils.metrics import metric, cumavg
from utils.tools import EarlyStopping, adjust_learning_rate
from utils.graph_algo import load_supports

warnings.filterwarnings('ignore')

class Exp(Exp_Basic):
    def __init__(self, args, scaler):
        self.args = args

        self.device = self._acquire_device()
        self.args.device = self.device
        self.scaler = scaler
        self.args.scaler = scaler
        self.n_inner = args.n_inner
        self.opt_str = args.opt

        self.model = DOL(args).to(self.device)
        self.support = self._calculate_supports(args, args.node_num)

        self.one_week_interval = args.one_week_interval
        self.awake_interval = args.awake_week_num * self.one_week_interval
        self.hib_interval = args.hib_week_num * self.one_week_interval
        self.one_ah_cycle = self.awake_interval + self.hib_interval
        self.update_type = args.update_type
        self.awake = True
        self.count = 0
        self.buffer = Buffer(self.args.buffer_size, self.device)

        self.x_recall = torch.zeros(self.args.pred_len, self.args.seq_len, self.args.node_num).to(self.device)
        self.y_recall = torch.zeros(self.args.pred_len, self.args.pred_len, self.args.node_num).to(self.device)

        self.loss = self._select_criterion(args.loss_type)
        self.opt = self._select_optimizer()

        self.train_data, self.train_loader = self._get_data(flag='train')
        self.vali_data, self.vali_loader = self._get_data(flag='val')

    def _calculate_supports(self, args, node_nums):
        adj_filename = os.path.join(args.root_path, args.adj_filename)
        supports = load_supports(adj_filename, node_nums)
        supports = [torch.tensor(i).to(self.device) for i in supports]
        return supports
    def _select_optimizer(self):
        if self.args.opt == 'adamw':
            self.opt = optim.AdamW(self.model.parameters(), lr=self.args.learning_rate)
        else:
            self.opt = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return self.opt

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_criterion(self, loss_type):
        return select_criterion(loss_type)

    def train(self, setting, logger):
        self.setting = setting

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(self.train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(self.train_loader):
                iter_count += 1

                self.opt.zero_grad()
                pred, true = self._process_one_batch(batch_x, batch_y)

                pred = self.scaler.inverse_transform(pred)
                true = self.scaler.inverse_transform(true)

                loss = self.loss(pred, true)
                train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    log = "\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item())
                    logger.info(log)
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    log = '\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time)
                    logger.info(log)
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(self.opt)
                    scaler.update()
                else:
                    loss.backward()
                    self.opt.step()

            log = "Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time)
            logger.info(log)
            train_loss = np.average(train_loss)
            vali_loss = self.vali(self.vali_loader)

            log = "Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss)
            logger.info(log)

            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                logger.info("Early stopping")
                break

            adjust_learning_rate(self.opt, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))
        self.args.checkpoint_path = best_model_path

        return self.model

    def vali(self, vali_loader):
        self.model.eval()
        total_loss = []
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
            pred, true = self._process_one_batch(batch_x, batch_y, mode='vali')
            loss = self.loss(pred.detach().cpu(), true.detach().cpu())
            total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def test(self, setting, logger):

        test_data, test_loader = self._get_data(flag='test')

        if self.args.mode == 'online':
            best_model_path = self.args.checkpoint_path
            self.model.load_state_dict(torch.load(best_model_path, map_location=self.args.device))
            self.setting = setting
            self.logger = logger
            self.vali(self.vali_loader)
        else:
            self.logger = logger

        self.model.eval()

        if self.update_type == 'none':
            for p in self.model.parameters():
                p.requires_grad = False
        elif self.update_type == 'adapter':
            for p in self.model.parameters():
                p.requires_grad = False
            for p in self.model.vi_adapter.parameters():
                p.requires_grad = True
            # for p in self.model.vs_adapter.parameters():
            #     p.requires_grad = True
        elif self.update_type == 'decoder':
            for p in self.model.parameters():
                p.requires_grad = False
            for p in self.model.vi_adapter.parameters():
                p.requires_grad = True
            # for p in self.model.vs_adapter.parameters():
            #     p.requires_grad = True
            for p in self.model.decoder.parameters():
                p.requires_grad = True

        preds, trues = [], []
        maes, mses, rmses, mapes, wmapes = [], [], [], [], []
        start = time.time()

        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(tqdm(test_loader)):
            pred, true = self._process_one_batch(batch_x, batch_y, mode='test')
            pred = pred.detach().cpu()
            true = true.detach().cpu()

            pred = self.scaler.inverse_transform(pred)
            pred[pred < 0] = 0
            true = self.scaler.inverse_transform(true)
            true[true < 0] = 0

            preds.append(pred)
            trues.append(true)

        exp_time = time.time() - start
        log = "counts: {} test time: {}".format(self.count, exp_time)
        logger.info(log)

        preds = torch.cat(preds, dim=0).numpy()
        trues = torch.cat(trues, dim=0).numpy()

        mae, mse, rmse, mape, wmape = metric(preds, trues)
        log = '{}, mae:{}, mse:{}, rmse:{}, mape: {}, wmape1:{}, {}, {}'.format('dynamic_test', mae, mse, rmse,
                                                                                mape, wmape,
                                                                                preds.shape,
                                                                                trues.shape)
        logger.info(log)
        torch.cuda.empty_cache()

        return exp_time, preds, trues

    def _process_one_batch(self, batch_x, batch_y, mode='train'):

        if mode == 'test' and self.update_type != 'none':
            return self._ol_one_batch(batch_x, batch_y)

        x = batch_x.float().to(self.device)
        batch_y = batch_y.float().to(self.device)
        if self.args.use_amp:
            with torch.cuda.amp.autocast():
                outputs = self.model(x, self.support)
        else:
            outputs = self.model(x, self.support)
        f_dim = -1 if self.args.features=='MS' else 0
        batch_y = batch_y[:,-self.args.pred_len:,f_dim:].to(self.device)

        if mode == 'vali':
            self.buffer.add_data(examples=x, labels=batch_y)
        return outputs, batch_y

    def _ol_one_batch(self, batch_x, batch_y):

        x = batch_x.float().to(self.device)
        batch_y = batch_y.float().to(self.device)

        if self.count >= self.args.pred_len:
            self.buffer.add_data(examples=self.x_recall[:1], labels=self.y_recall[:1])
            self.x_recall = torch.cat([self.x_recall[1:], x], dim=0)
            self.y_recall = torch.cat([self.y_recall[1:], batch_y], dim=0)
        else:
            self.x_recall[self.count] = x
            self.y_recall[self.count] = batch_y

        if self.awake:
            if self.args.mem_size > 0:
                if not self.buffer.is_empty():
                    buff_x, buff_y = self.buffer.get_data(self.args.mem_size)
                    out = self.model(buff_x, self.support)
                    buff_y = self.scaler.inverse_transform(buff_y)
                    out = self.scaler.inverse_transform(out)
                    buff_loss = self.loss(out, buff_y)
                    buff_loss.backward()
                    self.opt.step()
                    self.opt.zero_grad()

        for _ in range(self.n_inner):
            if self.args.use_amp:
                with torch.cuda.amp.autocast():
                    outputs = self.model(x, self.support)
            else:
                outputs = self.model(x, self.support)

            if self.hib_interval != 0:
                if self.count % self.awake_interval == 0 and self.awake and self.count != 0:
                    self.awake = False
                    self.buffer.empty()
                elif self.count % self.one_ah_cycle == 0 and not self.awake:
                    self.awake = True

        f_dim = -1 if self.args.features == 'MS' else 0
        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
        self.count += batch_y.size(0)
        return outputs, batch_y

