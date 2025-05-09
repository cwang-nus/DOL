import numpy as np
import torch
import random
def init_dl_program(
        device_name,
        seed=None,
        use_cudnn=True,
        deterministic=True,
        benchmark=False,
        use_tf32=False,
        max_threads=None
):
    if max_threads is not None:
        torch.set_num_threads(max_threads)
        if torch.get_num_interop_threads() != max_threads:
            torch.set_num_interop_threads(max_threads)
        try:
            import mkl
        except:
            pass
        else:
            mkl.set_num_threads(max_threads)

    if seed is not None:
        random.seed(seed)
        seed += 1
        np.random.seed(seed)
        seed += 1
        torch.manual_seed(seed)

    if isinstance(device_name, (str, int)):
        device_name = [device_name]

    devices = []
    for t in reversed(device_name):
        t_device = torch.device(t)
        devices.append(t_device)
        if t_device.type == 'cuda':
            assert torch.cuda.is_available()
            torch.cuda.set_device(t_device)
            if seed is not None:
                seed += 1
                torch.cuda.manual_seed(seed)
    devices.reverse()
    torch.backends.cudnn.enabled = use_cudnn
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = benchmark

    if hasattr(torch.backends.cudnn, 'allow_tf32'):
        torch.backends.cudnn.allow_tf32 = use_tf32
        torch.backends.cuda.matmul.allow_tf32 = use_tf32

    return devices if len(devices) > 1 else devices[0]
def adjust_learning_rate(optimizer, epoch, args, decision=False):
    if decision:
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] / 3
        print('Updating learning rate to {}'.format(param_group['lr']))
        return
    if args.lradj=='type1':
        lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch-1) // 1))}
    elif args.lradj=='type2':
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print('Updating learning rate to {}'.format(lr))

class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, path, name='checkpoint.pth'):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path, name)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path, name)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path, name='checkpoint.pth'):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), path+'/'+name)
        self.val_loss_min = val_loss


class StandardScaler():
    """
    Standard the input
    https://github.com/nnzhan/Graph-WaveNet/blob/master/util.py
    """
    def __init__(self):
        self.mean = 0.
        self.std = 1.

    def fit(self, data):
        # print(data.shape)

        # print(data.mean(), data.std())

        self.mean = data.mean()
        self.std = data.std()

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean

        #return (data * self.std) + self.mean


class StandardScaler_old():
    def __init__(self):
        self.mean = 0.
        self.std = 1.

    def fit(self, data):
        self.mean = data.mean(0)
        self.std = data.std(0)

    def transform(self, data):
        mean = torch.from_numpy(self.mean).type_as(data).to(data.device) if torch.is_tensor(data) else self.mean
        std = torch.from_numpy(self.std).type_as(data).to(data.device) if torch.is_tensor(data) else self.std
        return (data - mean) / std

    def inverse_transform(self, data):
        mean = self.mean.reshape(1, 1, -1)
        std = self.std.reshape(1, 1, -1)
        mean = torch.from_numpy(mean).type_as(data).to(data.device) if torch.is_tensor(data) else mean
        std = torch.from_numpy(std).type_as(data).to(data.device) if torch.is_tensor(data) else std
        return (data * std) + mean

    def inverse_transform_res(self, data):
        mean = self.mean.reshape(1, 1, -1, 1, 1)
        std = self.std.reshape(1, 1, -1, 1, 1)
        mean = torch.from_numpy(mean).type_as(data).to(data.device) if torch.is_tensor(data) else mean
        std = torch.from_numpy(std).type_as(data).to(data.device) if torch.is_tensor(data) else std
        return (data * std) + mean