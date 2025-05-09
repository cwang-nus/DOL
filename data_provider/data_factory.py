from data_provider.data_loader import Dataset_Speed, Dataset_SingapreT, Dataset_ChicagoT, Dataset_Pred
from torch.utils.data import DataLoader

data_dict = {
    'singapore-t': Dataset_SingapreT,
    'metr-la': Dataset_Speed,
    'pems-bay': Dataset_Speed,
    'chicago-t': Dataset_ChicagoT
}

def data_provider(args, flag):
    Data = data_dict[args.data]
    timeenc = int(args.embed) if args.embed != 'timeF' else 3

    if flag == 'test':
        shuffle_flag = False
        drop_last = False
        batch_size = args.test_bsz
    elif flag == 'val':
        shuffle_flag = False
        drop_last = False
        batch_size = args.batch_size
    elif flag == 'pred':
        shuffle_flag = False
        drop_last = False
        batch_size = 1
        Data = Dataset_Pred
    else:
        shuffle_flag = True
        drop_last = True
        batch_size = args.batch_size

    data_set = Data(
        root_path=args.root_path,
        data_path=args.data_path,
        flag=flag,
        size=[args.seq_len, args.pred_len],
        features=args.features,
        target=args.target,
        timeenc=timeenc,
        scaler=args.scaler
    )

    print(flag, len(data_set))
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last)
    return data_set, data_loader