# Distribution-Aware Online Learning for Urban Spatiotemporal Forecasting on Streaming Data

*This is the implement of DOL using Python 3.11.5, Pytorch 2.1.1, tested on Ubuntu 18.04 with a GeForce RTX 3090 GPU.*

## Requirements

DOL uses the following dependencies:

* Python 3.11.5 and its dependencies
* Numpy and Scipy
* CUDA 11.4 or latest version. And **cuDNN** is highly recommended


## Dataset
The datasets for Chicago (Chicago-T), Los Angeles (METR-LA) and the Bay Area (PEMS-BAY) are available at [Google Drive](https://drive.google.com/drive/folders/1fpHzT_jyoHhr2uQA10-v3JTcoHCVc8IU?usp=drive_link).

Please put the datasets in folder `./datasets`

## Model Training
Main input arguments:
- data: which dataset to use
- batch_size: training batch size
- learning_rate: learning rate
- train_epochs: the max number of training epochs
- mode: train or test
- seq_len: the length of observed segment
- pred_len: the length of target segment
- root_path: the root path of the data file
- save_path: the path to save the output
- adj_filename: the adj file path
- checkpoint_path: pretrain model path
- update_type: update type during online phase
- awake_week_num: number of awake week per AH cycle
- hib_week_num: number of hibernate week per AH cycle
- one_week_interval: the total time intervals per week
- buffer_size: the slots in streaming memory buffer
- mem_size: the slots in episodic memory
- lsa_dim: the dimension in LSA
- lsa_num: the number of LSA layers

## Model Warm Up Phase

First, please export the PYTHONPATH the model directory.

Then, the following examples are conducted on dataset Chicago-T:

* Example 1 (DOL with default settings under warm up phase):

```
python main.py --method dol --data chicago-t --seq_len 12 --pred_len 12 --root_path ./datasets --adj_filename ./datasets/chicago-t/adj_chicago.npy --test_bsz 1 --update_type none --save_path results/
```

* Example 2 (DOL using arbitrary settings under warm up phase):
```
python main.py --method dol --data chicago-t --seq_len 12 --pred_len 12 --root_path ./datasets --adj_filename ./datasets/chicago-t/adj.npy --test_bsz 1 --update_type none --lsa_dim 8 --save_path results/
```

The trained model will be saved to the save_path.

## Model Online Phase
The checkpoints obtained from the warm up phase are available at [Google Drive](https://drive.google.com/drive/folders/18X-GpgzhHk3M0V36EHXmZGPkSg0vZPLJ?usp=drive_link)

Please put the datasets in folder `./checkpoints`

The following examples are conducted on dataset Chicago-T:

* Example 3 (DOL with default settings under online phase):
```
python main.py --method dol --data chicago-t --seq_len 12 --pred_len 12 --root_path ./datasets --adj_filename ./datasets/chicago-t/adj_chicago.npy --test_bsz 1 --update_type adapter --awake_week_num 1 --hib_week_num 1 --one_week_interval 672 --checkpoint_path ./checkpoints/chicago-t/checkpoint.pth --save_path results/ --mode online
```

* Example 4 (DOL with arbitrary settings under online phase):
```
python main.py --method dol --data chicago-t --seq_len 12 --pred_len 12 --root_path ./datasets --adj_filename ./datasets/chicago-t/adj_chicago.npy --test_bsz 1 --update_type adapter --awake_week_num 1 --hib_week_num 1 --one_week_interval 672 --checkpoint_path ./checkpoints/chicago-t/checkpoint.pth --save_path results/ --buffer_size 500 --mem_size 4 --mode online
```
