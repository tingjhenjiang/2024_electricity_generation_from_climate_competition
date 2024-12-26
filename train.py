import pandas as pd
import pathlib
import torch
import math
import os
import tempfile
import ast
from sklearn.model_selection import StratifiedKFold
import json,hashlib
import h5py
import get_data
# from torch.optim import AdamW
# from torch.optim.lr_scheduler import OneCycleLR
# from transformers import EarlyStoppingCallback, Trainer, TrainingArguments, set_seed
# from transformers.integrations import INTEGRATION_TO_CALLBACK


googlecolab_output_dir = pathlib.Path("/content/drive/MyDrive/models")
kaggle_output_dir = pathlib.Path("/kaggle/working")
outputdir = googlecolab_output_dir if googlecolab_output_dir.exists() else (kaggle_output_dir/"models" if kaggle_output_dir.exists() else get_data.current_folder/"outputdir")
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
current_dir = pathlib.Path.cwd()
num_cores = get_data.dask_compute_kwargs['num_workers']

if __name__=="__main__":
    import argparse
    import pypots.imputation
    from pypots.utils.metrics import calc_mae
    parser = argparse.ArgumentParser(
        description="Example script",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size for training")
    parser.add_argument('--epoch', type=int, default=100, help="epoch")
    parser.add_argument('--windowsize', type=int, default=30, help="windowsize")
    parser.add_argument('--save_path', type=str, default=str(outputdir.resolve()), help="save_path")
    parser.add_argument('--modeltype', type=str, default='SAITS', help="modeltype")
    parser.add_argument('--loadmodelfile', type=str, default='', help="loadmodelfile")

    args = parser.parse_args()
    argp_dict = vars(args)
    # print(f"argp_dict: {argp_dict}")
    argp_dict_json_str = json.dumps(argp_dict, sort_keys=True)
    hash_object = hashlib.sha256(argp_dict_json_str.encode())
    hash_string = hash_object.hexdigest()

    outputdir = pathlib.Path(argp_dict['save_path'])/hash_string
    outputdir.mkdir(parents=True, exist_ok=True)
    (outputdir/"hyperparameter.json").write_text(json.dumps(argp_dict), encoding='utf-8')

    custom_train = {
        'X':None
    }
    # Load from HDF5 file
    with h5py.File(current_dir/'traindata.h5', 'r') as f:
        custom_train['X'] = f['timeseries_observations_computed'][:]

    if argp_dict['modeltype']=='SAITS':
        model = pypots.imputation.saits.SAITS(
            n_steps=argp_dict['windowsize'],
            n_features=len(get_data.train_cols),
            n_layers=2,
            d_model=256,
            n_heads=4,
            d_k=64,
            d_v=64,
            d_ffn=128,
            dropout=0.1,
            batch_size=argp_dict['batch_size'],
            epochs=argp_dict['epoch'],
            num_workers=num_cores,
            saving_path=str(outputdir),
            model_saving_strategy="better"
        )
    if argp_dict['modeltype']=='TEFN':
        model = pypots.imputation.tefn.TEFN(
            n_steps=argp_dict['windowsize'],
            n_features=len(get_data.train_cols),
            batch_size=argp_dict['batch_size'],
            epochs=argp_dict['epoch'],
            num_workers=num_cores,
            saving_path=str(outputdir),
            model_saving_strategy="better"
        )
    if argp_dict['modeltype']=='SegRNN':
        model = pypots.imputation.segrnn.SegRNN(
            n_steps=argp_dict['windowsize'],
            n_features=len(get_data.train_cols),
            batch_size=argp_dict['batch_size'],
            epochs=argp_dict['epoch'],
            num_workers=num_cores,
            saving_path=str(outputdir),
            model_saving_strategy="better"
        )
    if argp_dict['modeltype']=='FITS':
        model = pypots.imputation.fits.FITS(
            n_steps=argp_dict['windowsize'],
            n_features=len(get_data.train_cols),
            cut_freq=10,
            batch_size=argp_dict['batch_size'],
            epochs=argp_dict['epoch'],
            num_workers=num_cores,
            saving_path=str(outputdir),
            model_saving_strategy="better"
        )
    if argp_dict['modeltype']=='TimeMixer':
        model = pypots.imputation.timemixer.TimeMixer(
            n_steps=argp_dict['windowsize'],
            n_features=len(get_data.train_cols),
            n_layers=2,
            d_model=256,
            d_ffn=128,
            top_k=10,
            dropout=0.1,
            batch_size=argp_dict['batch_size'],
            epochs=argp_dict['epoch'],
            num_workers=num_cores,
            saving_path=str(outputdir),
            model_saving_strategy="better"
        )
    if argp_dict['modeltype']=='ImputeFormer':
        model = pypots.imputation.imputeformer.ImputeFormer(
            n_steps=argp_dict['windowsize'],
            n_features=len(get_data.train_cols),
            n_layers=2,
            d_input_embed=256,
            d_learnable_embed=256,
            d_proj=256,
            d_ffn=128,
            dropout=0.1,
            n_temporal_heads=4,
            batch_size=argp_dict['batch_size'],
            epochs=argp_dict['epoch'],
            num_workers=num_cores,
            saving_path=str(outputdir),
            model_saving_strategy="better"
        )
    if argp_dict['modeltype']=='iTransformer':
        model = pypots.imputation.itransformer.iTransformer(
            n_steps=argp_dict['windowsize'],
            n_features=len(get_data.train_cols),
            n_layers=2,
            d_model=256,
            n_heads=4,
            d_k=64,
            d_v=64,
            d_ffn=128,
            dropout=0.1,
            batch_size=argp_dict['batch_size'],
            epochs=argp_dict['epoch'],
            num_workers=num_cores,
            saving_path=str(outputdir),
            model_saving_strategy="better"
        )
    if argp_dict['modeltype']=='Koopa':
        model = pypots.imputation.koopa.Koopa(
            n_steps=argp_dict['windowsize'],
            n_features=len(get_data.train_cols),
            n_seg_steps=len(get_data.train_cols),
            d_dynamic=256,
            d_hidden=256,
            n_hidden_layers=2,
            n_blocks=10,
            batch_size=argp_dict['batch_size'],
            epochs=argp_dict['epoch'],
            num_workers=num_cores,
            saving_path=str(outputdir),
            model_saving_strategy="better"
        )
    if argp_dict['modeltype']=='FreTS':
        model = pypots.imputation.frets.FreTS(
            n_steps=argp_dict['windowsize'],
            n_features=len(get_data.train_cols),
            batch_size=argp_dict['batch_size'],
            epochs=argp_dict['epoch'],
            num_workers=num_cores,
            saving_path=str(outputdir),
            model_saving_strategy="better"
        )
    if argp_dict['modeltype']=='Crossformer':
        model = pypots.imputation.crossformer.Crossformer(
            n_steps=argp_dict['windowsize'],
            n_features=len(get_data.train_cols),
            n_layers=2,
            d_model=256,
            n_heads=4,
            d_ffn=128,
            factor=5,
            seg_len=8,
            win_size=3,
            batch_size=argp_dict['batch_size'],
            epochs=argp_dict['epoch'],
            num_workers=num_cores,
            saving_path=str(outputdir),
            model_saving_strategy="better"
        )
    if argp_dict['loadmodelfile']!='':
        model.load(argp_dict['loadmodelfile'])
    model.fit(custom_train)
    # elif argp_dict['modeltype']=='timemixer':
# pypots.imputation.timemixer.TimeMixer(n_steps, n_features, n_layers, d_model, d_ffn, top_k, dropout=0, channel_independence=False, decomp_method='moving_avg', moving_avg=5, downsampling_layers=3, downsampling_window=2, apply_nonstationary_norm=False, batch_size=32, epochs=100, patience=None, optimizer=<pypots.optim.adam.Adam object>, num_workers=0, device=None, saving_path=None, model_saving_strategy='best', verbose=True)