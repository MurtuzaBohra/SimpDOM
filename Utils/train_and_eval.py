import os
import torch
import pickle
import random

import numpy as np
import pandas as pd
import torch.nn as nn
import pytorch_lightning as pl

from torchviz import make_dot
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from Utils.logger import logger
from Model.SimpDOM_model import SeqModel
from Prediction.PRSummary import cal_PR_summary
from Prediction.test_step import main as get_predictions
from Utils.pretrainedGloVe import pretrainedWordEmeddings
from Prediction.WebsiteLevel_PR_Generator import cal_PR_summary as websiteLevel_cal_PR_summary

data_path = './data'
device = 'cpu'

n_workers=0 # Important to keep this at zero as otherwise we get a shared memory error
n_gpus=0
char_emb_dim = 16
char_hid_dim = 100
char_emb_dropout = 0.1

tag_emb_dim = 16
tag_hid_dim = 30

leaf_emb_dim = 30
pos_emb_dim = 20

char_dict_filename = os.path.join(data_path, 'English_charDict.pkl')
tag_dict_filename = os.path.join(data_path, 'HTMLTagDict.pkl')
word_emb_filename = os.path.join(data_path, 'glove.6B.100d.txt')
train_model_weights_file = os.path.join(data_path, 'weights.ckpt')
pred_dump_file_name = 'test_predictions.csv'

def create_config(train_websites, val_websites, test_websites, attributes, learning_rate = 1e-5):
    n_classes = len(attributes) + 1
    class_weights = [1, 100, 100, 100, 100]
    config = {
        'out_dim': n_classes,
        'train_websites': train_websites,
        'val_websites': val_websites,
        'test_websites': test_websites,
        'datapath': data_path,
        'n_workers': n_workers,
        'char_emb_dim' : char_emb_dim,
        'char_hid_dim' : char_hid_dim,
        'char_emb_dropout' : char_emb_dropout,
        'tag_emb_dim': tag_emb_dim,
        'tag_hid_dim': tag_hid_dim,
        'leaf_emb_dim': leaf_emb_dim,
        'pos_emb_dim': pos_emb_dim,
        'attributes': attributes,
        'n_gpus' : n_gpus,
        'class_weights':class_weights,
        'char_dict_filename' : char_dict_filename,
        'tag_dict_filename': tag_dict_filename,
        'word_emb_filename': word_emb_filename,
        'learning_rate': learning_rate,
        'patience' : 2
        
    }
    return config

def train_model(config, num_train_epochs):
    logger.info('Instantiating the Model Checkpoint callback')
    cp_callback = ModelCheckpoint(
        dirpath=os.path.join('.', 'data'),
        filename='weights',
        save_top_k=1,
        save_last = True,
        verbose=True,
        monitor='val_loss',
        mode='min'
    )
    
    logger.info('Instantiating the Early Stopping callback')
    es_callback = EarlyStopping(monitor="val_loss", patience=config['patience'], mode="min")
    
    logger.info('Instantiating the Sequential model')
    model = SeqModel(config)

    logger.info('Instantiating the Training object')
    trainer = pl.Trainer(accelerator='cpu', devices=10, max_epochs=num_train_epochs,
                         strategy = 'ddp_fork_find_unused_parameters_false',
                         callbacks=[cp_callback, es_callback])

    logger.info('Fitting the model')
    trainer.fit(model)

    return model

def visualize_model(model, file_name='model_structure', fmt='png'):
    val_data_loader = model.train_dataloader()
    batch = next(iter(val_data_loader))
    yhat = model(*batch[:-1])
    logger.info(f'Dumping the SimpDOM model visualization into: ./{file_name}.{fmt}')
    make_dot(yhat, params=dict(list(model.named_parameters()))).render(file_name, format=fmt)

def test_model(config, model=None, model_weights_file=None, test_websites=None):
    if model is None:
        logger.warning(f'The pre-trained model is not provided, loading from disk')
        if model_weights_file is None:
            logger.warning(f'The model weights file is not provided, loading weights from: {model_weights_file}')
            model_weights_file = train_model_weights_file
        else:
            logger.info(f'Loading weights from: {model_weights_file}')
        model = SeqModel.load_from_checkpoint(model_weights_file, config=config)
        model = model.eval()
        model = model.to(device)

    # Get the validation sites
    if test_websites is None:
        logger.warning(f'The test websites was not explit, extracting from config!')
        test_websites = config['test_websites']
    else:
        logger.warning(f'The test websites was explit, updating config!')
        config['test_websites'] = test_websites
    logger.info(f'The test websites are: {test_websites}')
    
    # Get the model predictions
    df = get_predictions(model, device, 0.7)
    print(f'Dumping predictions dataframe into: {pred_dump_file_name}')
    df.to_csv(pred_dump_file_name)
    
    # Get the overall summary
    n_classes = config['out_dim']
    avg_p_r_f1_dict = cal_PR_summary(df, n_classes)
    logger.info(f'Prediction summary:\n{avg_p_r_f1_dict}')

    # Get the website level summary
    pr_summary_df, pr_results_df = websiteLevel_cal_PR_summary(df, n_classes)
    logger.info(f'Website-level prediction summary:\n{pr_results_df}')
    
    return avg_p_r_f1_dict


