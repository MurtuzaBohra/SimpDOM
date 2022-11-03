import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pickle
import numpy as np
import pandas as pd
import random
from Utils.pretrainedGloVe import pretrainedWordEmeddings
from DataLoader.swde_dataLoader import swde_data_test, collate_fn_test
from Model.SimpDOM_model import SeqModel
from Prediction.test_step import main as get_predictions
from Prediction.PRSummary import cal_PR_summary as pageLevel_cal_PR_summary
from Prediction.WebsiteLevel_PR_Generator import cal_PR_summary as websiteLevel_cal_PR_summary
from Utils.logger import logger

datapath = './data'
random.seed(7)
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
word_emb_filename= '{}/glove.6B.100d.txt'.format(datapath)

def load_dict(fname):
    logger.info(f'Loading {fname}')
    Dict = pickle.load(open(fname,'rb'))
    logger.info(f'Dictionary {fname} length: {len(Dict)}')
    return Dict

def train(websites, attributes):
    train_websites, val_websites = websites[:1], websites[1:]
    print( 'training websites - {}'.format(train_websites))
    print( 'validation websites - {}'.format(val_websites))
    n_classes = len(attributes)+1
    class_weights = [1,100,100,100,100]

    charDict = load_dict(f'{datapath}/English_charDict.pkl')
    tagDict = load_dict(f'{datapath}/HTMLTagDict.pkl')

    logger.info('Instantiating the Model checkpoint.')
    checkpoint_callback = ModelCheckpoint(
        filename='./data/weights',
        save_top_k=1,
        save_last = True,
        verbose=True,
        monitor='val_loss',
        mode='min'
    )
    
    logger.info('Instantiating the Sequential model')
    config = {
        'out_dim': n_classes,
        'train_websites': train_websites,
        'val_websites': val_websites,
        'datapath': datapath,
        'n_workers': n_workers,
        'charDict' : charDict,
        'char_emb_dim' : char_emb_dim,
        'char_hid_dim' : char_hid_dim,
        'char_emb_dropout' : char_emb_dropout,
        'tagDict': tagDict,
        'tag_emb_dim': tag_emb_dim,
        'tag_hid_dim': tag_hid_dim,
        'leaf_emb_dim': leaf_emb_dim,
        'pos_emb_dim': pos_emb_dim,
        'attributes': attributes,
        'n_gpus' : n_gpus,
        'class_weights':class_weights,
        'word_emb_filename': word_emb_filename
    }
    model = SeqModel(config)

    logger.info('Instantiating the Training object')
    trainer = pl.Trainer(gpus=n_gpus, max_epochs=1, callbacks=[checkpoint_callback])

    logger.info('Fitting the model')
    trainer.fit(model)
    
    logger.info('Saving the check point')
    trainer.save_checkpoint("weights_wpix_manual_ckpt.ckpt")

    logger.info('Re-loading the Sequential model from Checkpoint')
    model = SeqModel.load_from_checkpoint("weights_wpix_manual_ckpt.ckpt",config=config)
    model = model.to(device)

    return val_websites, charDict, tagDict, model, n_classes


def test(val_websites, charDict, tagDict, model, n_classes):
    WordEmeddings = pretrainedWordEmeddings(word_emb_filename)
    test_dataset = DataLoader(dataset = swde_data_test(val_websites, datapath, charDict, \
                                      tagDict, n_gpus, WordEmeddings), num_workers=n_workers, \
                                      batch_size=32, shuffle=False, pin_memory = True, collate_fn = collate_fn_test)
    model = model.eval()
    df = get_predictions(test_dataset, model, device, 0.7)
    avg_p_r_f1_dict = pageLevel_cal_PR_summary(df, n_classes)
    # pr_summary_df, pr_results_df = websiteLevel_cal_PR_summary(df, n_classes)
    # print(pr_results_df)
    return avg_p_r_f1_dict


