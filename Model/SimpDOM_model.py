import torch
import pickle
import torchvision

import torch.nn as nn
import pytorch_lightning as pl

from torch.nn import functional as F
from torch.utils.data import DataLoader

from Utils.logger import logger
from Model.char_cnn import CharCNN
from Model.BiLSTM import BiLSTM, BiLSTM_xpath
from DataLoader.swde_dataLoader import swde_data
from DataLoader.swde_dataLoader import collate_fn
from DataLoader.swde_dataLoader import swde_data_test
from DataLoader.swde_dataLoader import collate_fn_test
from DataLoader.swde_dataLoader import get_attrs_encoding
from Utils.pretrainedGloVe import pretrainedWordEmeddings

device = 'cpu'

class SeqModel(pl.LightningModule):

    def __init__(self, config):
        super().__init__()
        
        logger.info(f'Instantiating the new model with config: {config}')
        
        self.out_dim = config['out_dim']
        self.train_websites = config['train_websites']
        self.val_websites = config['val_websites']
        self.test_websites = config['test_websites']
        self.datapath = config['datapath']
        self.n_workers = config['n_workers']
        self.learning_rate = config['learning_rate']

        self.char_dict = self.load_dict(config['char_dict_filename'])
        self.char_emb_dim = config['char_emb_dim']#16
        self.char_hid_dim = config['char_hid_dim']#100
        self.char_emb_dropout = config['char_emb_dropout']

        self.tag_dict = self.load_dict(config['tag_dict_filename'])
        self.tag_emb_dim = config['tag_emb_dim'] # 16
        self.tag_hid_dim = config['tag_hid_dim'] #30
        self.leaf_emb_dim = config['leaf_emb_dim'] #30

        self.pos_emb_dim = config['pos_emb_dim'] #20
        self.word_emb_filename = config['word_emb_filename']
        
        self.attrs = config['attributes']
        self.n_gpus = config['n_gpus']
        self.class_weights = config['class_weights'] # list ->[1,80,80,80,80]
        self.word_emb_dim = 100
        self.dw = 100
        self.n_direction = 2
        self.batch_size = 32

        self.word_meddings = pretrainedWordEmeddings(self.word_emb_filename)
        self.process_attrs()

        self.charLevelWordEmbeddings = CharCNN(n_chars=len(self.char_dict)+2, channels=self.char_hid_dim,
                                               embedding_size=self.char_emb_dim, dropout=self.char_emb_dropout)
        
        self.BiLSTM = BiLSTM(self.char_hid_dim + self.word_emb_dim, self.dw)
        
        self.BiLSTM_xpath = BiLSTM_xpath(len(self.tag_dict)+2, self.tag_emb_dim, self.tag_hid_dim)

        self.Leaf_embedding = nn.Embedding(len(self.tag_dict)+2, self.leaf_emb_dim, padding_idx=0)
        self.pos_embedding = nn.Embedding(100, self.pos_emb_dim, padding_idx=0)
        
        node_emb_dim = self.dw*self.n_direction*3 + self.tag_hid_dim*self.n_direction + self.leaf_emb_dim + self.pos_emb_dim + len(self.attrs)
        self.classifier = nn.Sequential(nn.Linear(node_emb_dim, 64),
                                        nn.ReLU(inplace=True),
                                        nn.Dropout(p=0.3),
                                        nn.Linear(64, self.out_dim))
        self.class_weights = torch.tensor(self.class_weights, dtype=torch.float)#.to(device)
        self.loss = nn.CrossEntropyLoss(weight = self.class_weights)

    def load_dict(self, fname):
        logger.info(f'Loading {fname}')
        Dict = pickle.load(open(fname,'rb'))
        logger.info(f'Dictionary {fname} length: {len(Dict)}')
        return Dict
        
    def text_encoder(self, textRep):
        char_seqs, word_lens, word_embs, sent_lens = textRep
        #char_seqs (batch*sent_lens, max_word_len)
        char_embs = self.charLevelWordEmbeddings(char_seqs, word_lens)
        char_embs = char_embs.view(word_embs.shape[0], word_embs.shape[1],-1)

        textEmbedding = torch.cat((char_embs, word_embs), dim=-1)
        #textEmbedding -> (batch, #max_sent_len, char_hid_dim + word_emb)
        textFeatures = self.BiLSTM(textEmbedding, sent_lens)
        #textFeatures = (batch, BiLSTM_hid_dim*2)
        return textFeatures

    def semantic_similarity(self, e_a, e_p):
        e_p = torch.unsqueeze(e_p, dim=0)
        return F.cosine_similarity(e_p,e_a)

    def forward(self, xpath_seqs, xpath_lens, leaf_tag_indices, pos_indices, \
                nodes_word_embs, nodes_sent_lens, friends_word_embs, friends_sent_lens, partners_word_embs, partners_sent_lens, \
                nodes_char_seqs, nodes_word_lens, friends_char_seqs, friends_word_lens, partners_char_seqs, partners_word_lens):

        d_sem_feat = self.dw*self.n_direction # d_sem_feat is dimension of semantic features.
        e_s = torch.zeros(leaf_tag_indices.shape[0],d_sem_feat*3).to(device)

        textRep_list = [(nodes_char_seqs, nodes_word_lens, nodes_word_embs, nodes_sent_lens), \
                        (friends_char_seqs, friends_word_lens, friends_word_embs, partners_sent_lens),\
                        (partners_char_seqs, partners_word_lens, partners_word_embs, partners_sent_lens)]
        
        for idx in range(len(textRep_list)): # for loop for e_x, e_f and e_p.
            e_s[:,idx*d_sem_feat: (idx+1)*d_sem_feat] = self.text_encoder(textRep_list[idx])
        
        e_a = self.text_encoder(( self.attrs_char_seqs, self.attrs_word_lens, self.attrs_word_embs, self.attrs_sent_lens))

        e_p = e_s[:,2* d_sem_feat: 3* d_sem_feat]
        e_cos = torch.stack([self.semantic_similarity(e_a, e_p[i,:]) for i in range(e_p.shape[0])],0)

        e_xpath = self.BiLSTM_xpath(xpath_seqs, xpath_lens) # e_xpath.shape = tag_hid_dim*2
        e_leaf = torch.squeeze(self.Leaf_embedding(torch.unsqueeze(leaf_tag_indices, dim=0)), dim=0)
        # e_leaf.shape = leaf_emb_dim
        
        e_pos = torch.squeeze(self.pos_embedding(torch.unsqueeze(pos_indices, dim=0)), dim=0)

        e_n = torch.cat((e_s, e_xpath, e_leaf, e_pos, e_cos),dim=-1)
        output = self.classifier(e_n)
        return output

    
    def training_step(self, batch, batch_idx):
        xpath_seqs, xpath_lens, leaf_tag_indices, pos_indices, nodes_word_embs, nodes_sent_lens, friends_word_embs, friends_sent_lens, \
        partners_word_embs, partners_sent_lens, nodes_char_seqs, nodes_word_lens, friends_char_seqs, friends_word_lens, \
        partners_char_seqs, partners_word_lens, labels = batch

        outputs = self(xpath_seqs, xpath_lens, leaf_tag_indices, pos_indices, nodes_word_embs, nodes_sent_lens, friends_word_embs, friends_sent_lens, \
        partners_word_embs, partners_sent_lens, nodes_char_seqs, nodes_word_lens, friends_char_seqs, friends_word_lens, \
        partners_char_seqs, partners_word_lens)

        #loss = F.cross_entropy(outputs, labels)
        loss = self.loss(outputs, labels)
        self.log('train_loss', loss, sync_dist=True, batch_size=self.batch_size)
        return loss
    
    def validation_step(self, batch, batch_idx):
        xpath_seqs, xpath_lens, leaf_tag_indices, pos_indices, nodes_word_embs, nodes_sent_lens, friends_word_embs, friends_sent_lens, \
        partners_word_embs, partners_sent_lens, nodes_char_seqs, nodes_word_lens, friends_char_seqs, friends_word_lens, \
        partners_char_seqs, partners_word_lens, labels = batch

        outputs = self(xpath_seqs, xpath_lens, leaf_tag_indices, pos_indices, nodes_word_embs, nodes_sent_lens, friends_word_embs, friends_sent_lens, \
        partners_word_embs, partners_sent_lens, nodes_char_seqs, nodes_word_lens, friends_char_seqs, friends_word_lens, \
        partners_char_seqs, partners_word_lens)

        #loss = F.cross_entropy(outputs, labels)
        loss = self.loss(outputs, labels)
        self.log('val_loss', loss, sync_dist=True, batch_size=self.batch_size)
        return loss
    
    def validation_epoch_end(self, outputs):
        '''Called after every epoch, stacks validation loss
        '''
        val_loss_mean = torch.stack([x for x in outputs]).mean()
        logger.info(f'Mean validation loss: {val_loss_mean}')
        self.log('avg_val_loss', val_loss_mean, sync_dist=True, batch_size=self.batch_size)      
        
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer
    
    def train_dataloader(self):
        logger.info(f'Creating training data loader for: {self.train_websites}')
        isValidation = False
        return DataLoader(dataset = swde_data(self.train_websites, self.datapath, self.char_dict, self.tag_dict, self.n_gpus, self.word_meddings, isValidation), \
                          num_workers=self.n_workers, batch_size=self.batch_size, shuffle=True, pin_memory = True, collate_fn = collate_fn)
    
    def val_dataloader(self):
        logger.info(f'Creating validation data loader for: {self.val_websites}')
        isValidation = True
        return DataLoader(dataset = swde_data(self.val_websites, self.datapath, self.char_dict, self.tag_dict, self.n_gpus, self.word_meddings, isValidation), \
                          num_workers=self.n_workers, batch_size=self.batch_size, shuffle=False, pin_memory = True, collate_fn = collate_fn)

    def test_dataloader(self):
        logger.info(f'Creating testing data loader for: {self.test_websites}')
        return DataLoader(dataset = swde_data_test(self.test_websites, self.datapath, self.char_dict, self.tag_dict, self.n_gpus, self.word_meddings), \
                          num_workers=self.n_workers, batch_size=self.batch_size, shuffle=False, pin_memory = True, collate_fn = collate_fn_test)
    
    def process_attrs(self):
        attrs_word_embs, attrs_sent_lens, attrs_char_seqs, attrs_word_lens = get_attrs_encoding(self.attrs, self.word_meddings)
        
        self.register_buffer('attrs_word_embs', attrs_word_embs)
        self.register_buffer('attrs_sent_lens', attrs_sent_lens)
        self.register_buffer('attrs_char_seqs', attrs_char_seqs)
        self.register_buffer('attrs_word_lens', attrs_word_lens)
        
