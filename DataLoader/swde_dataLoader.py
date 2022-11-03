import pickle
import numpy as np
import re
import torch
import random
from tqdm.notebook import tqdm
from torch.utils.data.dataset import Dataset
from DataLoader.utils import sort_and_pad, padded_tensor
from DatasetCreation.namedTuples import DataLoaderNodeDetail
from Utils.logger import logger

charDict ={}
tagDict ={}
N_GPUS=1
MAX_SENT_LEN = 15
MAX_POS =99 # range of relative position
random.seed(7)
#----------------------------------------------------------------------
#----------------------------------------------------------------------

def _get_char_seq(text):
    text = text.lower()
    char_seq = []
    if len(text)==0:
        return torch.tensor([len(charDict)+1])
    for char in str(text):
        try:
            char_index = charDict[char]
        except:
            char_index = len(charDict)+1
        char_seq.append(char_index)
    return torch.tensor(char_seq)

def _get_xpath_seq(absxpath):
    xpath_seq = []
    for tagWithCount in absxpath.split('/')[1:]: # tagWithCount div[*]
        tag = re.search(r'[a-z]+',tagWithCount).group(0)# e.g. div[1] -> div
        try:
            tag_index = tagDict[tag]
            xpath_seq.append(tag_index)
        except:
            tag_index = len(tagDict)+1 #for unknown tag
            xpath_seq.append(tag_index)
    return torch.tensor(xpath_seq)

UNK_WORD = 'unk'
UNK_WORDS = [UNK_WORD for _ in range(MAX_SENT_LEN)]
UNK_WORD_CHAR_SEQS = [_get_char_seq(UNK_WORD) for _ in range(MAX_SENT_LEN)]

def _get_Text_representation(text):
    # Split into words limiting the sentence length
    words = text.split()[:MAX_SENT_LEN]
    
    # Limit the number of words
    num_words = len(words)
    
    # Get the number of unk words to pad with
    unk_words_to_add = (MAX_SENT_LEN-num_words)
    
    # Pad with 'unk' words
    words_List = words + UNK_WORDS[:unk_words_to_add]
    
    # Get the word character sequences
    list_char_seqs = [_get_char_seq(word) for word in words] + UNK_WORD_CHAR_SEQS[:unk_words_to_add]

    return list_char_seqs, words_List, max(num_words, 1)

def loadDataset( websites, isValidataion, datapath='/tmp'):
    nodes = {}
    sample_idx = 0
    for website in tqdm(websites, desc='Web sites'):
        key = f'{datapath}/nodesDetails/{website}.pkl'
        data = pickle.load(open(key,'rb'))
        pageIDs = list(data.keys())
        random.shuffle(pageIDs)
        if isValidataion:
            pageIDs = pageIDs[:int(len(pageIDs)*0.1)]
        for pageID in tqdm(pageIDs, desc='Web pages'):
            nodeIDs = list(data[pageID].keys())
            max_nodeID = max(nodeIDs)
            for nodeID in nodeIDs:
                nodeDetail = data[pageID][nodeID]
                if nodeDetail.isVariableNode:
                    node_char_seqs, node_words, node_sent_len = _get_Text_representation(nodeDetail.text)
                    xpath_seq = _get_xpath_seq(nodeDetail.absxpath)
                    leaf_tag_index = xpath_seq[-1]
                    pos_index = torch.tensor(int((float(nodeID)/max_nodeID)*MAX_POS))

                    friendsText = ' '.join([fnode_text for fnode_xpath, fnode_text in nodeDetail.friendNodes])
                    friend_char_seqs, friend_words, friend_sent_len = _get_Text_representation(friendsText)

                    partnerText = ' '.join([pnode_text for pnode_xpath, pnode_text in nodeDetail.partnerNodes])
                    partner_char_seqs, partner_words, partner_sent_len = _get_Text_representation(partnerText)

                    label = int(nodeDetail.label)

                    nodes[sample_idx] = (DataLoaderNodeDetail(pageID, xpath_seq, leaf_tag_index, pos_index, node_char_seqs, node_words, node_sent_len, friend_char_seqs,\
                                        friend_words, friend_sent_len, partner_char_seqs, partner_words, partner_sent_len), label)
                    sample_idx+=1
    return nodes

#----------------------------------------------------------------------
#----------------------------------------------------------------------

def collate_char_seq (list_char_seqs):
    #len(list_char_seqs)=batch_size
    all_char_seqs = []
    for char_seqs in list_char_seqs:
        all_char_seqs.extend(char_seqs)
    all_char_seqs, all_word_lens = padded_tensor(all_char_seqs)
    return all_char_seqs, all_word_lens


def collate_fn(data):
    nodes, labels = zip(*data)
    batch_size = len(labels)

    labels = torch.tensor(labels)
    #len(labels) = batch_size
    leaf_tag_indices = torch.stack([node.leaf_tag_index for node in nodes],0)
    pos_indices = torch.stack([node.pos_index for node in nodes],0)

    nodes_word_embs, nodes_sent_lens = torch.stack([node.node_words for node in nodes],0),  torch.tensor([node.node_sent_len for node in nodes])# (batch,max_sent_len, emb_dim), sent_lens
    friends_word_embs, friends_sent_lens = torch.stack([node.friend_words for node in nodes],0), torch.tensor([node.friend_sent_len for node in nodes])
    partners_word_embs,partners_sent_lens = torch.stack([node.partner_words for node in nodes],0), torch.tensor([node.partner_sent_len for node in nodes])


    xpath_seqs, xpath_lens  = padded_tensor([node.xpath_seq for node in nodes])
    nodes_char_seqs, nodes_word_lens = collate_char_seq([node.node_char_seqs for node in nodes])
    friends_char_seqs, friends_word_lens = collate_char_seq([node.friend_char_seqs for node in nodes])
    partners_char_seqs, partners_word_lens = collate_char_seq([node.partner_char_seqs for node in nodes])

    return xpath_seqs, xpath_lens, leaf_tag_indices, pos_indices, nodes_word_embs, nodes_sent_lens, friends_word_embs, friends_sent_lens, \
    partners_word_embs, partners_sent_lens, nodes_char_seqs, nodes_word_lens, friends_char_seqs, friends_word_lens, \
    partners_char_seqs, partners_word_lens, labels

def get_attrs_encoding(attributes, WordEmeddings):
    attrs_char_seqs = []
    attrs_word_embs_list = []
    attrs_sent_lens =[]
    for attribute in attributes:
        attr_char_seqs, attr_words_list, attr_sent_len = _get_Text_representation(attribute)
        attrs_sent_lens.append(attr_sent_len)
        attrs_char_seqs.append(attr_char_seqs)
        word_embs = torch.stack([torch.tensor(WordEmeddings.get_embedding(word)) for word in attr_words_list], 0)
        attrs_word_embs_list.append(word_embs)
    attrs_word_embs, attrs_sent_lens = torch.stack(attrs_word_embs_list,0), torch.tensor(attrs_sent_lens)
    attrs_char_seqs, attrs_word_lens = collate_char_seq(attrs_char_seqs)
    return attrs_word_embs, attrs_sent_lens, attrs_char_seqs, attrs_word_lens

class swde_data(Dataset):
    def __init__(self, websites, datapath, cDict, tDict, n_gpus, WordEmeddings, isValidataion):
        #attributes must be in order of labels in the training data.
        global charDict, tagDict, N_GPUS
        N_GPUS = n_gpus
        self.websites = websites
        charDict = cDict
        tagDict = tDict
        self.WordEmeddings = WordEmeddings
        self.nodes = loadDataset(self.websites, isValidataion, datapath)
        # self.nodes = self.loadDataset(datasetS3Bucket)
        
        self.len = len(self.nodes)
        logger.info(f'SWDE data loader has loaded: {self.len} nodes')

    def __getitem__(self, index):
        #self.nodes[index] = (nodeText, friendsText, partnerText, label)
        node, label = self.nodes[index] # namedTuple DataLoaderNodeDetail
        node = node._replace(node_words = torch.stack([torch.tensor(self.WordEmeddings.get_embedding(word)) for word in node.node_words], 0))
        node = node._replace(friend_words = torch.stack([torch.tensor(self.WordEmeddings.get_embedding(word)) for word in node.friend_words], 0))
        node = node._replace(partner_words = torch.stack([torch.tensor(self.WordEmeddings.get_embedding(word)) for word in node.partner_words], 0))
        label = torch.tensor(label)
        return node, label

    def __len__(self):
        return self.len

#-----------------------------------------------------------------------------
#--------------------TestDataLoader------------------------------------

def loadDataset_test( websites, datapath='/tmp'):
    nodes = {}
    raw_nodes = {}
    sample_idx = 0
    for website in tqdm(websites, desc='Web site'):
        key = f'{datapath}/nodesDetails/{website}.pkl'
        data = pickle.load(open(key,'rb'))
        pageIDs = list(data.keys())
        for pageID in tqdm(pageIDs, desc='Web pages'):
            nodeIDs = list(data[pageID].keys())
            max_nodeID = max(nodeIDs)
            for nodeID in nodeIDs:
                nodeDetail = data[pageID][nodeID]
                if nodeDetail.isVariableNode:
                    node_char_seqs, node_words, node_sent_len = _get_Text_representation(nodeDetail.text)
                    xpath_seq = _get_xpath_seq(nodeDetail.absxpath)
                    leaf_tag_index = xpath_seq[-1]
                    pos_index = torch.tensor(int((float(nodeID)/max_nodeID)*MAX_POS))

                    friendsText = ' '.join([fnode_text for fnode_xpath, fnode_text in nodeDetail.friendNodes])
                    friend_char_seqs, friend_words, friend_sent_len = _get_Text_representation(friendsText)

                    partnerText = ' '.join([pnode_text for pnode_xpath, pnode_text in nodeDetail.partnerNodes])
                    partner_char_seqs, partner_words, partner_sent_len = _get_Text_representation(partnerText)

                    label = int(nodeDetail.label)

                    nodes[sample_idx] = (DataLoaderNodeDetail(pageID, xpath_seq, leaf_tag_index, pos_index, node_char_seqs, node_words, node_sent_len, friend_char_seqs,\
                                        friend_words, friend_sent_len, partner_char_seqs, partner_words, partner_sent_len), label)
                    raw_nodes[sample_idx] = (website, pageID,nodeDetail.absxpath, nodeDetail.text, friend_words, partner_words, label)
                    sample_idx+=1
    return nodes, raw_nodes


def collate_fn_test(data):
    raw_nodes, nodes, labels = zip(*data)
    batch_size = len(labels)

    labels = torch.tensor(labels)
    #len(labels) = batch_size
    leaf_tag_indices = torch.stack([node.leaf_tag_index for node in nodes],0)
    pos_indices = torch.stack([node.pos_index for node in nodes],0)
    
    nodes_word_embs, nodes_sent_lens = torch.stack([node.node_words for node in nodes],0),  torch.tensor([node.node_sent_len for node in nodes])# (batch,max_sent_len, emb_dim), sent_lens
    friends_word_embs, friends_sent_lens = torch.stack([node.friend_words for node in nodes],0), torch.tensor([node.friend_sent_len for node in nodes])
    partners_word_embs,partners_sent_lens = torch.stack([node.partner_words for node in nodes],0), torch.tensor([node.partner_sent_len for node in nodes])


    xpath_seqs, xpath_lens  = padded_tensor([node.xpath_seq for node in nodes])
    nodes_char_seqs, nodes_word_lens = collate_char_seq([node.node_char_seqs for node in nodes])
    friends_char_seqs, friends_word_lens = collate_char_seq([node.friend_char_seqs for node in nodes])
    partners_char_seqs, partners_word_lens = collate_char_seq([node.partner_char_seqs for node in nodes])

    #attrs_encoding = [attrs_word_embs, attrs_char_seqs, attrs_word_lens]
    return raw_nodes, xpath_seqs, xpath_lens, leaf_tag_indices, pos_indices, nodes_word_embs, nodes_sent_lens, friends_word_embs, friends_sent_lens, \
    partners_word_embs, partners_sent_lens, nodes_char_seqs, nodes_word_lens, friends_char_seqs, friends_word_lens, \
    partners_char_seqs, partners_word_lens, labels


class swde_data_test(Dataset):
    def __init__(self, websites, datapath, cDict, tDict, n_gpus, WordEmeddings):
        #attributes must be in order of labels in the training data.
        global charDict, tagDict, N_GPUS
        N_GPUS = n_gpus
        self.websites = websites
        charDict = cDict
        tagDict = tDict
        self.WordEmeddings = WordEmeddings
        logger.info(f'Start loading data set for websites: {self.websites}')
        self.nodes, self.raw_nodes = loadDataset_test(self.websites, datapath)
        # self.nodes = self.loadDataset(datasetS3Bucket)
        
        self.len = len(self.nodes)
        print(' {} - nodes are loaded in swde_dataLoader_test'.format(self.len))

    def __getitem__(self, index):
        #self.nodes[index] = (nodeText, friendsText, partnerText, label)
        node, label = self.nodes[index] # namedTuple DataLoaderNodeDetail
        node = node._replace(node_words = torch.stack([torch.tensor(self.WordEmeddings.get_embedding(word)) for word in node.node_words], 0))
        node = node._replace(friend_words = torch.stack([torch.tensor(self.WordEmeddings.get_embedding(word)) for word in node.friend_words], 0))
        node = node._replace(partner_words = torch.stack([torch.tensor(self.WordEmeddings.get_embedding(word)) for word in node.partner_words], 0))
        label = torch.tensor(label)
        return self.raw_nodes[index], node, label

    def __len__(self):
        return self.len

#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
