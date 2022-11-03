import numpy as np
import torch
import pickle
import pandas as pd
import torch.nn.functional as F
from tqdm.notebook import tqdm
from DataLoader.swde_dataLoader import swde_data_test, collate_fn_test

PROB_THRESHOLD = 0.4
device = 'cuda'

#---------------------------------------------------------------------------------
#---------------------------------------------------------------------------------
def to_device(batch,device):
    raw_nodes, xpath_seqs, xpath_lens, leaf_tag_indices, pos_indices, nodes_word_embs, nodes_sent_lens, friends_word_embs, friends_sent_lens, \
    partners_word_embs, partners_sent_lens, nodes_char_seqs, nodes_word_lens, friends_char_seqs, friends_word_lens, \
    partners_char_seqs, partners_word_lens, labels = batch
        
    return raw_nodes, xpath_seqs.to(device), xpath_lens.to(device), leaf_tag_indices.to(device), pos_indices.to(device), nodes_word_embs.to(device), nodes_sent_lens.to(device), friends_word_embs.to(device), friends_sent_lens.to(device), \
    partners_word_embs.to(device), partners_sent_lens.to(device), nodes_char_seqs.to(device), nodes_word_lens.to(device), friends_char_seqs.to(device), friends_word_lens.to(device), \
    partners_char_seqs.to(device), partners_word_lens.to(device), labels

def test_step_SimpDOM(model, batch, batch_idx):
    raw_nodes, xpath_seqs, xpath_lens, leaf_tag_indices, pos_indices, nodes_word_embs, nodes_sent_lens, friends_word_embs, friends_sent_lens, \
    partners_word_embs, partners_sent_lens, nodes_char_seqs, nodes_word_lens, friends_char_seqs, friends_word_lens, \
    partners_char_seqs, partners_word_lens, labels = to_device(batch,device)

    outputs = model(xpath_seqs, xpath_lens, leaf_tag_indices, pos_indices, nodes_word_embs, nodes_sent_lens, friends_word_embs, friends_sent_lens, \
    partners_word_embs, partners_sent_lens, nodes_char_seqs, nodes_word_lens, friends_char_seqs, friends_word_lens, \
    partners_char_seqs, partners_word_lens)

    outputs = F.softmax(outputs, dim=1)
        

    preds = torch.max(outputs, dim=1)[1].cpu().detach().numpy()
    preds_probs = outputs.cpu().detach().numpy()
        
    return labels.detach().numpy(), preds, preds_probs, raw_nodes

#---------------------------------------------------------------------------------
#---------------------------------------------------------------------------------

def create_results_df(labels, preds, preds_probs, raw_nodes, PROB_THRESHOLD):
    df = pd.DataFrame(columns=['website','pageID', 'xpath', 'text', 'friends_text', 'partner_text', 'prediction', 'label', 'isMatch']+['class'+str(i) for i in range(preds_probs.shape[1])])
    preds_after_threshold = []
    for idx in range(len(raw_nodes)):
        pred = preds[idx] if preds_probs[idx,preds[idx]]>PROB_THRESHOLD else 0
        preds_after_threshold.append(pred)
        isMatch = True if pred==labels[idx] else False
        df.loc[len(df)] = [raw_nodes[idx][0], raw_nodes[idx][1], raw_nodes[idx][2], raw_nodes[idx][3],\
                        raw_nodes[idx][4], raw_nodes[idx][5], pred, labels[idx], isMatch]+ list(preds_probs[idx,:])
    return df, preds_after_threshold

def main(test_dataset, model, Device, PROB_THRESHOLD): 
    # PROB_THRESHOLD -> confidence to predict target attribute.
    global device
    device = Device
    df = pd.DataFrame()
    batch_idx = 0
    for batch in tqdm(test_dataset, desc='Testing batches'):
        labels, preds, preds_probs, raw_nodes = test_step_SimpDOM(model, batch, batch_idx)
        df_temp, preds = create_results_df(labels, preds, preds_probs, raw_nodes, PROB_THRESHOLD)
        df = pd.concat([df, df_temp])
        batch_idx += 1
    return df



if __name__ == "__main__":
    main(test_dataset, model, s3_bucket, results_s3path)
