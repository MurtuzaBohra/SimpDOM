import pandas as pd
import numpy as np
import collections

pr_params = collections.namedtuple('pr_params', 'precision recall')


def isMatch(pred_value, label_values):
    out = False
    for value in label_values:
        if (pred_value in value) or (value in pred_value):
            out=True
    return out

def cal_top_1_prec_recall(df, n_classes):
    pr_params_dict = {}
    predicted_classes = []
    for clas, group in df.groupby(['prediction']):
        if clas!='0' and clas!=0:
            predicted_classes.append(clas)
            row = group.iloc[group['class'+str(clas)].argmax()]
            if row['prediction']==row['label'] or isMatch(row['text'], list(df.loc[df.label==row['prediction']].text)):
                pr_params_dict[clas] = pr_params(1,1)
            else:
                if (clas in list(df.label)):
                    pr_params_dict[clas] = pr_params(0,0)
                else:
                    pr_params_dict[clas] = pr_params(0,None)

    for clas in list(set([clas for clas in range(1,n_classes)])-set(predicted_classes)):
        # print(pr_params_dict.keys())
        if clas not in list(df.label):
            pr_params_dict[clas] = pr_params(1,None)
        else:
            pr_params_dict[clas] = pr_params(None,0)
    return pr_params_dict

def cal_PR_summary(df, n_classes):
    page_level_prec = {clas: [] for clas in range(1,n_classes)}
    page_level_recall = {clas: [] for clas in range(1,n_classes)}
    for (website, pageID), group in df.groupby(['website', 'pageID']):
        pr_params_dict = cal_top_1_prec_recall(group, n_classes)
        for clas in pr_params_dict.keys():
            pr = pr_params_dict[clas]
            if pr.precision!=None:
                page_level_prec[clas].append(pr.precision)
            if pr.recall!=None:
                page_level_recall[clas].append(pr.recall)

    avg_prf1_dict = {clas: (0,0,0) for clas in range(1,n_classes)}
    for clas in range(1,n_classes):
        avg_prec = np.mean(page_level_prec[clas])
        avg_recall= np.mean(page_level_recall[clas])
        avg_f1 = (2*avg_prec*avg_recall)/(avg_prec+avg_recall)
        avg_prf1_dict[clas] = (avg_prec, avg_recall, avg_f1)
        print('class - {}: precision = {}, recall = {}, F1 = {}'.format(clas, avg_prec, avg_recall, avg_f1))
    return avg_prf1_dict

if __name__=="__main__":
    n_classes=4
    results_filename = './UnitTests/resources/results_camera_v1_df.csv'
    df = pd.read_csv(results_filename, dtype=str, na_values=str, keep_default_na=False)
    df[['class'+str(clas) for clas in range(n_classes)]] = df[['class'+str(clas) for clas in range(n_classes)]].apply(pd.to_numeric)
