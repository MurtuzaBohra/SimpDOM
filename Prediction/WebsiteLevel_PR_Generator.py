import pandas as pd
import numpy as np

def isMatch(pred_value, label_values):
    out = False
    for value in label_values:
        if (pred_value in value) or (value in pred_value):
            out=True
    return out

def get_PR_summary_for_top1_prediction(website, pageID, page_df, n_classes):
    predicted_classes = []
    out_df = pd.DataFrame(columns= ['website', 'pageID', 'attribute', 'prediction_xpath','predicted_value', 'reference_value', 'isMatch'])
    for clas, group in page_df.groupby(['prediction']):
        if clas!='0' and clas!=0:
            predicted_classes.append(clas)
            row = group.iloc[group['class'+str(clas)].argmax()]
            pred_xpath, pred_value = row['xpath'], row['text'] 
            ref_values = list(set(page_df.loc[page_df.label==clas].text))
            # if row['prediction']==row['label'] or isMatch(row['text'], list(page_df.loc[page_df.label==row['prediction']].text)):
            is_match = isMatch(pred_value, ref_values)

            if len(ref_values)==0:
                ref_values = ''
            out_df.loc[len(out_df)]= [website, pageID, clas, pred_xpath, pred_value, ref_values, is_match]

    for clas in list(set([clas for clas in range(1,n_classes)])-set(predicted_classes)):
        if clas not in list(page_df.label):
            out_df.loc[len(out_df)]= [website, pageID, clas,'', '', '', True]
        else:
            ref_values = list(set(page_df.loc[page_df.label==clas].text))
            out_df.loc[len(out_df)]= [website, pageID, clas, '','', ref_values, False]
    return out_df

def cal_pr(summary_df):
    relevant = len(summary_df.loc[summary_df.reference_value != ''])
    retrieved = len(summary_df.loc[summary_df.predicted_value != ''])
    retrieved_relevant = len(summary_df.loc[(summary_df.isMatch==True) & (summary_df.reference_value!='')])
    prec = retrieved_relevant/retrieved if retrieved!=0 else 0
    recall = retrieved_relevant/relevant if relevant!=0 else 0
    return prec, recall, retrieved_relevant,retrieved, relevant

def cal_pr_results(pr_summary_df):
    pr_results_df = pd.DataFrame(columns= ['website', 'attribute', 'precision', 'recall', 'retrieved_relevant', 'retrieved', 'relevant'])
    for (website, attribute), summary_df in pr_summary_df.groupby(['website', 'attribute']):
        pr_results_df.loc[len(pr_results_df)] = [website, attribute] + list(cal_pr(summary_df))
    return pr_results_df


def cal_PR_summary(df, n_classes):
    pr_summary_df = pd.DataFrame()
    for (website, pageID), page_df in df.groupby(['website', 'pageID']):
        pr_summary_df = pd.concat([pr_summary_df, get_PR_summary_for_top1_prediction(website, pageID, page_df, n_classes)])
    pr_results_df = cal_pr_results(pr_summary_df)
    return pr_summary_df, pr_results_df

if __name__=="__main__":
    n_classes=5
    results_filename = './UnitTests/resources/results_wpix_df.csv'
    df = pd.read_csv(results_filename, dtype=str, na_values=str, keep_default_na=False)
    df[['class'+str(clas) for clas in range(n_classes)]+ ['prediction', 'label']] = df[['class'+str(clas) for clas in range(n_classes)]+ ['prediction', 'label']].apply(pd.to_numeric)
    
    pr_summary_df, pr_results_df = cal_PR_summary(df,n_classes)
    pr_summary_df.to_csv('./UnitTests/resources/pr_summary_wpix.csv')
    pr_results_df.to_csv('./UnitTests/resources/pr_results_wpix.csv')



