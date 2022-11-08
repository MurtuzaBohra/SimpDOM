import re
import os
import pickle

import pandas as pd

from tqdm.notebook import tqdm

from Utils.logger import logger
from DatasetCreation.helperFunctions import remove_hidden_dir

data_path = '../data'
vertical = 'auto'

SIMILARITY_THRESHOLD = 0.9

def _read_groundTruth(data_path, vertical, website, attribute):
    groundTruth_filename = '{}/groundTruth/{}/{}-{}.txt'.format(data_path, vertical, website, attribute)
    with open(groundTruth_filename, 'r') as f:
        content = f.readlines()
    lines = [line.rstrip('\n').split('\t') for line in content[2:]]
    groundTruth = {line[0]: line[2:] for line in lines} # {page_ID : list(attribute values)}
    return groundTruth

def jaccard_similarity(val1, val2):
    # A Quick method to find jaccard similarity
    # All Non Alpha Numeric chars are ignored during computation of similarity
    val1_tokens = re.sub(r'[^a-z\s\d]', ' ', val1.lower()).split()
    val2_tokens = re.sub(r'[^a-z\s\d]', ' ', val2.lower()).split()

    intersection = len(list(set(val1_tokens).intersection(set(val2_tokens))))
    union = (len(set(val1_tokens)) + len(set(val2_tokens))) - intersection
    return float(intersection) / (union + 0.00001)

def _isAttributeNode(text, gt_values, attribute):
    text = ' '.join(text.split())#remove redundant white spaces.
    gt_values  = [re.sub('&nbsp;', ' ', gt_value) for gt_value in gt_values]
    bool_output = False
    for gt_value in gt_values:
        if (jaccard_similarity(text, gt_value)> SIMILARITY_THRESHOLD) or (attribute=='price' and jaccard_similarity(text, gt_value)>0.6):
            bool_output = True
    return bool_output

def _annotate_gt(nodesDetailsAllPages, groundTruth, label_index, annotation_statistics, website, attribute):
    for page_ID in nodesDetailsAllPages.keys():
        annotated_texts = []
        nodes = nodesDetailsAllPages[page_ID]
        gt_values = groundTruth[page_ID]
        for node_ID in nodes.keys():
            node = nodes[node_ID]
            if node.isVariableNode and _isAttributeNode(node.text, gt_values, attribute):
                annotated_texts.append(node.text.strip())
                node = node._replace(label=label_index)
                nodes[node_ID] = node
        nodesDetailsAllPages[page_ID] = nodes
        annotation_statistics.loc[len(annotation_statistics)] = [website, attribute, page_ID, len(annotated_texts), annotated_texts]
    return nodesDetailsAllPages, annotation_statistics


def main(data_path, vertical, attributes):
    websites = remove_hidden_dir(os.listdir(os.path.join(data_path, vertical)))
    websites = [dirname.split('(')[0] for dirname in websites]

    label_indices = {attribute: str(idx+1) for idx,attribute in enumerate(attributes)}
    annotation_statistics = pd.DataFrame(columns = ['website', 'attribute', 'page_ID', 'annotation_count', 'annotation_text'])
    for website in tqdm(websites, desc='Web sites'):
        dump_file_name = os.path.join(data_path, 'nodesDetails',f'{website}.pkl')
        nodesDetailsAllPages = pickle.load(open(dump_file_name, 'rb'))
        for attribute in tqdm(attributes, desc=f'Attributes for website: {website}'):
            groundTruth = _read_groundTruth(data_path, vertical, website, attribute)
            nodesDetailsAllPages, annotation_statistics = _annotate_gt(nodesDetailsAllPages, groundTruth, label_indices[attribute],
                                                                       annotation_statistics, website, attribute)
        logger.info(f'Re-dumping node details (all pages) into: {dump_file_name}')
        pickle.dump(nodesDetailsAllPages, open(dump_file_name, 'wb'))
    
    #annotation_statistics.to_csv('{}/nodesDetails/annotation_statistics.csv'.format(data_path), index=False)

if __name__ == "__main__":
    main(data_path, vertical, attributes)
