import os
import pickle
import random

import numpy as np
import pandas as pd

from pathlib import Path
from tqdm.notebook import tqdm

from Utils.logger import logger
from Utils.DOMTree import DOMTree
from DatasetCreation.helperFunctions import get_text_nodes
from DatasetCreation.helperFunctions import remove_hidden_dir

Datapath = '../'
vertical = 'auto'
def get_text_nodes_details(html_filename, fixedNodes):
    with open(html_filename, 'r') as f:
        html_content = f.read()
    root = DOMTree('xxx', str(html_content)).get_page_root()
    nodes_dict = get_text_nodes(root, fixedNodes)
    return nodes_dict

def get_count_of_variable_and_fixed_nodes(nodes_dict):
    count=0
    for node_ID in nodes_dict.keys():
        if nodes_dict[node_ID].isVariableNode ==True:
            count+=1
    return count, len(nodes_dict)-count

def main(Datapath, vertical):
    websites = remove_hidden_dir(os.listdir(os.path.join(Datapath, vertical)))

    fixedNodes_filename = os.path.join(Datapath, 'fixedNodes_camera.csv')
    fixedNodes = pd.read_csv(fixedNodes_filename,  dtype= str, na_values=str, keep_default_na=False)

    nd_path = Path(os.path.join(Datapath, 'nodesDetails'))
    nd_path.mkdir(parents=True, exist_ok=True)

    for dirname in tqdm(websites, desc='Web sites'):
        website = dirname.split('(')[0]
        num_pages = int(dirname.split('(')[1].strip(')'))
        nodesDetails = {}
        for idx in tqdm(range(num_pages), desc=f'Pages for website: {website}'):
            page_ID = list('0000')
            page_ID[-len(str(idx)):] = str(idx)
            page_ID = ''.join(page_ID)
            html_filename = os.path.join(Datapath, vertical, dirname, f'{page_ID}.htm')
            nodesDetails[page_ID] = get_text_nodes_details(html_filename, fixedNodes.loc[fixedNodes.website == website])

        dump_file_name = os.path.join(Datapath, 'nodesDetails',f'{website}.pkl')
        logger.info(f'Dumping node details into: {dump_file_name}')
        pickle.dump(nodesDetails, open(dump_file_name, 'wb'))
        
        variableAndFixedNodesCounts = [get_count_of_variable_and_fixed_nodes(nodesDetails[page_ID]) for page_ID in nodesDetails.keys()]
        logger.info(f'The average variabe/mixed node counts: {website} are: {np.mean(np.array(variableAndFixedNodesCounts), axis=0)}')

if __name__ == "__main__":
    main(Datapath)