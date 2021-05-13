import pandas as pd
import pickle
import numpy as np
import os
import random
from Utils.DOMTree import DOMTree
from DatasetCreation.helperFunctions import remove_hidden_dir, get_text_nodes

Datapath = '/Users/bmurtuza/Documents/Research/data/swde/SimpDOM'
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
    websites = remove_hidden_dir(os.listdir('{}/{}'.format(Datapath,vertical)))
    print(websites)
    fixedNodes_filename = '{}/fixedNodes_camera.csv'.format(Datapath)
    fixedNodes = pd.read_csv(fixedNodes_filename,  dtype= str, na_values=str, keep_default_na=False)
    
    for dirname in websites:
        website = dirname.split('(')[0]
        print(website)
        num_pages = int(dirname.split('(')[1].strip(')'))
        nodesDetails = {}
        for idx in range(num_pages):
            page_ID = list('0000')
            page_ID[-len(str(idx)):] = str(idx)
            page_ID = ''.join(page_ID)
            html_filename = '{}/{}/{}/{}.htm'.format(Datapath,vertical,dirname,page_ID)
            nodesDetails[page_ID] = get_text_nodes_details(html_filename, fixedNodes.loc[fixedNodes.website == website])
        pickle.dump(nodesDetails, open('{}/nodesDetails/{}.pkl'.format(Datapath, website), 'wb'))
        
        variableAndFixedNodesCounts = [get_count_of_variable_and_fixed_nodes(nodesDetails[page_ID]) for page_ID in nodesDetails.keys()]
        print('avg no. of variable nodes = {}'.format(np.mean(np.array(variableAndFixedNodesCounts), axis=0)))
        print('=====================')

if __name__ == "__main__":
    main(Datapath)