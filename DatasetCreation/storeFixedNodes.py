import os
import numpy
import pickle
import random

import pandas as pd

from tqdm.notebook import tqdm

from Utils.logger import logger
from Utils.DOMTree import DOMTree
from DatasetCreation.helperFunctions import get_text_nodes
from DatasetCreation.helperFunctions import remove_hidden_dir

data_path = '../data'
vertical = 'auto'
FIXED_NODE_THRESHOLD = 0.4

def createXpathTextCount(html_filename, xpathTextCount):
    with open(html_filename, 'r') as f:
        html_content = f.read()
    root = DOMTree('xxx', str(html_content)).get_page_root()
    node_dict = get_text_nodes(root)
    for nodeDetail in node_dict.values():
        try:
            xpathTextCount[(nodeDetail.absxpath, nodeDetail.text)] +=1
        except Exception as e:
            xpathTextCount[(nodeDetail.absxpath, nodeDetail.text)] = 1
    return xpathTextCount

def updateFixedNode(xpathTextCount, num_sample_pages, fixedNodes, website):
    for key in xpathTextCount.keys():
        if xpathTextCount[key] >= int(num_sample_pages*FIXED_NODE_THRESHOLD):
            fixedNodes.loc[len(fixedNodes)] = [website, key[0], key[1]]
    return fixedNodes

def main(data_path, vertical):
    websites = remove_hidden_dir(os.listdir(os.path.join(data_path, vertical)))

    fixedNodes = pd.DataFrame(columns= ['website', 'absxpath', 'text'])
    for dirname in tqdm(websites, desc='Web sites'):
        website = dirname.split('(')[0]
        num_pages = int(dirname.split('(')[1].strip(')'))
        sample_pages_ID = random.sample([i for i in range(num_pages)], int(num_pages*1.0)) #sample 10% pages to get the fixed nodes
        logger.debug(f'Considering: {website}, with: {len(sample_pages_ID)} sample pages')

        xpathTextCount = {}
        for page_ID in tqdm(sample_pages_ID, desc=f'Pages for website: {website}'):
            filename = list('0000')
            filename[-len(str(page_ID)):] = str(page_ID)
            filename = ''.join(filename)
            html_filename = os.path.join(data_path, vertical, dirname, f'{filename}.htm')
            xpathTextCount = createXpathTextCount(html_filename, xpathTextCount)

        # Update the fixed nodes dataframe
        fixedNodes = updateFixedNode(xpathTextCount, len(sample_pages_ID), fixedNodes, website)

    # Store the fixed nodes data into file
    file_path = os.path.join(data_path, 'fixedNodes_camera.csv')
    logger.info(f'Got: {len(fixedNodes)} fixed nodes for: {vertical}, dumping into: {file_path}')
    fixedNodes.to_csv(file_path, index=False)

if __name__ == "__main__":
    main(data_path)