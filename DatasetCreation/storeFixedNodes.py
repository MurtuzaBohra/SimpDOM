import pandas as pd
import pickle
import numpy
import os
import random
from Utils.DOMTree import DOMTree
from DatasetCreation.helperFunctions import remove_hidden_dir, get_text_nodes

Datapath = '/Users/bmurtuza/Documents/Research/data/swde/SimpDOM'
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

def main(Datapath, vertical):
    websites = remove_hidden_dir(os.listdir('{}/{}'.format(Datapath, vertical)))
    print(websites)
    fixedNodes = pd.DataFrame(columns= ['website', 'absxpath', 'text'])
    for dirname in websites:
        website = dirname.split('(')[0]
        num_pages = int(dirname.split('(')[1].strip(')'))
        sample_pages_ID = random.sample([i for i in range(num_pages)], int(num_pages*1.0)) #sample 10% pages to get the fixed nodes
        print(website, len(sample_pages_ID))
        xpathTextCount = {}
        for page_ID in sample_pages_ID:
            filename = list('0000')
            filename[-len(str(page_ID)):] = str(page_ID)
            filename = ''.join(filename)
            html_filename = '{}/{}/{}/{}.htm'.format(Datapath, vertical,dirname,filename)
            xpathTextCount = createXpathTextCount(html_filename, xpathTextCount)
        fixedNodes = updateFixedNode(xpathTextCount, len(sample_pages_ID), fixedNodes, website)
        print(len(fixedNodes))
        fixedNodes.to_csv('{}/fixedNodes_camera.csv'.format(Datapath), index=False)

if __name__ == "__main__":
    main(Datapath)