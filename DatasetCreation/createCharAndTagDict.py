import os
import re
import sys
import pickle

import pandas as pd
from tqdm.notebook import tqdm

from Utils.logger import logger
from DatasetCreation.helperFunctions import remove_hidden_dir

#sys.path.append('/Users/bmurtuza/Documents/Research/code/site_agnostic_extraction/src/SiteAgnosticClosedIE')

data_path = '../data'
MIN_DOCUMENT_FREQUENCY = 10
vertical = 'auto'

def addToCharDictionary(text, charFreq):
    uniqueChars = set(text.lower())
    for char in uniqueChars:
        try:
            charFreq[char] +=1
        except:
            charFreq[char] = 1
    return charFreq

def getTagsFromAbsxpath(absxpath):
    tags = []
    for tagWithCount in absxpath.split('/')[1:]: # tagWithCount div[*]
        tag = re.search(r'[a-z]+',tagWithCount).group(0)
        tags.append(tag)
    return set(tags)

def addToTagDictionary(absxpath, tagFreq):
    uniqueTags = getTagsFromAbsxpath(absxpath)
    for tag in uniqueTags:
        try:
            tagFreq[tag] +=1
        except:
            tagFreq[tag] = 1
    return tagFreq

def getDictAfterThresholding(freqDict):
    Dict = {}
    Index = 1
    for token in freqDict.keys():
        if freqDict[token]>=MIN_DOCUMENT_FREQUENCY:
            Dict[token]=Index
            Index+=1
    return Dict

def dumpFileDict(data_dict, data_path, file_name, desc):
    dump_file_name = os.path.join(data_path, f'{file_name}.pkl')
    logger.info(f'Dumping {desc} of: {len(data_dict)} elements into: {dump_file_name}')
    pickle.dump(data_dict, open(dump_file_name, 'wb'))

def main(data_path, vertical):
    websites = remove_hidden_dir(os.listdir(os.path.join(data_path, vertical)))
    websites = [dirname.split('(')[0] for dirname in websites]

    charFreq = {}
    tagFreq ={}
    for website in tqdm(websites, desc='Web sites'):
        node_info_file_name = os.path.join(data_path, 'nodesDetails',f'{website}.pkl')
        nodesDetailsAllPages = pickle.load(open(node_info_file_name, 'rb'))
        for page in tqdm(nodesDetailsAllPages.values(), desc=f'Pages for website: {website}'):
            for node in page.values():
                charFreq = addToCharDictionary(node.text, charFreq)
                tagFreq = addToTagDictionary(node.absxpath, tagFreq)
    charDict = getDictAfterThresholding(charFreq)
    tagDict = getDictAfterThresholding(tagFreq)

    dumpFileDict(charDict, data_path, 'English_charDict', 'English character dictionary')
    dumpFileDict(tagDict, data_path, 'HTMLTagDict', 'HTML tags dictionary')

if __name__=="__main__":
    main(data_path, vertical)
