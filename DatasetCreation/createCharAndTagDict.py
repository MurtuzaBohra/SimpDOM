import pickle
import pandas as pd
import re
import sys
sys.path.append('/Users/bmurtuza/Documents/Research/code/site_agnostic_extraction/src/SiteAgnosticClosedIE')

Datapath = '/Users/bmurtuza/Documents/Research/data/wpix3_uec/wpix3/data/SimpDOM'
MIN_DOCUMENT_FREQUENCY = 10

# websites = ['auto-kbb','auto-autoweb','auto-aol','auto-yahoo','auto-motortrend','auto-autobytel','auto-carquotes','auto-cars','auto-msn','auto-automotive']

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

def main(Datapath, websites):
    charFreq = {}
    tagFreq ={}
    for website in websites:
        print(website)
        nodesDetailsAllPages = pickle.load(open('{}/nodesDetails/{}.pkl'.format(Datapath, website), 'rb'))
        for page in nodesDetailsAllPages.values():
            for node in page.values():
                charFreq = addToCharDictionary(node.text, charFreq)
                tagFreq = addToTagDictionary(node.absxpath, tagFreq)
    charDict = getDictAfterThresholding(charFreq)
    tagDict = getDictAfterThresholding(tagFreq)
    return charDict,tagDict

if __name__=="__main__":
    charDict, tagDict = main(Datapath, websites)
    print('size of char vocab - {}'.format(len(charDict)))
    print(charDict)
    print('===========================')
    print('size of tag vocab - {}'.format(len(tagDict)))
    print(tagDict)
    pickle.dump(charDict, open('{}/English_charDict.pkl'.format(Datapath), 'wb'))
    pickle.dump(tagDict, open('{}/HTMLTagDict.pkl'.format(Datapath), 'wb'))

