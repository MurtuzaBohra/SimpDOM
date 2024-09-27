import os
import re
import sys
import pickle

import pandas as pd
from tqdm.notebook import tqdm

from Utils.logger import logger
from DatasetCreation.helperFunctions import get_site_info
from DatasetCreation.helperFunctions import remove_hidden_dir

#sys.path.append('/Users/bmurtuza/Documents/Research/code/site_agnostic_extraction/src/SiteAgnosticClosedIE')

data_path = '../data'
vertical = 'auto'

def __record_frequences(elements, frequences_map):
    for elem in elements:
        if elem in frequences_map:
            frequences_map[elem] +=1
        else:
            frequences_map[elem] = 1

def __record_char_frequences(text, char_freqs):
    __record_frequences(set(text.lower()), char_freqs)

def __get_xpath_tags(absxpath):
    # Get all the xpath tags, but the first element is ignored as it is an 
    # empty string ''. Note that, tag_with_count can be of a form like: div[*]
    return {re.search(r'[a-z]+', tag_with_count).group(0) for tag_with_count in absxpath.split('/')[1:]}

def __record_tag_frequences(absxpath, tag_freqs):
    __record_frequences(__get_xpath_tags(absxpath), tag_freqs)

def __get_tokens_dictionary(freq_dict, threshold):
    # Define the index incrementing function
    index = 0
    def get_next_id():
        nonlocal index
        index += 1
        return index
    
    # Return the dictionary of the token to unique id mapping
    return {token : get_next_id() for token, frequency in freq_dict.items() if frequency >= threshold}

def __dump_dictionary_file(data_dict, data_path, file_name, desc):
    dump_file_name = os.path.join(data_path, f'{file_name}.pkl')
    logger.info(f'Dumping {desc} of: {len(data_dict)} elements into: {dump_file_name}')
    pickle.dump(data_dict, open(dump_file_name, 'wb'))

def create_char_and_tag_dict(data_path, vertical, threshold=10):
    # Get the list of the vertical's web-site directory names
    website_dirs = remove_hidden_dir(os.listdir(os.path.join(data_path, vertical)))
    logger.info(f'The character and tag dictionary frequency filtering threshold: {threshold}')

    # Default initialise the dictionaries
    char_freqs, tag_freqs = {}, {}
    
    # Iterate over the vertical's web site directories
    for dir_name in tqdm(website_dirs, desc='Web sites'):
        # Get the main site info
        website, _, _, _ = get_site_info(data_path, vertical, dir_name)

        # Read the pre-generated node details file for the website pages
        dump_file_name = os.path.join(data_path, 'node_details', f'{website}.pkl')
        website_pages_node_details = pickle.load(open(dump_file_name, 'rb'))

        # Iterate over the web pages
        for page_nodes in tqdm(website_pages_node_details.values(), desc=f'Pages for website: {website}'):
            # Iterate over web page's node details
            for node in page_nodes.values():
                # Update the char frequences
                __record_char_frequences(node.text, char_freqs)
                # Update the tag frequences
                __record_tag_frequences(node.absxpath, tag_freqs)
    
    # Do the threshold filtering of the characters and tags
    char_dict = __get_tokens_dictionary(char_freqs, threshold)
    tag_dict = __get_tokens_dictionary(tag_freqs, threshold)

    __dump_dictionary_file(char_dict, data_path, 'English_charDict', 'English character dictionary')
    __dump_dictionary_file(tag_dict, data_path, 'HTMLTagDict', 'HTML tags dictionary')

if __name__=="__main__":
    create_char_and_tag_dict(data_path, vertical)
