import os
import pickle
import random

import numpy as np
import pandas as pd

from pathlib import Path
from tqdm.notebook import tqdm

from Utils.logger import logger
from Utils.DOMTree import DOMTree
from DatasetCreation.helperFunctions import get_site_info
from DatasetCreation.helperFunctions import get_text_nodes
from DatasetCreation.helperFunctions import remove_hidden_dir
from DatasetCreation.helperFunctions import get_html_file_name

data_path = '../data'
vertical = 'auto'

def __get_text_nodes_details(html_filename, fixed_nodes):
    # Open the webpage file
    with open(html_filename, 'r') as f:
        html_content = f.read()
        
    # Get the root of the DOM tree
    root = DOMTree('xxx', str(html_content)).get_page_root()
    
    # Get all of the text nodes which are non in the list of fixed nodes
    nodes_dict = get_text_nodes(root, fixed_nodes)
    
    return nodes_dict

def __get_count_of_variable_and_fixed_nodes(page_nodes_dict):
    # Count the number of variable nodes for the given page
    count = 0
    for node in page_nodes_dict.values():
        if node.isVariableNode == True:
            count += 1
    
    # Return the number of variable nodes and fixed nodes
    return count, len(page_nodes_dict)-count

def store_all_text_nodes(data_path, vertical):
    # Get the list of the vertical's web-site directory names
    website_dirs = remove_hidden_dir(os.listdir(os.path.join(data_path, vertical)))

    # Load the previously found fixed nodes
    fixed_nodes_file_name = os.path.join(data_path, 'fixed_nodes_camera.csv')
    logger.info(f'Loading the fixed nodes list from: {fixed_nodes_file_name}')
    fixed_nodes = pd.read_csv(fixed_nodes_file_name,  dtype= str, na_values=str, keep_default_na=False)

    # Create the node details folder, if not present yet
    nd_path = Path(os.path.join(data_path, 'node_details'))
    nd_path.mkdir(parents=True, exist_ok=True)

    # Go over the vertical's websites
    for dir_name in tqdm(website_dirs, desc='Web sites'):
        # Get the main site info
        website, num_pages, page_name_templates, file_path = get_site_info(data_path, vertical, dir_name)

        # Get the node details
        nodes_details = {}
        for page_id in tqdm(range(num_pages), desc=f'Pages for website: {website}'):
            html_file_name, page_id_str = get_html_file_name(page_name_templates, file_path, page_id)
            website_fixed_nodes_df = fixed_nodes.loc[fixed_nodes.website == website]
            nodes_details[page_id_str] = __get_text_nodes_details(html_file_name, website_fixed_nodes_df)

        # Dump the variable node counts for the website
        dump_file_name = os.path.join(data_path, 'node_details',f'{website}.pkl')
        logger.info(f'Dumping node details into: {dump_file_name}')
        pickle.dump(nodes_details, open(dump_file_name, 'wb'))
        
        all_node_counts = [__get_count_of_variable_and_fixed_nodes(nodes_details[page_id]) for page_id in nodes_details.keys()]
        logger.info(f'The "{website}" page average variabe/fixed node counts: {np.mean(np.array(all_node_counts), axis=0)}')

if __name__ == "__main__":
    store_all_text_nodes(data_path, vertical)