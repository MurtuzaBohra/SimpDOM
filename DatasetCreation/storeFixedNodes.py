import os
import numpy
import pickle
import random

import pandas as pd

from tqdm.notebook import tqdm

from Utils.logger import logger
from Utils.DOMTree import DOMTree
from DatasetCreation.helperFunctions import get_site_info
from DatasetCreation.helperFunctions import get_text_nodes
from DatasetCreation.helperFunctions import remove_hidden_dir
from DatasetCreation.helperFunctions import get_html_file_name

data_path = '../data'
vertical = 'auto'

def __count_xpath_text(html_filename, xpath_text_counts):
    # Read the HTML document
    with open(html_filename, 'r') as f:
        html_content = f.read()
    
    # Parse the HTML document
    root = DOMTree('xxx', str(html_content)).get_page_root()
    
    # Extract the text nodes
    node_dict = get_text_nodes(root)
    
    # Iterate over the text nodes and update the node counts
    for nodeDetail in node_dict.values():
        key = (nodeDetail.absxpath, nodeDetail.text)
        if key in xpath_text_counts:
            xpath_text_counts[key] += 1
        else:
            xpath_text_counts[key] = 1
    
    return xpath_text_counts

def __update_fixed_node(xpath_text_counts, counts_threshold, fixed_nodes_df, website):
    # Iterate ovet the (xpath, text) tuples
    for xpath_text, count in xpath_text_counts.items():
        # Remember as constant text node if the count is larget than the threshold
        if count >= counts_threshold:
            fixed_nodes_df.loc[len(fixed_nodes_df)] = [website, xpath_text[0], xpath_text[1]]
    
    return fixed_nodes_df

def store_fixed_nodes(data_path, vertical, sample_size = 1.0, node_threshold = 0.4):
    # Get the list of the vertical's web-site directory names
    website_dirs = remove_hidden_dir(os.listdir(os.path.join(data_path, vertical)))
    logger.info(f'The fixed nodes threshold is: {node_threshold}, the sub-sample size is: {sample_size}')

    # Prepare the dataframe to fill in with the abstract xpath and text of the fixed nodes
    fixed_nodes_df = pd.DataFrame(columns= ['website', 'absxpath', 'text'])
    for dir_name in tqdm(website_dirs, desc='Web sites'):
        # Get the main site info
        website, num_pages, page_name_templates, file_path = get_site_info(data_path, vertical, dir_name)
        
        # Sub-sample pages to use for fixed nodes extraction
        num_sample_pages = int(num_pages * sample_size)
        sample_page_ids = random.sample(range(num_pages), num_sample_pages)
        logger.info(f'Considering: {website}, taking: {num_sample_pages}/{num_pages} sample pages')

        # Count all of the text node's (xpath, text) tuples
        xpath_text_counts = {}
        for page_id in tqdm(sample_page_ids, desc=f'Pages for website: {website}'):
            html_file_name, _ = get_html_file_name(page_name_templates, file_path, page_id)
            xpath_text_counts = __count_xpath_text(html_file_name, xpath_text_counts)

        # Update the fixed nodes dataframe
        counts_threshold = int(num_sample_pages * node_threshold)
        logger.info(f'The "{website}" web site threshold count is: {counts_threshold}/{num_sample_pages}')
        fixed_nodes_df = __update_fixed_node(xpath_text_counts, counts_threshold, fixed_nodes_df, website)

    # Dump the overall statistics
    stats_df = fixed_nodes_df.groupby('website').count()
    logger.info(f'The website fixed node counts are:\n{stats_df}')
        
    # Store the fixed nodes data into file
    file_path = os.path.join(data_path, 'fixed_nodes_camera.csv')
    logger.info(f'Got: {len(fixed_nodes_df)} fixed nodes for: {vertical}, dumping into: {file_path}')
    fixed_nodes_df.to_csv(file_path, index=False)

if __name__ == "__main__":
    store_fixed_nodes(data_path, vertical)