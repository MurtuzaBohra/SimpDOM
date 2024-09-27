import re
import os
import pickle

import pandas as pd

from tqdm.notebook import tqdm

from Utils.logger import logger
from DatasetCreation.helperFunctions import get_site_info
from DatasetCreation.helperFunctions import remove_hidden_dir

data_path = '../data'
vertical = 'auto'

def __read_ground_truth(data_path, vertical, website, attribute):
    # Read the ground truth file for the given website/attribute
    ground_truth_filename = os.path.join(data_path, 'groundtruth', vertical, f'{website}-{attribute}.txt')
    logger.debug(f'Reading ground truth from: {ground_truth_filename}')
    with open(ground_truth_filename, 'r') as f:
        content = f.readlines()
    
    # Clean up and split
    lines = [line.rstrip('\n').split('\t') for line in content[2:]]

    # Create the map of {page_id : list(attribute values)}
    ground_truth = {line[0]: line[2:] for line in lines}
    
    return ground_truth

def __jaccard_similarity(val1, val2):
    # A Quick method to find Jaccard similarity https://en.wikipedia.org/wiki/Jaccard_index
    
    # WARNING: All Non Alpha Numeric chars are ignored during computation of similarity
    val1_set = set(re.sub(r'[^a-z\s\d]', ' ', val1.lower()).split())
    val2_set = set(re.sub(r'[^a-z\s\d]', ' ', val2.lower()).split())
    
    # Number of common tokens
    intersection = len(val1_set.intersection(val2_set))
    # Total number of tokens
    union = (len(val1_set) + len(val2_set)) - intersection
    
    return float(intersection) / union if intersection else 0.0

def __is_jaccard_similar(val1, val2, threshold):
    return __jaccard_similarity(val1, val2) > threshold

def __is_attribute_node(text, gt_values, threshold):
    # Remove redundant white spaces
    text = ' '.join(text.split())
    gt_values  = [re.sub('&nbsp;', ' ', gt_value) for gt_value in gt_values]

    # If at least one is sufficiently similar, then the node matches the attribute
    for gt_value in gt_values:
        if __is_jaccard_similar(text, gt_value, threshold):
            return True
    
    return False

def __annotate_ground_truth(website_pages_node_details, ground_truth, label_index, annotation_statistics, website, attribute, threshold):
    # For each of the website pages
    for page_id, nodes in website_pages_node_details.items():
        annotated_texts = []
        gt_values = ground_truth[page_id]
        
        # Label each individual node if its content is sufficiently 
        # similar to the provided ground truth attribute value
        for node_id, node in nodes.items():
            if node.isVariableNode:
                if __is_attribute_node(node.text, gt_values, threshold):
                    # Remember the annotated text for the statistics
                    annotated_texts.append(node.text.strip())
                    
                    # Change the attribute and store the new node details
                    nodes[node_id] = node._replace(label=label_index)
        
        # Update the annotations dataframe
        annotation_statistics.loc[len(annotation_statistics)] = [website, attribute, page_id, len(annotated_texts), annotated_texts]
    
    return website_pages_node_details, annotation_statistics

def assign_ground_truth_to_each_node(data_path, vertical, attributes, threshold_def=0.9, thresholds={'price':0.6}):
    # Get the list of the vertical's web-site directory names
    website_dirs = remove_hidden_dir(os.listdir(os.path.join(data_path, vertical)))
    logger.info(f'The Jaccard similarity attribute matching thresholds, default: {threshold_def}, specific: {thresholds}')

    # Create attribute to index mapping
    label_indices = {attribute: str(idx + 1) for idx, attribute in enumerate(attributes)}
    
    # Create the annotations statistics data frame
    annotation_statistics = pd.DataFrame(columns = ['website', 'attribute', 'page_id', 'annotation_count', 'annotation_text'])

    # Iterate over the vertical's web site directories
    for dir_name in tqdm(website_dirs, desc='Web sites'):
        # Get the main site info
        website, _, _, _ = get_site_info(data_path, vertical, dir_name)

        # Read the pre-generated node details file for the website pages
        dump_file_name = os.path.join(data_path, 'node_details', f'{website}.pkl')
        website_pages_node_details = pickle.load(open(dump_file_name, 'rb'))

        # For each of the attributes under consideration
        for attribute in tqdm(attributes, desc=f'Attributes for website: {website}'):
            # Get the ground truth information
            ground_truth = __read_ground_truth(data_path, vertical, website, attribute)
            
            # Devise the similarity threshold
            threshold = thresholds.get(attribute, threshold_def) if thresholds else threshold_def
            
            # Annotate the nodes based on the ground truth information
            website_pages_node_details, annotation_statistics = __annotate_ground_truth(website_pages_node_details, ground_truth, label_indices[attribute],
                                                                                        annotation_statistics, website, attribute, threshold)
        
        logger.info(f'Re-dumping node details (all pages) into: {dump_file_name}')
        pickle.dump(website_pages_node_details, open(dump_file_name, 'wb'))
    
    stats_file_path = os.path.join(data_path, 'node_details', 'annotation_statistics.csv')
    logger.info(f'Dumping annotations statistics into: {stats_file_path}')
    annotation_statistics.to_csv(stats_file_path, index=False)

if __name__ == "__main__":
    assign_ground_truth_to_each_node(data_path, vertical, attributes)
