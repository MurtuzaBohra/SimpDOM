import os
import pickle

import numpy as np
import pandas as pd

from tqdm.notebook import tqdm
from collections import defaultdict

from Utils.logger import logger
from DatasetCreation.helperFunctions import get_site_info
from DatasetCreation.helperFunctions import remove_hidden_dir

Datapath = '../'
vertical = 'auto'

def __get_node_ancestors(node, max_num_ancestors):
    # Take the abstract xpath representing the node and split it into elements
    xpath_list = node.absxpath.split('/')
    logger.debug(f'Node: "{node.absxpath}", list: {xpath_list}')

    # Get the total number of ancestors.
    num_ancestors = len(xpath_list)

    # Limit the number of ancestor nodes by the specified maximum to consider.
    # The xpath starting with /, so the true number of acestors is 1 less.
    sel_num_ancestors = min(max_num_ancestors, num_ancestors - 1)
    
    # Get the list of at most num_ancestors ancestor nodes
    ancestor_node_xpaths = ['/'.join(xpath_list[:num_ancestors - k]) for k in range(0, sel_num_ancestors)]
    logger.debug(f'Node: "{node.absxpath}", {sel_num_ancestors}/{num_ancestors} ancestor node paths:\n{ancestor_node_xpaths}')

    return ancestor_node_xpaths

def __create_ancestor_xpath_to_node_ids_map(node_details, max_num_ancestors):
    # This dictionary will store a mapping from the xpath to the node ids 
    # of the nodes for which this xpath-defined node is the ancestor
    anc_xpath_to_node_ids_map = defaultdict(set)
    
    # Iterate over the web page node ids which are DFS node positions in the DOM tree
    for node_id, node in node_details.items():
        # Get the current node ancestors
        ancestor_node_xpaths = __get_node_ancestors(node, max_num_ancestors)
        
        # For each ancestor node xpath add the child node id
        for anc_node_xpath in ancestor_node_xpaths:
            anc_xpath_to_node_ids_map[anc_node_xpath].add(node_id)
    
    return anc_xpath_to_node_ids_map

def __remove_given_node_from_desc(desc_node_ids, node_id):
    node_ids = desc_node_ids.copy()
    node_ids.remove(node_id)
    return node_ids

def __get_avg_size_of_friend_circle(node_details):
    count, count_friends, count_partners = 0, 0, 0
    
    for node in node_details.values():
        if node.isVariableNode:
            count += 1
            count_friends += len(node.friendNodes)
            count_partners += len(node.partnerNodes)
    
    return float(count_friends)/count, float(count_partners)/count

def __sort_closest_descendents(node_details, descendents, node_absxpath):
    # Get the descendent node xpath lengths
    descendents_absxpath_lens = [len(node_details[node_id].absxpath.split()) for node_id in descendents]
    
    # Get the xpath length of the node under consideration
    node_absxpath_len = len(node_absxpath.split())
    
    # Compute the list of xpath length differences
    # WARNING: How representative is this? The xpath lengths can even be the same but the horisontal distance in HTML can be very big!
    descendents_absxpath_lens = [abs(descendent_absxpath_len - node_absxpath_len) for descendent_absxpath_len in descendents_absxpath_lens]
    
    # Sort by the differences and the descendent node ids
    # WARNING: The order of DFS node ids impacts the preference for the same length difference!
    descendents = [x for _, x in sorted(zip(descendents_absxpath_lens, descendents))]
    
    return descendents

def __get_partner_friend_node_rep(node_details, node_id):
    # The partner/prient node representation is limited to its abstract xpath and the node text
    # TODO: Extend the representation by the node attributes
    return (node_details[node_id].absxpath, node_details[node_id].text)

def __update_Df_and_Dp_in_node_details(node_details, anc_xpath_to_node_ids_map, max_num_ancestors, max_num_friends):
    '''
    INPUT
        node_details: for a single web-page is a dictionary of node_id -> DOMNodeDetails (namedTuple)
        anc_xpath_to_node_ids_map: for a web page, is dictionary of node xpath to it's descendent ids
        max_num_ancestors: the maximum number of ancestors to be considered for friend circle
        max_num_friends: the maximum number of close friends to consider per node
    OUTPUT:
        node_details: updated friendsNodes and partnerNodes for each node.
    '''
    # Iterate over all the variable nodes
    for node_id, node in node_details.items():
        # Only do this for variable nodes
        if node.isVariableNode:
            partner_node_id, friend_node_ids = None, set()
            
            # Extract the max_num_ancestors node ancestors
            ancestor_node_xpaths = __get_node_ancestors(node, max_num_ancestors)
            
            # For each of the node ancestors
            for anc in ancestor_node_xpaths:
                # Get the descendent node ids, excluding the node under consideration itself
                desc_set = __remove_given_node_from_desc(anc_xpath_to_node_ids_map[anc], node_id)
                
                # If there is just one descendent and there are no other partner and friends yet, assign it as a partner
                if (len(desc_set) == 1) and (partner_node_id is None) and (len(friend_node_ids) == 0):
                    partner_node_id = next(iter(desc_set))
                
                # Extend the set of friend with the new descendents
                new_friend_node_ids = friend_node_ids.union(desc_set)
                
                # Check we do not exceed the max_num_friends
                if len(new_friend_node_ids) > max_num_friends:
                    # Get the new candidate friend nodes
                    cand_friend_node_ids = list(desc_set - friend_node_ids)
                    
                    # Sort the list by how close they are to the node under consideration
                    ordered_desc_list = __sort_closest_descendents(node_details, cand_friend_node_ids, node.absxpath)
                    
                    # Make sure we only top up to max_num_friends closest friends
                    friend_node_ids.update(set(ordered_desc_list[:max_num_friends - len(friend_node_ids)]))
                    
                    # Finish iterations as we are done
                    break
                else:
                    # Use the new set of friend ids as there numbers are not exceeding max_num_friends
                    friend_node_ids = new_friend_node_ids
            
            # Update the friend nodes
            friend_nodes = [__get_partner_friend_node_rep(node_details, friend_node_id) for friend_node_id in friend_node_ids]
            node = node._replace(friendNodes = friend_nodes)

            # Update the partner nodes. Note that there is currently at most one partner node allowed!
            partner_nodes = [] if partner_node_id is None else [__get_partner_friend_node_rep(node_details, partner_node_id)]
            node = node._replace(partnerNodes = partner_nodes)
            
            # Set back the updated node, we need to do that as the _replace method returns a new instance!
            node_details[node_id] = node
    
    # Compute the average size of friend and partner sets for the give webpage
    avg_no_friends, avg_no_of_partners = __get_avg_size_of_friend_circle(node_details)
    
    return node_details, avg_no_friends, avg_no_of_partners

def friend_circle_extractioin(data_path, vertical, max_num_ancestors=5, max_num_friends=10):
    # Get the list of the vertical's web-site directory names
    website_dirs = remove_hidden_dir(os.listdir(os.path.join(data_path, vertical)))

    # Go over the vertical's websites
    for dir_name in tqdm(website_dirs, desc='Web sites'):
        # Get the main site info
        website, _, _, _ = get_site_info(data_path, vertical, dir_name)

        # Read the pre-generated node details file for the website pages
        dump_file_name = os.path.join(data_path, 'node_details', f'{website}.pkl')
        website_pages_node_details = pickle.load(open(dump_file_name, 'rb'))

        # Iterate over the websie page's node details
        avg_friends, avg_partners = [], []
        for page_id in tqdm(website_pages_node_details.keys(), desc=f'Pages for website: {website}'):
            # Get the node details for the given website page
            node_details = website_pages_node_details[page_id]
            
            # Create a map for the given web page where we map each 
            # ancestor xpath to a list of its considered child nodes
            anc_xpath_to_node_ids_map = __create_ancestor_xpath_to_node_ids_map(node_details, max_num_ancestors)
            
            # Devise node partners and firends
            website_pages_node_details[page_id], avg_no_friends, avg_no_of_partners = \
                        __update_Df_and_Dp_in_node_details(node_details, anc_xpath_to_node_ids_map, max_num_ancestors, max_num_friends)
            
            # Collect the average number of friends and parents statistics
            avg_friends.append(avg_no_friends)
            avg_partners.append(avg_no_of_partners)

        logger.info(f'Re-dumping node details (all pages) into: {dump_file_name}')
        pickle.dump(website_pages_node_details, open(dump_file_name, 'wb'))
        
        logger.info(f'The friend/partner nodes for: "{website}" is: {np.mean(avg_friends)}/{np.mean(avg_partners)}')

if __name__ =="__main__":
    friend_circle_extractioin(data_path, vertical)