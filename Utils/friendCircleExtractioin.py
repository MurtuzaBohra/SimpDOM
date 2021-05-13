import pickle
import pandas as pd
import numpy as np
from collections import defaultdict
import os
from DatasetCreation.helperFunctions import remove_hidden_dir

Datapath = '/Users/bmurtuza/Documents/Research/data/swde/SimpDOM'
vertical = 'auto'

def _get_K_ancestors(node, K):
    xpath_list = node.absxpath.split('/')
    K = min(K,len(xpath_list)-1)
    return ['/'.join(xpath_list[:len(xpath_list)-k]) for k in range(0,K)]

def create_Dd(nodesDetails, K):
    Dd = defaultdict(list)
    for node_ID in nodesDetails.keys():
        ANC = _get_K_ancestors(nodesDetails[node_ID], K)
        for anc in ANC:
            Dd[anc].append(node_ID)
    return Dd

def _remove_given_node_from_desc(DESC, node_ID):
    return list(set(DESC) - set([node_ID]))

def _get_avg_size_of_friend_circle(nodesDetails):
    count, count_friends, count_partners = 0,0,0
    for node in nodesDetails.values():
        if node.isVariableNode:
            count +=1
            count_friends += len(node.friendNodes)
            count_partners +=len(node.partnerNodes)
    return float(count_friends)/count, float(count_partners)/count

def sort_closest_DESC(nodesDetails, DESC, node_absxpath):
    DESC_nodes_absxpath_lens = [len(nodesDetails[n_id].absxpath.split()) for n_id in DESC]
    node_absxpath_len = len(node_absxpath.split())
    DESC_nodes_absxpath_lens = [abs(elem-node_absxpath_len) for elem in DESC_nodes_absxpath_lens]
    DESC = [x for _, x in sorted(zip(DESC_nodes_absxpath_lens, DESC))]
    return DESC

def update_Df_and_Dp_in_nodeDetails(nodesDetails, Dd, K):
    '''
	INPUT
		nodesDetails: dictionary of node_ID -> DOMNodeDetails (namedTuple)
		Dd: dictionary of node xpath to it's descendents.
		K: number of ancestors to be considered for friend circle.
	OUTPUT:
		nodesDetails: updated friendsNodes and partnerNodes for each node.
	'''
    for node_ID in nodesDetails.keys():
        Dp, Df = [],[]
        node = nodesDetails[node_ID]
        if node.isVariableNode:
            ANC = _get_K_ancestors(node, K)
            for anc in ANC:
                DESC = _remove_given_node_from_desc(Dd[anc], node_ID)
                if len(DESC)==1 and len(Dp)==0 and len(Df)==0:
                    Dp+=DESC
                temp = list(set(Df).union(set(DESC)))
                if len(temp)>=10:
                    DESC = list(set(DESC) - set(Df))
                    DESC = sort_closest_DESC(nodesDetails, DESC, node.absxpath)
                    Df += DESC[:10-len(Df)]
                    break
                else:
                    Df = list(set(Df).union(set(DESC)))
            node = node._replace(friendNodes = [(nodesDetails[n_ID].absxpath, nodesDetails[n_ID].text) for n_ID in Df])
            node = node._replace(partnerNodes = [(nodesDetails[n_ID].absxpath, nodesDetails[n_ID].text) for n_ID in Dp])
            nodesDetails[node_ID] = node
    avg_no_friends, avg_no_of_partners = _get_avg_size_of_friend_circle(nodesDetails)
    return nodesDetails, avg_no_friends, avg_no_of_partners

def main(Datapath, vertical, K=5):
    websites = remove_hidden_dir(os.listdir('{}/{}'.format(Datapath,vertical)))
    
    for dirname in websites:
        website = dirname.split('(')[0]
        print(website)
        avg_friends, avg_partners = [],[]
        nodesDetailsAllPages = pickle.load(open('{}/nodesDetails/{}.pkl'.format(Datapath, website), 'rb'))
        for page_ID in nodesDetailsAllPages.keys():
            nodesDetails = nodesDetailsAllPages[page_ID]
            Dd = create_Dd(nodesDetails, K)
            nodesDetailsAllPages[page_ID], avg_no_friends, avg_no_of_partners = update_Df_and_Dp_in_nodeDetails(nodesDetails, Dd, K)
            avg_friends.append(avg_no_friends)
            avg_partners.append(avg_no_of_partners)
        pickle.dump(nodesDetailsAllPages, open('{}/nodesDetails/{}.pkl'.format(Datapath, website), 'wb'))
        print('average friend nodes - {}, partner nodes - {}'.format(np.mean(avg_friends), np.mean(avg_partners)))
        print('===========================')

if __name__ =="__main__":
    main(Datapath, vertical, K)


