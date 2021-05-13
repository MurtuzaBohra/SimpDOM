from DatasetCreation.namedTuples import DOMNodeDetails

def remove_hidden_dir(websites):
    reduced_list = []
    for website in websites:
        if website[0]!='.':
            reduced_list.append(website)
    return reduced_list

def get_text_nodes(root, fixedNodes={}):
    tree = root.getroottree()
    node_dict = {}
    for nodeID,elem in enumerate(root.iter()):#nodeID is DFS position of node in the DOM tree.
        text=''
        if elem.text !=None:
            text += elem.text
        if elem.tail !=None:
            text+= elem.tail
        if elem.tag not in ['script', 'style'] and text.strip()!='':
            absxpath = tree.getpath(elem)
            isVariableNode = True
            if len(fixedNodes)!=0 and (absxpath in list(fixedNodes.absxpath)) and (text in list(fixedNodes.text)):
                isVariableNode = False
            nodeDetails = DOMNodeDetails(absxpath, text, isVariableNode, [], [], '0')
            node_dict[nodeID] = nodeDetails
    return node_dict