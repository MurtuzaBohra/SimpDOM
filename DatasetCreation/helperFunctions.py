import os

from DatasetCreation.namedTuples import DOMNodeDetails

def __get_page_name_templates(num_pages):
    # Create the templates for the page file name depending on how many pages there are
    num_pages_len = len(str(num_pages))
    templates = ['0'*(num_pages_len - page_id_len) + '{}' for page_id_len in range(num_pages_len + 1)]
    
    # The pate id length can not be zero, so set this one to blank
    templates[0] = ''

    return templates

def get_html_file_name(page_name_templates, file_path, page_id):
    # Create the page file id string
    file_id = page_name_templates[len(str(page_id))].format(page_id)
    
    # Return the page file path and id string
    return os.path.join(file_path, f'{file_id}.htm'), file_id

def get_site_info(data_path, vertical, dir_name):
    # Extract the website name
    website = dir_name.split('(')[0]
    
    # extract the number of web pages
    num_pages = int(dir_name.split('(')[1].strip(')'))
    
    # Get the page name templates
    page_name_templates = __get_page_name_templates(num_pages)
    
    # Get the file path
    file_path = os.path.join(data_path, vertical, dir_name)
    
    return website, num_pages, page_name_templates, file_path

def remove_hidden_dir(websites):
    reduced_list = []
    for website in websites:
        if website[0]!='.':
            reduced_list.append(website)
    return reduced_list

def get_text_nodes(root, fixedNodes={}):
    tree = root.getroottree()
    node_dict = {}
    
    for node_id, elem in enumerate(root.iter()): #node_id is DFS position of node in the DOM tree.
        text=''
        if elem.text !=None:
            text += elem.text
        if elem.tail !=None:
            text += elem.tail
        if elem.tag not in ['script', 'style'] and text.strip()!='':
            absxpath = tree.getpath(elem)
            isVariableNode = True
            if len(fixedNodes)!=0 and (absxpath in list(fixedNodes.absxpath)) and (text in list(fixedNodes.text)):
                isVariableNode = False
            nodeDetails = DOMNodeDetails(absxpath, text, isVariableNode, [], [], '0')
            node_dict[node_id] = nodeDetails
    
    return node_dict