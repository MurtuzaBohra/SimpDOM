import os

from Utils.logger import logger
from DatasetCreation.namedTuples import DOMNodeDetails

def __get_page_name_templates(num_pages_str):
    # Create the templates for the page file name depending on how many pages there are
    num_pages_len = max(len(num_pages_str), 4)
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
    num_pages_str = dir_name.split('(')[1].strip(')')
    
    # Get the page name templates
    page_name_templates = __get_page_name_templates(num_pages_str)
    
    # Get the file path
    file_path = os.path.join(data_path, vertical, dir_name)
    
    return website, int(num_pages_str), page_name_templates, file_path

def remove_hidden_dir(websites):
    reduced_list = []
    for website in websites:
        if website[0]!='.':
            reduced_list.append(website)
    return reduced_list

def __get_node_text(elem):
    # Get the node text
    text = ''
    if elem.text != None:
        text += elem.text
    if elem.tail != None:
        text += ' ' + elem.tail

    # Make sure the text is stripped
    return text.strip()

__IRRELEVANT_ATTRIBUTES_SET = {'style', 'width', 'height', 'color', 'size', 'face', 'frameborder', \
                               'scrolling', 'accesskey', 'onclick', 'valign', 'halign', 'onmousedown', \
                               'align', 'reviewtype', 'border', 'align', 'colspan', 'background', 'overlay', \
                               'onmouseout', 'onmouseover', 'bgcolor', 'meta:ctype', 'meta:image', 'mapleultparams', \
                               'hspace', 'vspace', 'maxlength', 'nowrap', 'cptest:id', 'maxlength', 'tabindex', \
                               'counter', 'cols', 'anti', 'rows', 'cellspacing', 'cellpadding', 'onchange', 'onkeyup', \
                               'headers', 'data-cmelementid', 'foo', 'noshade', 'center', 'clear', 'length', 'rowspan', \
                               'cssclass', 'xmlns:htm', 'mso-ansi-language:', 'line', 'color:', 'mso-fareast-font-family:', \
                               'color:black', 'font-family:', 'mce_style', 'arial', 'mso-bidi-font-size:', 'jade_visible', \
                               'mso-bidi-font-family:arial', 'u4:st', 'scrolldelay', 'area', 'black', 'color:red', 'lang', \
                               'sans-serif', 'xml:lang', 'method', 'action', 'onsubmit', 'autocomplete', 'dir', 'onlogin', \
                               'data-count', 'data-via', 'onkeypress', 'onblur', 'minmax_bound', 'onfocus', 'onmouseup', 'share_url'}
__RELEVANT_ATTRIBUTES_SET = {'role', 'class', 'id', 'name', 'title', 'for', 'target', 'rel', 'ref', 'href', 'src', \
                             'property', 'alt', 'content', 'hidden', 'type', 'value', 'disabled',  'selected', \
                             'checked', 'datatype', 'scope', 'to', 'from', 'with', 'on', 'abbr', 'alttext'}
# Stores all the attributes being encountered
__all_atrib_set = set()

def report_missed_attributes():
    global __all_atrib_set
    
    # Check and report on the missed attributes
    missed_attrib_set = __all_atrib_set - __RELEVANT_ATTRIBUTES_SET - __IRRELEVANT_ATTRIBUTES_SET
    if missed_attrib_set:
        logger.info(f'Missed the following attributes: {missed_attrib_set}')
    
    # Re-set the set
    __all_atrib_set = set()
    
    # Return the difference
    return missed_attrib_set

def __get_node_attributes(elem):
    global __all_atrib_set

    # Get the attributes names
    __all_atrib_set.update(elem.attrib.keys())
    
    # Only select as subset of interesting attributes
    return {key : value for key, value in elem.attrib.items() if key in __RELEVANT_ATTRIBUTES_SET}

def get_text_nodes(root, fixed_nodes_df = None):
    tree = root.getroottree()
    node_dict = {}
    
    # Get the list of fixed nodes xpath/text pairs
    filxed_nodes = []
    if (fixed_nodes_df is not None) and (len(fixed_nodes_df) > 0):
        filxed_nodes = zip(fixed_nodes_df['absxpath'].values, fixed_nodes_df['text'].values)
    
    # Iterate over all of the tree nodes, the node_id is DFS position of node in the DOM tree.
    for node_id, elem in enumerate(root.iter()):
        if elem.tag not in ['script', 'style']:
            # Get the node text
            text = __get_node_text(elem)
                
            # If there is some text present in the node
            if text != '':
                # Get the xpath of the node
                absxpath = tree.getpath(elem)
                
                # Get node attributes
                attributes = __get_node_attributes(elem)
                
                # Check if this is a variable node
                is_variable_node = (absxpath, text) not in filxed_nodes

                # Record the node details
                node_dict[node_id] = DOMNodeDetails(absxpath, text, attributes, is_variable_node, [], [], '0')
    
    return node_dict