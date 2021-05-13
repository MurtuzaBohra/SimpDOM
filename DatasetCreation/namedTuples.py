import collections
DOMNodeDetails = collections.namedtuple('DOMNodeDetails', 'absxpath text isVariableNode friendNodes partnerNodes label')
DataLoaderNodeDetail = collections.namedtuple('DataLoaderNodeDetail', 'page_ID xpath_seq leaf_tag_index pos_index node_char_seqs node_words node_sent_len friend_char_seqs friend_words friend_sent_len partner_char_seqs partner_words partner_sent_len')