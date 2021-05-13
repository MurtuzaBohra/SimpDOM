import re
import lxml.html
import lxml
import logging

class DOMTree:
    """
        This represents the DOM Tree of the Page.
        Any Operation on the DOM Tree of the Page should be accessed through this Class
    """
    def __init__(self, cpid, page_content):
        self.cpid = cpid
        self.root = None
        try:
            sm_html_parser = lxml.html.HTMLParser(no_network=True,
                                                  remove_comments=True,
                                                  remove_pis=True)
            page_content = re.sub('&nbsp;', ' ', page_content)
            tree = lxml.etree.ElementTree(lxml.html.fromstring(page_content, parser=sm_html_parser))
            self.root = tree.getroot()
        except Exception as e:
            logging.info('Exception occurred during creating Element Tree. Ignoring the exception')
            logging.info(e)
            pass

    def get_page_root(self):
        return self.root

    def get_nodes_represented_by_xpath(self, xpath):
        return self.root.xpath(xpath)
