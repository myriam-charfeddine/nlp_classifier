from bs4 import BeautifulSoup
import re
import unicodedata

class Cleaner:
    def __init__(self):
        pass

    def add_line_beaks(self, text):
        return text.replace('</p>', '</p>\n')
    
    def remove_html(self, text):
        clean_text = BeautifulSoup(text, "html.parser").text #extracts all the visible text content from the HTML structure while ignoring the tags and any hidden elements
        return clean_text
    
    def clean_text(self, text):
        text = self.add_line_beaks(text)
        text = self.remove_html(text)
        text = re.sub(r'.\\u[0-9a-fA-F]{4}', '', text) 
        text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode()
        text = text.strip()
        return text