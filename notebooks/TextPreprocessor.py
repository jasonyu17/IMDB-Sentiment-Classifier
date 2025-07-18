#move to src later, figure out the module error
import re
import string
from bs4 import BeautifulSoup
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

class TextPreprocessor:
    def __init__(self, remove_stopwords=True, lowercase=True, remove_punctuation=True):
        self.remove_stopwords = remove_stopwords
        self.lowercase = lowercase
        self.remove_punctuation = remove_punctuation

    def clean_html(self, text):
        return BeautifulSoup(text, "html.parser").get_text()
    
    