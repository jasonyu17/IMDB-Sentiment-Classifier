# move to src later, figure out the module error
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
    
    def clean(self, text):
        text = self.clean_html(text)

        if self.lowercase:
            text = text.lower()

        if self.remove_punctuation:
            text = re.sub(f"[{re.escape(string.punctuation)}]", " ", text)
        
        words = text.split()
        if self.remove_stopwords:
            words = [word for word in words if word not in stop_words]

        return ' '.join(words)
    
    def process(self, texts, textcol='review', drop_original=True):
        texts['cleaned_review'] = texts[textcol].apply(self.clean)
        if drop_original:
            texts.drop(columns=[textcol], inplace=True)
        return texts
