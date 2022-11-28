import string
import numpy as np
from Utils.logger import logger

filename= '../data/glove.6B.100d.txt'

class pretrainedWordEmeddings:
    def __init__(self, filename):
        logger.info(f'Loading pretrained word emeddings from: {filename}')
        with open(filename) as f:
            self.word2embeddings = f.readlines()
        self.word2embeddings = self.get_pretrained_word2embeddings(self.word2embeddings)

    def get_pretrained_word2embeddings(self, text_content):
        word2embeddings = dict()
        for line in text_content:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            word2embeddings[word] = coefs
        
        logger.info(f'Loaded {len(word2embeddings)} pretrained word vectors')
        
        return word2embeddings
    
    def get_embedding(self, word):
        word = word.lower().strip(string.punctuation)
        try:
            embedding = self.word2embeddings[word]
        except:
            logger.debug(f'Could not find word embedding for: "{word}", using zeros!');
            embedding = np.zeros((100)).astype('float32')
        
        return embedding

if __name__ == "__main__":
    pretrainedWordEmeddings(filename)
