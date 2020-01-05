import spacy
spacy.load('en_core_web_sm')
from spacy.lang.en import English
parser = English()

import nltk
nltk.download('wordnet')

from nltk.corpus import wordnet as wn

from nltk.stem.wordnet import WordNetLemmatizer

nltk.download('stopwords')

from datetime import datetime

from gensim import corpora
import pickle
import gensim
import random

import os

import pyLDAvis.gensim
    
def tokenize(text):
    lda_tokens = []
    tokens = parser(text)
    for token in tokens:
        if token.orth_.isspace():
            continue
        elif token.like_url:
            lda_tokens.append('URL')
        elif token.orth_.startswith('@'):
            lda_tokens.append('SCREEN_NAME')
        else:
            lda_tokens.append(token.lower_)
    return lda_tokens

def get_lemma(word):
    lemma = wn.morphy(word)
    if lemma is None:
        return word
    else:
        return lemma

def get_lemma2(word):
    return WordNetLemmatizer().lemmatize(word)

def prepare_text_for_lda(text):
    en_stop = set(nltk.corpus.stopwords.words('english'))
    tokens = tokenize(text)
    tokens = [token for token in tokens if len(token) > 3]
    tokens = [token for token in tokens if token not in en_stop]
    tokens = [get_lemma(token) for token in tokens]
    return tokens

def visualisation():
    dictionary = gensim.corpora.Dictionary.load('dictionary.gensim')
    corpus = pickle.load(open('corpus.pkl', 'rb'))
    lda = gensim.models.ldamodel.LdaModel.load('model5.gensim')

    print('models loaded!')
    lda_display = pyLDAvis.gensim.prepare(lda, corpus, dictionary, sort_topics=False)
    pyLDAvis.display(lda_display)

def create_model(file_name, filepath):
    file_name = file_name.split('.')[0]

    text_data = []
    with open(filepath, 'r') as f:
        for line in f:
            # print(line)
            tokens = prepare_text_for_lda(line)
            # print(tokens)
            # print(tokens[0])
            text_data.append(tokens)
            # if random.random() > .99:
            #     # print(tokens)
            #     text_data.append(tokens)

    # print(text_data)

    dictionary = corpora.Dictionary(text_data)
    corpus = [dictionary.doc2bow(text) for text in text_data]
    # pickle.dump(corpus, open(output_location + '/corpus.pkl', 'wb'))
    # dictionary.save(output_location + '/dictionary.gensim')

    pickle.dump(corpus, open('corpus.pkl', 'wb'))
    dictionary.save('dictionary.gensim')

    NUM_TOPICS = 20
    ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics = NUM_TOPICS, id2word=dictionary, passes=100)
    # ldamodel.save(output_location + '/' + filename + '.gensim')
    ldamodel.save(file_name + '.gensim')
    topics = ldamodel.print_topics(num_words=20)
    return topics

def generate_output_row(equation):
    equation = equation.replace(' + ', ';')
    equation = equation.replace('"', '')
    equation = equation.replace('*', ';')
    return equation

def main():
    start_time = datetime.now()

    input_folder = '/Users/puudeli/Documents/python/input'
    output_folder = '/Users/puudeli/Documents/python/output'

    for txt_file in os.listdir(input_folder):
        if txt_file.endswith('.txt'):
            source_filepath = os.path.join(input_folder, txt_file)
            topics = create_model(txt_file, source_filepath)

            output_location = os.path.join(output_folder, txt_file)
            with open(output_location, 'w') as f:
                for topic in topics:
                    print(topic)            
                    f.write(generate_output_row(topic[1]))
                    f.write('\n')


    # visualisation()


    print('total time', (datetime.now() - start_time).seconds, 'seconds')


if __name__ == "__main__":
    main()