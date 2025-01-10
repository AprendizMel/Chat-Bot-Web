#nltk_utils.py
# La biblioteca NLTK se utiliza para construir programas en Python que trabajan con datos del lenguaje humano 
# para aplicarlos en el procesamiento estadístico del lenguaje natural (NLP).
import numpy as np
import nltk
from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()

def tokenize(sentence):
    """
    divide la oración en un array de palabras/tokens
    un token puede ser una palabra, un carácter de puntuación o un número
"""
    return nltk.word_tokenize(sentence)


def stem(word):
    """
    stemming = encontrar la forma raíz de la palabra
    ejemplos:
    palabras = ["organize", "organizes", "organizing"]
    palabras = [stem(w) for w in palabras]
    -> ["organ", "organ", "organ"]
"""

    return stemmer.stem(word.lower())


def bag_of_words(tokenized_sentence, words):
    """
    devuelve el arreglo de bolsa de palabras:
    1 por cada palabra conocida que exista en la frase, 0 en caso contrario
    ejemplo:
    frase = ["hola", "cómo", "estás", "tú"]
    palabras = ["hola", "yo", "tú", "adiós", "gracias", "genial"]
    bog   = [  1 ,    0 ,    1 ,   1 ,    0 ,    0 ,      0]
"""

    # realizar el stemming de cada palabra
    sentence_words = [stem(word) for word in tokenized_sentence]
   
    # inicializar la bolsa con 0 para cada palabra
    bag = np.zeros(len(words), dtype=np.float32)
    for idx, w in enumerate(words):
        if w in sentence_words: 
            bag[idx] = 1
    return bag