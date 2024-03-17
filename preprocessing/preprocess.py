import numpy as np
import tensorflow as tf
import re
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

def preprocess_sentence(sentence):
    sentence = sentence.lower().strip() 
    sentence = re.sub(r"([?.!,¿])", r" \1 ", sentence)
    sentence = re.sub(r'[" "]+', " ", sentence)
    return sentence

def load_conversations(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    questions, answers = [], []
    for line in lines:
        parts = line.split('\t')
        questions.append(preprocess_sentence(parts[0]))
        answers.append(preprocess_sentence(parts[1].strip()))
    return questions, answers

def tokenize_and_pad(questions, answers):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(questions + answers)
    questions_seq = tokenizer.texts_to_sequences(questions)
    answers_seq = tokenizer.texts_to_sequences(answers)
    max_length = max(len(x) for x in questions_seq + answers_seq)
    questions_padded = pad_sequences(questions_seq, maxlen=max_length, padding='post')
    answers_padded = pad_sequences(answers_seq, maxlen=max_length, padding='post')
    return questions_padded, answers_padded, tokenizer
