# main.py
from preprocessing.preprocess import load_conversations, tokenize_and_pad
from models.seq2seq import Encoder, Decoder
from train import train
import tensorflow as tf

if __name__ == "__main__":
    # Suppose we've calculated vocab size based on our tokenizer
    VOCAB_SIZE = 10000
    EMBEDDING_DIM = 256
    UNITS = 1024
    BATCH_SIZE = 5

    # Get the conversation pairs
    questions, answers = load_conversations('data/conversations.txt')
    print(f"questions: {questions}")
    print(f"answers: {answers}")
    # Tokenize and pad conversations
    questions_padded, answers_padded, tokenizer = tokenize_and_pad(questions, answers)
    print(f"questions_padded: {questions_padded}")
    print(f"answers_padded: {answers_padded}")
    print(f"tokenizer: {tokenizer}")
    # Calculate max length based on the padded sequences
    max_length = questions_padded.shape[1]
    print(f"max_length: {max_length}")
    # Calculate dataset size and steps per epoch
    dataset_size = len(questions_padded)
    print(f"dataset_size: {dataset_size}")
    steps_per_epoch = dataset_size 
    if steps_per_epoch == 0:
        print(f"Error: Your BATCH_SIZE ({BATCH_SIZE}) is too large for the dataset size ({dataset_size}).")
        exit()
        
    # Create a tf.data.Dataset from the padded questions and answers
    dataset = tf.data.Dataset.from_tensor_slices((questions_padded, answers_padded))
    dataset = dataset.shuffle(dataset_size).batch(BATCH_SIZE)

    # Instantiate models
    encoder = Encoder(VOCAB_SIZE, EMBEDDING_DIM, UNITS, BATCH_SIZE)
    decoder = Decoder(VOCAB_SIZE, EMBEDDING_DIM, UNITS, BATCH_SIZE)
   
    # Initialize the optimizer and loss function
    optimizer = tf.keras.optimizers.Adam()
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction='none')
    
    # Train the model
    train(dataset, steps_per_epoch, encoder, decoder, optimizer, loss_object, tokenizer)
