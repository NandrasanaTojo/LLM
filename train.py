import tensorflow as tf
from models.seq2seq import Encoder, Decoder

# Assuming you define the values for these constants elsewhere in your script
# VOCAB_SIZE, EMBEDDING_DIM, UNITS, BATCH_SIZE

# Define the loss function
def loss_function(real, pred, loss_object):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)

    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask

    return tf.reduce_mean(loss_)

# Define a single training step
def train_step(inp, targ, enc_hidden, encoder, decoder, optimizer, loss_object, tokenizer):
    with tf.GradientTape() as tape:
        enc_output, enc_hidden = encoder(inp, enc_hidden)
        dec_hidden = enc_hidden
        dec_input = tf.expand_dims([tokenizer.word_index['<start>']] * BATCH_SIZE, 1)

        # Teacher forcing - feeding the target as the next input
        for t in range(1, targ.shape[1]):
            # passing the output of encoder, along with decoder's input to decoder
            predictions, dec_hidden, _ = decoder(dec_input, dec_hidden, enc_output)
            loss = loss_function(targ[:, t], predictions, loss_object)
            dec_input = tf.expand_dims(targ[:, t], 1)  # Teacher forcing

        batch_loss = (loss / int(targ.shape[1]))
        variables = encoder.trainable_variables + decoder.trainable_variables
        gradients = tape.gradient(loss, variables)
        optimizer.apply_gradients(zip(gradients, variables))
        return batch_loss

# The 'train' function that handles the training process
def train(dataset, steps_per_epoch, encoder, decoder, optimizer, loss_object, tokenizer):
    for epoch in range(10):  # Assuming you want to train for 10 epochs
        enc_hidden = encoder.initialize_hidden_state()
        total_loss = 0

        for (batch, (inp, targ)) in enumerate(dataset.take(steps_per_epoch)):
            batch_loss = train_step(inp, targ, enc_hidden, encoder, decoder, optimizer, loss_object, tokenizer)
            total_loss += batch_loss.numpy()

            if batch % 100 == 0:
                print(f'Epoch {epoch+1} Batch {batch} Loss {batch_loss.numpy():.4f}')

        print(f'Epoch {epoch+1} Loss {total_loss/steps_per_epoch:.4f}')
