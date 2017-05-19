import tensorflow as tf






if __name__ == '__main__':


    lstm_size = 


    lstm = tf.contrib.rnn.BasicLSTMCell(lstm_size, state_is_tuple=False)
    stacked_lstm = tf.contrib.rnn.MultiRNNCell([lstm] * number_of_layers,
        state_is_tuple=False)
    
    
    # Initial state of the LSTM memory.
    # initial_state  = state = tf.zeros([batch_size, lstm.state_size])
    initial_state  = state = stacked_lstm.zero_state(batch_size, tf.float32)
    probabilities = []
    loss = 0.0
    for current_batch_of_words in words_in_dataset:
        # The value of state is updated after processing each batch of words.
        output, state = stacked_lstm(current_batch_of_words, state)

        # The LSTM output can be used to make next word predictions
        logits = tf.matmul(output, softmax_w) + softmax_b
        probabilities.append(tf.nn.softmax(logits))
    loss += loss_function(probabilities, target_words)