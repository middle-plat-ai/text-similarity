##单独使用词向量或字向量
from keras.optimizers import Adam
from keras.layers import Input, LSTM, Dense, merge, Conv1D, MaxPooling1D, Flatten, Embedding, Dropout, Bidirectional
from keras.layers import *
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras import backend as K
from keras import regularizers
from keras.engine.topology import Layer
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from keras.layers.merge import concatenate

def fscore(y_true, y_pred):
    y_pred = K.round(y_pred)
    tp = K.sum(y_true * y_pred)
    pred_p = K.sum(y_pred) + K.epsilon()
    actual_p = K.sum(y_true) + K.epsilon()
    precision = tp / pred_p
    recall = tp / actual_p
    return (2 * precision * recall) / (precision + recall + K.epsilon())

def my_bilstm(vocab_size, emb_matrix, maxlen, n, rnn_num, hidden_num, lr=0.001, loss='binary_crossentropy'):
    input1 = Input(shape=(maxlen * n,))
    input2 = Input(shape=(maxlen * n,))

    embedding = Embedding(vocab_size + 1,
                          emb_matrix.shape[1],
                          weights=[emb_matrix],
                          input_length=maxlen * n,
                          trainable=False)

    #lstm_layer = Bidirectional(LSTM(128, dropout=0.2, recurrent_dropout=0.2, return_sequences=True))
    lstm1 = Bidirectional(LSTM(rnn_num, return_sequences=True))

    x1_embed = SpatialDropout1D(0.3)(embedding(input1))
    x2_embed = SpatialDropout1D(0.3)(embedding(input2))
	
    x1_encoded = Dropout(0.3)(lstm1(x1_embed))
    x1_encoded = Concatenate(axis=-1)([x1_encoded, x1_embed])
    x1_encoded = GlobalAveragePooling1D()(x1_encoded)
    
    x2_encoded = Dropout(0.3)(lstm1(x2_embed))
    x2_encoded = Concatenate(axis=-1)([x2_encoded, x2_embed])
    x2_encoded = GlobalAveragePooling1D()(x2_encoded)
    
    diff  = Lambda(lambda x: K.abs(x[0] - x[1]))([x1_encoded, x2_encoded]) 
    angle = Lambda(lambda x: x[0] * x[1])([x1_encoded, x2_encoded])   
    
	
    # Classifier
    merged = concatenate([diff, angle])
    merged = Dropout(0.5)(merged)
    merged = BatchNormalization()(merged) 
    merged = Dense(hidden_num, activation="relu")(merged)
    merged = Dropout(0.5)(merged)
    merged = BatchNormalization()(merged)
	
    preds = Dense(1, activation='sigmoid', kernel_regularizer=regularizers.l2(1e-5))(merged)

    model = Model(inputs=[input1, input2], outputs=preds)

    model.compile(optimizer=Adam(lr=lr, decay=1e-6),
                  loss=loss,
                  metrics=['accuracy', fscore])
    return model

##词向量和字向量
##词向量和字向量
def my_bilstm_wc(nb_words, nb_chars, embedding_word_matrix, embedding_char_matrix, maxlen1, maxlen2, n, rnn_num, hidden_num, lr=0.001, loss='binary_crossentropy'):
    embedding_word_layer = Embedding(nb_words + 1, 300, input_length=maxlen1 * n, \
								weights=[embedding_word_matrix], trainable=False)

    embedding_char_layer = Embedding(nb_chars + 1, 300, input_length=maxlen2 * n, \
                                weights=[embedding_char_matrix], trainable=False)

    # Creating LSTM Encoder
    lstm1 = Bidirectional(LSTM(rnn_num, return_sequences=True))

    # Creating LSTM Encoder layer for First Sentence
    word1_input = Input(shape=(maxlen1 * n,))
    word1_embedded = SpatialDropout1D(0.3)(embedding_word_layer(word1_input))
    x1_word = lstm1(word1_embedded)
    #x1_word = lstm_layer2(x1_word)
    x1_encoded = Dropout(0.3)(x1_word)
    x1_encoded = Concatenate(axis=-1)([x1_encoded, word1_embedded])
    x1_encoded = GlobalAveragePooling1D()(x1_encoded)

    char1_input = Input(shape=(maxlen2 * n,))
    char1_embedded = SpatialDropout1D(0.3)(embedding_char_layer(char1_input))
    x1_char = lstm1(char1_embedded)
    #x1_char = lstm_layer2(x1_char)
    char1_encoded = Dropout(0.3)(x1_char)
    char1_encoded = Concatenate(axis=-1)([char1_encoded, char1_embedded])
    char1_encoded = GlobalAveragePooling1D()(char1_encoded)

    # Creating LSTM Encoder layer for Second Sentence
    word2_input = Input(shape=(maxlen1 * n,))
    word2_embedded = SpatialDropout1D(0.3)(embedding_word_layer(word2_input))
    x2_word = lstm1(word2_embedded)
    #x2_word = lstm_layer2(x2_word)
    x2_encoded = Dropout(0.3)(x2_word)
    x2_encoded = Concatenate(axis=-1)([x2_encoded, word2_embedded])
    x2_encoded = GlobalAveragePooling1D()(x2_encoded)

    char2_input = Input(shape=(maxlen2 * n,))
    char2_embedded = SpatialDropout1D(0.3)(embedding_char_layer(char2_input))
    x2_char = lstm1(char2_embedded)
    #x2_char = lstm_layer2(x2_char)
    char2_encoded = Dropout(0.3)(x2_char)
    char2_encoded = Concatenate(axis=-1)([char2_encoded, char2_embedded])
    char2_encoded = GlobalAveragePooling1D()(char2_encoded)

    x1 = Concatenate(axis=1)([x1_encoded, char1_encoded])
    x2 = Concatenate(axis=1)([x2_encoded, char2_encoded])

    diff  = Lambda(lambda x: K.abs(x[0] - x[1]))([x1, x2])
    angle = Lambda(lambda x: x[0] * x[1])([x1, x2])
    # Creating leaks input
    # leaks_input = Input(shape=(leaks_train.shape[1],))
    # leaks_dense = Dense(64, activation='relu')(leaks_input)

    # Merging two LSTM encodes vectors from sentences to
    # pass it to dense layer applying dropout and batch normalisation
    #merged = concatenate([x1_word, x1_char, x2_word, x2_char])
    merged = concatenate([diff, angle])
    merged = Dropout(0.5)(merged)
    merged = BatchNormalization()(merged)
    merged = Dense(hidden_num, activation="relu")(merged)
    merged = Dropout(0.5)(merged)
    merged = BatchNormalization()(merged)
    preds = Dense(1, activation='sigmoid', kernel_regularizer=regularizers.l2(1e-5))(merged)

    model = Model(inputs=[word1_input, char1_input, word2_input, char2_input], outputs=preds)
    model.compile(optimizer=Adam(lr=lr, decay=1e-6),
                  loss=loss,
                  metrics=['accuracy', fscore])
    return model
