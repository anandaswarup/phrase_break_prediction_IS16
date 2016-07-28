#!/usr/bin/env python

"""Script to train phrase break prediction using uni-directional Elman RNN"""

# python imports
import os, codecs, argparse
from data_load import process_data

# numpy imports
import numpy as np

# scikit learn imports
from sklearn.cross_validation import train_test_split
from sklearn.metrics import f1_score

# keras imports
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Merge, TimeDistributedDense
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import SimpleRNN
from keras.optimizers import SGD, RMSprop, Adam
from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping

def batch_generator(text, labels, win_size = 1):
    """Generator to generate batches of data (one sequence is one batch)"""
    while 1:
        for text_seq, label_seq in zip(text, labels):
            x = np.array(text_seq)
            x = x.reshape(1, -1)
            
            # Convert integer class labels to one-hot vectors
            y = np_utils.to_categorical(label_seq)
            y = y.reshape(1, y.shape[0], y.shape[1])

            yield x, y 

def build_RNN_model(vocab_size, embedding_dims, rnn_layer_dim, num_classes):
    """Build the RNN model"""
    model = Sequential()             # Sequential model
    # Embedding layer
    model.add(Embedding(vocab_size, embedding_dims))
    # Recurrent layer
    model.add(SimpleRNN(int(rnn_layer_dim), init = 'glorot_uniform', inner_init = 'orthogonal', activation = 'tanh', W_regularizer = None, U_regularizer = None, b_regularizer = None, dropout_W = 0.0, dropout_U = 0.0, return_sequences = True, stateful = False))
    # Time distributed dense layer (activation is softmax, since it is a classification problem)
    model.add(TimeDistributedDense(num_classes, init = 'glorot_uniform', activation = 'softmax'))

    return model

def run_experiment():
    """Main method (where all the work is done)"""
    
    random_seed = np.random.seed(1337)         # For reproducibility

    ########## COMMAND LINE ARGUMENTS ##########
    parser = argparse.ArgumentParser(description = "Train an Elman RNN to do phrase break prediction using word embeddings")
    parser.add_argument("--data_dir", help = "Path to the data dir", required = True)
    parser.add_argument("--embedding_dims", type = int, help = "Dimensions of word embeddings", required = True)
    parser.add_argument("--rnn_layer_dim", type = int, help = "Dimension of the recurrent (hidden) layer", required = True)
    parser.add_argument("--l_rate", type = float, help = "Learning rate for SGD optimizer", required = True)
    parser.add_argument("--momentum", type = float, help = "Momentum factor for SGD optimizer", required = True)
    parser.add_argument("--num_epochs", type = int, help = "Number of epochs to train the model", required = True)
    parser.add_argument("--model_dir", help = "Path to the dir where model weights will be stored", required = True)

    args = parser.parse_args()

    data_dir = args.data_dir
    embedding_dims = args.embedding_dims
    rnn_layer_dim = args.rnn_layer_dim
    l_rate = args.l_rate
    mom = args.momentum
    num_epochs = args.num_epochs
    model_dir = args.model_dir
    
    ########## READING AND PROCESSING DATA ##########
    print("Reading and processing data")
    text, labels, word2index, label2index = process_data(data_dir)
    print("Number of text / label sequences: ", len(text), "/", len(labels))
    
    # Reversing word2index and label2index
    index2word = dict((v, k) for (k, v) in word2index.items())
    index2label = dict((v, k) for (k, v) in label2index.items())
    
    # Split the data into train and test
    text_train_valid, text_test, labels_train_valid, labels_test = train_test_split(text, labels, test_size = 0.1, random_state = random_seed)
    # Split into train and valid sets
    text_train, text_valid, labels_train, labels_valid = train_test_split(text_train_valid, labels_train_valid, test_size = 0.1, random_state = random_seed)

    print("Number of text / label sequences for training :", len(text_train), "/", len(labels_train))
    print("Number of text / label sequences for validation :", len(text_valid), "/", len(labels_valid))
    print("Number of text / label sequences for test :", len(text_test), "/", len(labels_test))
    
    ########## RNN MODEL BUILDING AND COMPILING ##########
    vocab_size = len(word2index)
    num_classes = len(label2index)
    maxlen = np.max([len(seq) for seq in text])
    # RNN model
    model = build_RNN_model(vocab_size, embedding_dims, rnn_layer_dim, num_classes)
    # Optimizer
    sgd = SGD(lr = l_rate, decay = 1e-6, momentum = mom, nesterov = True)
   
    print("Compiling the model")
    model.compile(loss = 'categorical_crossentropy', optimizer = 'sgd', metrics = ['accuracy'])

    ########## RNN TRAINING AND EVALUATION ##########
    print("Training the model")
    checkpointer = ModelCheckpoint(filepath = os.path.join(model_dir, "RNN_weights.hdf5"), monitor = 'val_loss', verbose = 1, save_best_only = True, mode = 'auto')
    earlystopping = EarlyStopping(monitor = 'val_loss', patience = 10, verbose = 1, mode = 'auto')
    history = model.fit_generator(batch_generator(text_train, labels_train), samples_per_epoch = len(text_train), nb_epoch = num_epochs, show_accuracy = True, callbacks = [checkpointer, earlystopping], validation_data = batch_generator(text_valid, labels_valid), nb_val_samples = len(text_valid), nb_worker = 1)

    print("Print loading best model weights")
    model.load_weights(os.path.join(model_dir, "RNN_weights.hdf5"))

    print("Evaluating model on test data with best model weights")
    f1 = []
    for test_sentence, ground_truth in zip(text_test, labels_test):
        x = np.array(test_sentence)
        x = x.reshape(1, -1)
        predicted_labels = model.predict_classes(x)
        true_labels = np.array(ground_truth)
        true_labels = true_labels.reshape(1, -1)
        sentence_f1 = f1_score(true_labels, predicted_labels, pos_label = None, average = 'micro')
        f1.append(sentence_f1)
    print("Mean F1 score over test sentences : %f" %(np.mean(f1)))

if __name__ == "__main__":
    run_experiment()
