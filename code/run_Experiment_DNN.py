#!/usr/bin/env python

"""Script to train a DNN to do phrase break prediction"""

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
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.embeddings import Embedding
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping

def sequences_to_matrix(sequences, win_size = 1):
    """Convert a list of sequences to a list of elements (destroying sequence information)"""
    data = []
    for seq in sequences:
        padded_seq = win_size // 2 * [0] + seq + win_size // 2 * [0]        # index 0 is for padding token in words2index
        cwin = [padded_seq[i : (i + win_size)] for i in range(len(seq))]
        for vector in cwin:
            data.append(vector)

    return np.array(data)

def build_DNN_model(input_dims, vocab_size, embedding_dims, hidden_layers, num_classes):
    """Build the DNN model"""
    
    model = Sequential()    # Sequential model 
    # Embedding layer
    model.add(Embedding(vocab_size, embedding_dims, input_length = input_dims))
    # Flatten output of embedding layer
    model.add(Flatten())
    # Hidden layers
    for layer in hidden_layers:
        # Dimensionality of hidden layer
        layer_dim = layer[:-1]       
        # Hidden layer activations
        if layer[-1]== 'N':
            layer_activation = 'tanh'
        elif layer[-1] == 'R':
            layer_activation = 'relu'
        elif layer[-1] == 'S':
            layer_activation = 'sigmoid'
        elif layer[-1] == 'L':
            layer_activation = 'linear'
        else:
            raise ValueError("Unknown activation")
        layer_dim = int(layer_dim)
        layer_activation = str(layer_activation)
        # Add the hidden layer to the model
        model.add(Dense(layer_dim, init = 'glorot_uniform', activation = layer_activation))
    #output layer (activation is softmax, since it is a classification problem)
    model.add(Dense(num_classes, init = 'glorot_uniform', activation = 'softmax'))

    return model

def run_experiment():
    """Main method (where all the work is done)"""
    
    random_seed = np.random.seed(1337)         # For reproducibility

    ########## COMMAND LINE ARGUMENTS ##########
    parser = argparse.ArgumentParser(description = "Train a DNN to do phrase break prediction using word embeddings")
    parser.add_argument("--data_dir", help = "Path to the data dir", required = True)
    parser.add_argument("--embedding_dims", type = int, help = "Dimensions of word embeddings", required = True)
    parser.add_argument("--hidden_layers", help = "Hidden layers size along with activations seperated by a `:` (For eg. input of 512N:512N denotes two hidden layers each of size 512 units and tanh() nonlinear activation), Current activations supported are : N - tanh(), R - relu, S - sigmoid, L - linear", required = True)
    parser.add_argument("--l_rate", type = float, help = "Learning rate for SGD optimizer", required = True)
    parser.add_argument("--momentum", type = float, help = "Momentum factor for SGD optimizer", required = True)
    parser.add_argument("--num_epochs", type = int, help = "Number of epochs to train the model", required = True)
    parser.add_argument("--context_win_size", type = int, help = "Size of context window", required = True)
    parser.add_argument("--batch_size", type = int, help = "Size of minibatch", required = True)
    parser.add_argument("--model_dir", help = "Path to the dir where model weights will be stored", required = True)

    args = parser.parse_args()

    data_dir = args.data_dir
    embedding_dims = args.embedding_dims
    hidden_layers = args.hidden_layers.split(":")
    l_rate = args.l_rate
    mom = args.momentum
    num_epochs = args.num_epochs
    context_win_size = args.context_win_size
    batch_size = args.batch_size
    model_dir = args.model_dir
    
    ########## READING AND PROCESSING DATA ##########
    print("Reading and processing data")

    text, labels, word2index, label2index = process_data(data_dir)
    
    print("Number of text / label sequences: ", len(text), "/", len(labels))

    # Reversing word2index and label2index
    index2word = dict((v, k) for (k, v) in word2index.items())
    index2label = dict((v, k) for (k, v) in label2index.items())
    
    # Split the data into train and test
    text_train, text_test, labels_train, labels_test = train_test_split(text, labels, test_size = 0.1, random_state = random_seed)

    # Generate matrices for training DNN
    x_train = sequences_to_matrix(text_train, win_size = context_win_size)
    y_train = sequences_to_matrix(labels_train)
    
    # Convert labels from integer class labels to one-hot vectors (only for training)
    y_train = np_utils.to_categorical(y_train)
    
    # Generate matrices for test
    #x_test = sequences_to_matrix(text_test, win_size = context_win_size)
    #y_test = sequences_to_matrix(labels_test)

    print("Training  : Text / Labels shape : ", x_train.shape, "/", y_train.shape)
    #print("Training : Text / Labels shape : ", x_test.shape, "/", y_test.shape)

    ########## DNN MODEL BUILDING AND COMPILING ##########
    print("Building DNN model")

    vocab_size = len(word2index)
    input_dims = x_train.shape[1]
    num_classes = y_train.shape[1]
    
    # DNN model
    model = build_DNN_model(input_dims, vocab_size, embedding_dims, hidden_layers, num_classes)
    # Optimizer
    sgd = SGD(lr = l_rate, decay = 1e-6, momentum = mom, nesterov = True) 
    
    print("Compiling the model")
    model.compile(loss = "categorical_crossentropy", optimizer = sgd)

    ########## DNN TRAINING AND TESTING ##########
    print("Training the model")
    
    #Save model weights after each epoch if the validation loss decreases. Each time the model file is written
    #the earlier best file is overwritten
    checkpointer = ModelCheckpoint(filepath = os.path.join(model_dir, "DNN_weights.hdf5"), monitor = 'val_loss', verbose = 1, save_best_only = True, mode = 'auto')
    earlystopping = EarlyStopping(monitor = 'val_loss', patience = 10, verbose = 1, mode = 'auto')
    history = model.fit(x_train, y_train, nb_epoch = num_epochs, batch_size = batch_size, validation_split = 0.1, show_accuracy = True, callbacks = [checkpointer, earlystopping])
    
    print("Print loading best model weights")
    model.load_weights(os.path.join(model_dir, "DNN_weights.hdf5"))

    print("Evaluating model on test data with best model weights")
    f1 = []
    for test_sentence, ground_truth in zip(text_test, labels_test):
        test_sentence = [[index for index in test_sentence]]
        x = sequences_to_matrix(test_sentence, win_size = context_win_size)
        predicted_labels = model.predict_classes(x)
        true_labels = np.array(ground_truth)
        sentence_f1 = f1_score(true_labels, predicted_labels, pos_label = None, average = 'micro')
        f1.append(sentence_f1)
    print("Mean F1 score over test sentences : %f" %(np.mean(f1)))
        
if __name__ == "__main__":
    run_experiment()

