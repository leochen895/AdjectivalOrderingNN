import time
import random

import torch
import bcolz
import pickle
import numpy as np
from torch import nn
from torch import optim as optim
from torch.utils.data import Dataset, DataLoader

from model import AdjOrderModel
from dataset import AdjDataset
from tools import EarlyStopping

#Uncomment if getting environment errors
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

def train(model: AdjOrderModel, training_data: Dataset, testing_data: Dataset, batch_size:int):
    
    early_stopping = EarlyStopping(patience=3, verbose=False, filename='checkpoint.pt')
    train_loader = DataLoader(dataset = training_data, batch_size=batch_size, shuffle=True, num_workers = 2)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    epochs = 50

    epoch_loss, total_batches = 0., 0.
    for i in range(epochs):
        start_time = time.time()
        for batch_idx, (x, y) in enumerate(train_loader):
        	total_batches += 1
        	model.zero_grad()
        	y_pred = model.forward(x)  
        	loss = criterion(y_pred, y)
        	epoch_loss += loss
        	loss.backward()
        	optimizer.step()

        valid_accuracy, valid_loss = eval(model, testing_data)
        duration = (time.time() - start_time)
        print("Epoch %d:" % i, end=" ")
        print("loss per batch = %.2f" % (epoch_loss / total_batches), end=", ")
        print("val loss = %.4f, val acc = %.4f" % (valid_loss, valid_accuracy), end=" ")
        print("(%.3f sec)" % duration)
        early_stopping(valid_loss, model)
        if early_stopping.early_stop:
            print("Early stopping, reloading checkpoint model")
            model.load_state_dict(torch.load('checkpoint.pt'))
            break

     
        

def eval(model: AdjOrderModel, testing_data: Dataset):
	test_loader = DataLoader(dataset = testing_data, batch_size=256, num_workers = 2)
	criterion = nn.BCELoss()
	batch_count, correct_pred, total_pred, total_loss = 0, 0, 0, 0
	for batch_idx, (x, y) in enumerate(test_loader):

		yhat = model(x)
		batch_size = len(yhat)
		for i in range(batch_size):
			total_pred += 1
			pred = yhat[i][0]
			targ = y[i][0]
			if pred > 0.5 and targ == 1:
				correct_pred += 1
			if pred <= 0.5 and targ == 0:
				correct_pred += 1
		loss = criterion(yhat, y)
		total_loss += loss
		batch_count += 1

	loss = total_loss/batch_count
	accuracy = correct_pred/total_pred

	return accuracy, loss

def tests(model: AdjOrderModel, adj_pairs):
	print("Preparing for personal tests...")
	print("Loading GloVe dataset...")
	print("This will take a LONG time, please be patient:)")
	# Loads in GloVe dataset to allow lookup each test adjective->vector	
	words = []
	idx = 0
	word2idx = {}
	vectors = bcolz.carray(np.zeros(1), rootdir=f'glove/6B.50.dat', mode='w')

	with open(f'glove/glove.6B.50d.txt', 'rb') as f:
	    for l in f:
	        line = l.decode().split()
	        word = line[0]
	        words.append(word)
	        word2idx[word] = idx
	        idx += 1
	        vect = np.array(line[1:]).astype(np.float)
	        vectors.append(vect)

	vectors = bcolz.carray(vectors[1:].reshape((400000, 50)), rootdir=f'glove/6B.50.dat', mode='w')
	vectors.flush()
	pickle.dump(words, open(f'glove/6B.50_words.pkl', 'wb'))
	pickle.dump(word2idx, open(f'glove/6B.50_idx.pkl', 'wb'))

	vectors = bcolz.open(f'glove/6B.50.dat')[:]
	words = pickle.load(open(f'glove/6B.50_words.pkl', 'rb'))
	word2idx = pickle.load(open(f'glove/6B.50_idx.pkl', 'rb'))

	glove = {w: vectors[word2idx[w]] for w in words}

	print("Glove loaded:")
	print("Predictions:")

	# Load in adjective pairs
	npairs = len(adj_pairs)
	idx = 0
	for pair in adj_pairs:
		#Get adjectives
	    s1 = pair[0].lower()
	    s2 = pair[1].lower()
	    try: 
	    	#Lookup word embedding
	        v1 = glove[s1]
	        v2 = glove[s2]
	    except KeyError:
	    	#Generate random word embedding if no match
	        v1 = np.random.random(50).tolist()
	        v2 = np.random.random(50).tolist()  
	    # Concatenate word embedding (so that we wont end up with a 3D tensor)
	    row = np.concatenate((v1,v2))
	    inp = np.empty((0,100))
	    inp = torch.from_numpy(np.vstack((inp, row))).float()
	    prediction = model(inp)

	    print("Input: {%s, %s}, Output: %f" % (s1, s2, prediction))



if __name__ == "__main__":
	#Set up model
	classifier = AdjOrderModel(125,75)
	print("Setting up data...")

	# Get data
	dataset = AdjDataset()

	# Splitting testing and training data
	n_total = dataset.__len__()
	n_train = int( 0.8 * n_total )
	n_test = n_total - n_train
	train_dataset, test_dataset = torch.utils.data.random_split(dataset, [n_train, n_test])
	print("Beginning training:")
	# Train
	train(classifier, train_dataset, test_dataset, 256)
	print("Finished training.")

	# User tests
	myadjpairs = [("small", "red"), ("red", "small"), ("big", "green"), ("green", "big"), ("lovely", "blue"), ("blue", "lovely"), 
	("dark", "gold"), ("gold", "dark"), ("interesting", "old"), ("old", "interesting"), ("tall", "funny"), ("funny", "tall"),
	("interesting", "tiny"), ("tiny", "interesting"), ("beautiful", "new"), ("new", "beatiful"),
	("good", "sharp"), ("sharp", "good"), ("black", "woolen"), ("woolen", "black"),
	("cozy", "woolen"), ("woolen", "cozy"), ("stinky", "woolen"), ("woolen", "stinky"),
	("gnomic", "tumescent"), ("tumescent", "gnomic"), ("inelegant", "tumescent"), ("tumescent", "inelegant"), ("inelegant", "woolen"), ("woolen", "inelegant"),
	("stinky", "tumescent"), ("tumescent", "stinky"),
	("french", "rocking"), ("french", "wooden"), ("wooden", "french"), ("french", "metallic"), ("metallic", "french"),
	("french", "stony"), ("stony", "french"), ("big","french"), ("french","big"), ("young", "french"), ("french", "young"),
	("pale", "french"), ("french", "pale"), ("french","woolen"), ("woolen", "french"), ("lovely","intelligent"), ("intelligent", "lovely"), 
	("old", "medieval"), ("medieval", "old"), ("big", "bad"), ("bad", "big"), ("stinky", "smelly"), ("smelly", "stinky")]
	tests(classifier, myadjpairs)



