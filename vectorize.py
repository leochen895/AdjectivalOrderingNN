#load glove
import bcolz
import pickle
import torch
import numpy as np

# Glove loader as bolz from link 
# Allows easy lookup from word as str -> word as vector
# Thank you Martin; Source: https://medium.com/@martinpella/how-to-use-pre-trained-word-embeddings-in-pytorch-71ca59249f76
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

# My code
# First load in Adjective pairs
adj = np.loadtxt('adj_all.txt', dtype=str).tolist()
npairs = len(adj)

#Result1 = word embedding lookups for each adj pair
#Result0 = word embedding lookup for each adj pair FLIPPED
result1 = np.empty((npairs, 100))
result0 = np.empty((npairs, 100))

idx = 0
for pair in adj:
	#get adjective
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
    row1 = np.concatenate((v1,v2))
    row0 = np.concatenate((v2,v1))
    # Put into Result np arrays
    for i in range(100):
        np.put(result1, idx, row1[i])
        np.put(result0, idx, row0[i])
        idx += 1
# Save to file
np.save('adj_emb1', result1)
np.save('adj_emb0', result0)