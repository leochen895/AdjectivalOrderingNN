import nltk
from nltk.corpus import BracketParseCorpusReader
from nltk.corpus import BNCCorpusReader

# Get 2 prenominal adjectives from a tagged word list
# return as list
def get_prenominal_adj2(taggedList):
    size = taggedList.__len__()
    result = []
    for i in range(size-2):
        word3 = taggedList[i+2]
        if  "NN" in word3[1] or "NNS" in word3[1] or "NNP" in word3[1] or "NNPS" in word3[1]:
            word1 = taggedList[i]
            word2 = taggedList[i+1]
            if "JJ" in word1[1] and "JJ" in word2[1]:
                result.append((word1, word2, word3))
    return result

def get_prenominal_adj3(taggedList):
    size = taggedList.__len__()
    result = []
    for i in range(size-3):
        word4 = taggedList[i+3]
        if  "NN" == word4[1] or "NNS" == word4[1] or "NNP" == word4[1] or "NNPS" == word4[1]:
            word1 = taggedList[i]
            word2 = taggedList[i+1]
            word3 = taggedList[i+2]
            if "JJ" == word1[1] and "JJ" == word2[1] and "JJ" == word3[1]:
                result.append((word1, word2, word3, word4))
    return result

def get_prenominal_adj4(taggedList):
    size = taggedList.__len__()
    result = []
    for i in range(size-4):
        word5 = taggedList[i+4]
        if "NN" == word5[1] or "NNS" == word5[1] or "NNP" == word5[1] or "NNPS" == word5[1]:
            word1 = taggedList[i]
            word2 = taggedList[i+1]
            word3 = taggedList[i+2]
            word4 = taggedList[i+3]
            if "JJ" == word1[1] and "JJ" == word2[1] and "JJ" == word3[1] and "JJ" == word4[1]:
                result.append((word1, word2, word3, word4, word5))
    return resultM



# Using nltk to parse and load Penn Treebank Corpus
ptb_root = "treebank_3/parsed/mrg/"
ptb_fileid = r".*\.mrg"

ptb = BracketParseCorpusReader(ptb_root, ptb_fileid)

ptb_tagged = ptb.tagged_words()

# Extract 2-seq adjectives from corpus
ptb_adj2 = get_prenominal_adj2(ptb_tagged)
    #ptb_adj3 = get_prenominal_adj3(ptb_tagged)
    #ptb_adj4 = get_prenominal_adj4(ptb_tagged)

# Write to file
with open('adj_ptb.txt', 'w') as fp:
    #fp.write('\n'.join('{}/{} {}/{} {}/{}'.format(x[0][0],x[0][1],x[1][0],x[1][1],x[2][0],x[2][1]) for x in ptb_adj2))
    fp.write('\n'.join('{} {}'.format(x[0][0],x[1][0]) for x in ptb_adj2))   
    #fp.write('\n'.join('{}/{} {}/{} {}/{} {}/{}'.format(x[0][0],x[0][1],x[1][0],x[1][1],x[2][0],x[2][1],x[3][0],x[3][1]) for x in ptb_adj3))
    #fp.write('\n'.join('{}/{} {}/{} {}/{} {}/{} {}/{}'.format(x[0][0],x[0][1],x[1][0],x[1][1],x[2][0],x[2][1],x[3][0],x[3][1],x[4][0],x[4][1]) for x in ptb_adj4))