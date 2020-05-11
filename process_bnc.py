import nltk
from nltk.corpus import BNCCorpusReader

# Param: tagged word list from CorpusReader
# Gets all 2-seq prenominal adjectives from a tagged word list
# return as list
def bnc_get_prenominal_adj2(taggedList):
    size = taggedList.__len__()
    result = []
    # iterate through entire taggedlist
    for i in range(size-2):
        word3 = taggedList[i+2]
        # Look 2 words ahead for a Noun
        if  "NN0" in word3[1] or "NN1" in word3[1] or "NN2" in word3[1] or "NP0" in word3[1]:
            word1 = taggedList[i]
            word2 = taggedList[i+1]
            # Check for adjectives
            if "AJ0" in word1[1] and "AJ0" in word2[1]:
                result.append((word1, word2, word3))
    return result

# Use nltk package to parse and load in British National Corpus
bnc_root = "BNC/Texts"
bnc_fileid = r'[A-K]/\w*/\w*\.xml'

bnc = BNCCorpusReader(bnc_root, bnc_fileid)
bnc_tagged = bnc.tagged_words(c5 = True)

# Extract adjectives
bnc_adj2 = get_prenominal_adj2(bnc_tagged)

    # with open('bnc_adjectives2.txt', 'w') as fp:
    #     fp.write('\n'.join('{}/{} {}/{} {}/{}'.format(x[0][0],x[0][1],x[1][0],x[1][1],x[2][0],x[2][1]) for x in bnc_adj2))
# Write to file
with open('adj_bnc.txt', 'w') as fp:
    fp.write('\n'.join('{} {}'.format(x[0][0], x[1][0]) for x in bnc_adj2))