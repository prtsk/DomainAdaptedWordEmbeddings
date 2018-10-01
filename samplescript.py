import numpy as np
from sklearn.cross_decomposition import CCA
import rccaMod

'''
This sample script illustrates how to obtain domain
adapted word embeddings as described in our ACL submission.
module rccaMod performs KCCA as well as linear CCA.
Note that linear CCA also exists within sklearn. Here
we demonstrate how to use linear CCA from sklearn and 
KCCA from rccaMod. However, rccaMod can be used to perform
linear CCA as well.
'''
######## STEP1: read data ############

### example provided for input as a .txt file

data_file = 'input_text_file1.txt'

####### STEP2: convert data into list of text ######
dobj = open(data_file,'r')
data = dobj.readlines()
for i in list(range(len(data))):
	data[i] = data[i].rstrip()


'''
Input text is usually the vocabulary from data set of interest.
Look at utils code on how to extract unique vocabulary from 
input text, if you do not know how to already do it.
'''

#### STEP3: load precomputed glove/word2vec vocabulary###
glove_vocab = 'glove_vocab.txt'

dobj1 = open(glove_vocab,'r')
gvocab = dobj1.readlines()
for i in list(range(len(gvocab))):
	gvocab[i] = gvocab[i].rstrip()

#### STEP4: take words common to glove vocabulary and vocabulary of interest ###
common_vocab = list(set(data).intersection(set(gvocab)))

### STEP5: extract glove vectors for words in common_vocab and build LSA vectors
###for vocabulary of interest using LSA (Tf-Idf) etc ########
'''
Look at utlis code for building LSA vectors if needed.
'''
### STEP6: assume vocab1 = glove embeddings as numpy matrix and vocab2 = LSA vectors#####
vocab1 = glove_embeddings ### numpy array (v by d1 where v = len(common_vocab), d1= 300 for glove)
vocab2 = Tf-Idf, LSA_vectors #### numpy array (v by d2 where v = len(common_vocab), d2 = dimension of LSA vectors)

### STEP 7: CCA and KCCA steps

## linear CCA ####
k = number of independent components # k <= min(d1,d2)
cca = CCA(n_components=k,max_iter=1000)
cca.fit(vocab1,vocab2)
[train_vec1,train_vec2] = cc.transform(vocab1,vocab2)
#### NOTE: usually if vocabularies we have are too small to partition into test and train sets ####
#### hence we fit CCA on the entire vocabulary and transform all of it into the CCA space ####
#### However, if vocabulary if big enought do cca.train prior to cca.fit

cca_transformed_vectors = 0.5*(train_vec1+train_vec2)
### NOTE: cca_transformed_vectors should have dimension len(common_vocab) by k #####

### non linear/KCCA #####
guess_sig = 1 # guess value of gaussian kernel, trainable parameter
## NOTE: reg is also a trainable parameter #####
cca = rccaMod.CCA(reg=0.01,numCC=k,kernelcca=True,ktype="gaussian",
                  gausigma=guess_sig)
### obtain canonical components #####
cancomps=cca.train([vocab1,vocab2]).comps
train_vec1 = cancomps[0]
train_vec2 = cancomps[1]

kcca_transformed_vectors = 0.5*(train_vec1+train_vec2)
#### perform cca.predict if vocabulary is large enough to partition into train and test splits.

#### STEP8: Use of CCA or KCCA vectors in downstream tasks #####
