#!/usr/bin/env python
# coding: utf-8

# In[216]:


import numpy as np
import pandas as pd
import csv
import itertools
from scipy.special import logsumexp
import scipy.optimize as sciop
import time


# In[2]:


PATH = './'


# In[3]:


transition_gradient = pd.read_csv(PATH + 'model/transition-gradient.txt', delimiter=' ', header=None)
transition_params = pd.read_csv(PATH + 'model/transition-params.txt', delimiter=' ', header=None)
feature_gradient = pd.read_csv(PATH + 'model/feature-gradient.txt', delimiter=' ', header=None)
feature_params = pd.read_csv(PATH + 'model/feature-params.txt', delimiter=' ', header=None)


# In[4]:


train_labels = open(PATH + 'data/train_words.txt').read().splitlines()
test_labels = open(PATH + 'data/test_words.txt').read().splitlines()


# In[5]:


AllChars = "etainoshrd"
CharMapping = {'e': 0,
 't': 1,
 'a': 2,
 'i': 3,
 'n': 4,
 'o': 5,
 's': 6,
 'h': 7,
 'r': 8,
 'd': 9
}


# In[6]:


pd.set_option('display.float_format', '{:.4e}'.format)
# pd.reset_option('display.float_format')


# In[9]:


def computePotentials(wordFeats, featureParams):
    """
    Compute single Node log potentials 
    wordFeats: X : (L, 321) 
    featureParams: W_F : (10, 321)
    Returns an array of log potentials for each position : (L, 321)
    """
    phis = []
    for letter_id in range(len(wordFeats)):
        phis.append(np.sum(np.multiply(featureParams, wordFeats[letter_id, :]), axis=1))
    return np.array(phis).T


# In[11]:


def computeEnergy(wordFeats, wordLabel, logPotentials, transitionParams):
    """
    Compute Energy of a given word, E = - sum log(potential) + W_T
    wordFeats: X : (L, 321)
    Returns computed energy
    """
    wordLabel_CharIdx = [CharMapping[x] for x in wordLabel]
    sum_phis = np.sum(logPotentials[wordLabel_CharIdx, range(0, len(wordFeats))])
    
    arr = [(x, y) for (x, y) in zip(wordLabel_CharIdx[:-1], wordLabel_CharIdx[1:])]
    transition_param_sum = 0
    for elem in arr:
        transition_param_sum += transitionParams.iloc[elem]
    return transition_param_sum + sum_phis
    


# In[84]:


####### Nodes are 1 indexed
def messageDict(word, featParams, transParams, wordPhis=None):
    """
    Returns Messages dictionary for every pair of adjacent nodes
    word: X : (L, 321)
    featParams: W_F : (10, 321)
    transParams: W_T : (10, 10)
    wordPhis: Precomputed log potentials of the word
    
    Every element of dict has shape (10,)
    """
    ########### Compute Word Potentials #########
    if wordPhis is None:
        wordPhis = computePotentials(word, featParams)
        
    logM = dict()
    logM[len(word) + 1, len(word)] = np.zeros(len(AllChars))
    logM[0, 1] = np.zeros(len(AllChars))
    
    #### Backward Messages ####
    for fromNode in range(len(word), 1, -1):
        logM[fromNode, fromNode - 1] = logsumexp(wordPhis[:, fromNode-1] +                               logM[fromNode + 1, fromNode] +                               transParams.values, axis = 1)    
        
    #### Forward Messages ####
    for fromNode in range(1, len(word)):
        logM[fromNode, fromNode + 1] = logsumexp(wordPhis[:, fromNode-1] +                               logM[fromNode - 1, fromNode] +                                 transParams.values, axis=1)    
    return logM

def singleVariableMarginal(node, word, featParams, transParams, wordPhis=None, logPartition=None):
    """
    Returns log Marginals
    """
    if wordPhis is None:
        wordPhis = computePotentials(word, featParams)
    
    _messageDict = messageDict(word, featParams, transParams, wordPhis)
    if (node > 0):
        m_left = _messageDict[node-1, node]
    if (node < len(word)+1):
        m_right = _messageDict[node+1, node]
    
    total_incoming_message = m_left + m_right
    
    ############
    if (logPartition is None):
        logPartition = logsumexp(wordPhis[:, node-1] + total_incoming_message)
        
    logMarginalProbs = wordPhis[:, node-1] + total_incoming_message - logPartition
    return np.exp(logMarginalProbs)

def pairwiseMarginal(nodeL, nodeR, word, featParams, transParams, logPartition=None):
    """ 
    PairWise Marginals
    """
    wordPhis = computePotentials(word, featParams)
    _messageDict = messageDict(word, featParams, transParams, wordPhis)
    
    if (logPartition is None):
        logPartition = logsumexp(wordPhis[:, 0] + _messageDict[2, 1])
        
    chars_to_consider = AllChars
    pairwise_marginals = np.zeros( (len(chars_to_consider), len(chars_to_consider)) )
    
    mLeft = 0
    mRight = 0
    if (nodeL >= 1):
        mLeft = _messageDict[nodeL - 1, nodeL]
    if (nodeR <= len(word)):
        mRight = _messageDict[nodeR + 1, nodeR]
        
    logSingleMarginalsL = wordPhis[:, nodeL-1] + mLeft
    logSingleMarginalsL = np.stack( [logSingleMarginalsL]*len(AllChars) ).T
    logSingleMarginalsR = wordPhis[:, nodeR-1] + mRight
    logSingleMarginalsR = np.stack( [logSingleMarginalsR]*len(AllChars) )
    
    SingleMarginals = logSingleMarginalsL + logSingleMarginalsR
    jointMarginal = SingleMarginals + transParams - logPartition

    return np.exp(jointMarginal)


# # Question 1:
# 

# In[330]:


####### Compute Log Likelihood ######
def loglikelihood(W, dataSize, AllWords, AllLabels):
    """
    Evaluated the average log likelihood (lll) of AllWords
    W: value of weights to comptue lll at
    dataSize: Consider 0->dataSize cases from AllWords to compute lll
    AllWords: Array containing all training word features
    AllLabels: True labels for each characeter of each training word
    
    Returns average loglikelihood evaluated at W (float)
    """
    featParams, transParams = W[:len(AllChars)*featSize], W[len(AllChars)*featSize:]
    featParams = pd.DataFrame(np.reshape(featParams, (len(AllChars), featSize)))
    transParams = pd.DataFrame(np.reshape(transParams, (len(AllChars), len(AllChars))))
    
    loglikelihood = 0
    for WordIdx in range(0, dataSize):
        Word = AllWords[WordIdx]
        Word_label = AllLabels[WordIdx]
        Word_labelIdxs = [CharMapping[x] for x in Word_label]

        WordPhis = computePotentials(Word, featParams)
        WordlogM = messageDict(Word, featParams, transParams)
        marginal = WordPhis[:, 0] + WordlogM[(2, 1)]
        logZ = logsumexp(marginal)

        WordEnergy = computeEnergy(Word, Word_label, WordPhis, transParams)
        loglikelihood += WordEnergy - logZ

    print("Log likelihood for first {} train data points:".format(dataSize), loglikelihood/dataSize )
    return loglikelihood/dataSize


# In[314]:


avglll = loglikelihood(np.append(feature_params.values.flatten(), transition_params.values.flatten()),                        50, AlltrainWords)


# In[331]:


def _computeGradWF(word, wordLabel, singleMarginals):
     """
    Compute the feature parameter gradient for a single word
    word: image features of the word
    wordLabel: true ground truth label
    singleMarginals: Marginal distribution of each node in CRF
    
    Returns 10 x 321 np array that contains the gradient of lll of 'word' w.r.t. W^T evaluated at current W
    """
    wordLabelList = np.array([x for x in wordLabel])
    mask = [np.where(wordLabelList == c)[0] for c in CharMapping] # 10 x L(word)
    selectedX = [np.sum(word[m, :], axis=0) for m in mask] # 10 x _ x 321

    secondTerm = []
    for c_id, c in enumerate(AllChars):
        probbyc = np.expand_dims(singleMarginals[:, c_id], axis=1)
        secondTerm.append(np.sum(word*probbyc, axis=0))
    secondTerm = np.array(secondTerm)
    grad = selectedX - secondTerm
    return grad

def computeGradF(W, dataSize, AlltrainingWords, trainingPotentials, Alltraininglabels):
    """
    Compute the feature parameter gradient for words in AlltrainingWords
    W: current value of weights
    dataSize: 0->dataSize words will be used from AlltrainingWords for computation of gradient
    AlltrainingWords: Array containing all training word features
    trainingPotentials: Precomputed node Potentials of every training word
    Alltraininglabels: True labels for each characeter of each training word
    
    Returns 10x321 array containing gradient of avg. lll w.r.t. W^F evaluated at W
    """
    wF, wT = W[:len(AllChars)*featSize], W[len(AllChars)*featSize:]
    wF = pd.DataFrame(np.reshape(wF, (len(AllChars), featSize)))
    wT = pd.DataFrame(np.reshape(wT, (len(AllChars), len(AllChars))))
    grad = 0
    for trainWordIdx in range(dataSize):
        trainWord = AlltrainingWords[trainWordIdx]
        trainWord_label = Alltraininglabels[trainWordIdx]
        trainWord_labelIdxs = [CharMapping[x] for x in trainWord_label]
        trainWordPhis = trainingPotentials[trainWordIdx] # computePotentials(trainWord, wF)

        singleVarMargs = []
        for node in range(1, len(trainWord)+1):
            sVM = singleVariableMarginal(node, trainWord, wF, wT, trainWordPhis)
            singleVarMargs.append(sVM)
        grad += _computeGradWF(trainWord, trainWord_label, np.array(singleVarMargs))
    return grad/dataSize

def _computeGradWT(word, wordLabel, _pairwiseMarginals):
    """
    Compute the transition parameter gradient for a single word
    word: image features of the word
    wordLabel: true ground truth label
    _pairwiseMarginals: Pairwise Marginal distribution of each pair of adjacent nodes in CRF
    
    Returns 10 x 10 np array that contains the gradient of lll of 'word' w.r.t. W^T evaluated at current W
    """
    gradwt = np.zeros((len(AllChars), len(AllChars)))
    for c in CharMapping:
        for cprime in CharMapping:
            firstTerm = 0
            secondTerm = 0
            for letter_id in range(0, len(word)-1):
                if ((wordLabel[letter_id] == c) and (wordLabel[letter_id + 1] == cprime)):
                    firstTerm += 1
                secondTerm += _pairwiseMarginals[letter_id][CharMapping[c]][CharMapping[cprime]]
            gradwt[CharMapping[c], CharMapping[cprime]] = (firstTerm - secondTerm)
    return gradwt

def computeGradT(W, dataSize, AlltrainingWords, trainingPotentials, Alltraininglabels):
    """
    Compute Gradient for Transition parameters
    W: Current values of weights to evaluate the gradient at
    dataSize: size of the training data to be used
    AlltrainingWords: Array containing all training word features
    trainingPotentials: Precomputed node Potentials of every training word
    Alltraininglabels: True labels for each characeter of each training word
    
    Returns 10 x 10 np array containing the gradient of lll w.r.t. W^T evaluated at current W
    """
    wF, wT = W[:len(AllChars)*featSize], W[len(AllChars)*featSize:]
    wF = pd.DataFrame(np.reshape(wF, (len(AllChars), featSize)))
    wT = pd.DataFrame(np.reshape(wT, (len(AllChars), len(AllChars))))
    gradWT = 0
    for trainWordIdx in range(dataSize):
        trainWord = AlltrainingWords[trainWordIdx]
        trainWord_label = Alltraininglabels[trainWordIdx]
        trainWord_labelIdxs = [CharMapping[x] for x in trainWord_label]
        trainWordPhis = trainingPotentials[trainWordIdx] # computePotentials(trainWord, wF)

        pairwiseMargs = []
        for node in range(1, len(trainWord)):
            pM = pairwiseMarginal(node, node+1, trainWord, wF, wT)
            pairwiseMargs.append(pM.values)
        gradWT += _computeGradWT(trainWord, trainWord_label, np.array(pairwiseMargs))
    return gradWT/dataSize


# In[332]:


def gradlll(W, dataSize, AlltrainingWords, Alltraininglabels):
    """ Compute the gradient of log likelihood at wF and wT """
    wF, wT = W[:len(AllChars)*featSize], W[len(AllChars)*featSize:]
    wF = pd.DataFrame(np.reshape(wF, (len(AllChars), featSize)))
    wT = pd.DataFrame(np.reshape(wT, (len(AllChars), len(AllChars))))
    
    trainingPotentials = []
    for dataIter in range(dataSize):
        thisWordPotential = computePotentials(AlltrainingWords[dataIter], wF)
        trainingPotentials.append(thisWordPotential)
    
    gradWF = computeGradF(W, dataSize, AlltrainingWords, trainingPotentials, Alltraininglabels)
    gradWT = computeGradT(W, dataSize, AlltrainingWords, trainingPotentials, Alltraininglabels)

    grads = np.concatenate( (gradWF.flatten(), gradWT.flatten()) )
    return grads


# In[296]:


######## Verify the computation of gradients is correct
W0 = np.append(feature_params.values.flatten(), (transition_params.values.flatten()))
_grads_ll = gradlll(W0, 50, AlltrainWords)
_wF, _wT = _grads_ll[:len(AllChars)*featSize], _grads_ll[ len(AllChars)*featSize: ]
_wF = pd.DataFrame(np.reshape(_wF, (len(AllChars), featSize)))
_wT = pd.DataFrame(np.reshape(_wT, (len(AllChars), len(AllChars))))

# print(_wT)
# print(transition_gradient)


# In[333]:


def nll(W, dataSize, AlltrainingWords, Alltraininglabels):
    return -1*loglikelihood(W, dataSize, AlltrainingWords, Alltraininglabels)

def gradnll(W, dataSize, AlltrainingWords, Alltraininglabels):
    return -1*gradlll(W, dataSize, AlltrainingWords, Alltraininglabels)


# In[275]:


####### Load all training words #####
AlltrainWords = []
for trainWordIdx in range(0, 400):
    trainWord = pd.read_csv(PATH + 'data/train_img{}.txt'.format(trainWordIdx+1), header=None, delimiter=" ")
    trainWord = trainWord.values
    AlltrainWords.append(trainWord)


# In[305]:


####### Load all testing words #####
AlltestWords = []
for testWordIdx in range(0, 200):
    testWord = pd.read_csv(PATH + 'data/test_img{}.txt'.format(testWordIdx+1), header=None, delimiter=" ")
    testWord = testWord.values
    AlltestWords.append(testWord)


# In[236]:


def getAccuracy(W):
    wF, wT = W[:len(AllChars)*featSize], W[len(AllChars)*featSize:]
    wF = pd.DataFrame(np.reshape(wF, (len(AllChars), featSize)))
    wT = pd.DataFrame(np.reshape(wT, (len(AllChars), len(AllChars))))
    
    #### Predictions on test set
    preds = []
    for testWordIdx in range(1, 201):
        testWord = pd.read_csv(PATH + 'data/test_img{}.txt'.format(testWordIdx), header=None, delimiter=" ")
        testWord = testWord.values

        testWordPhis = computePotentials(testWord, wF)
        singleVarMarginals = []
        for node in range(1, len(testWord)+1):
            singleVarMarginals.append(singleVariableMarginal(node, testWord, wF, wT,                                                              testWordPhis))
        singleVarMarginals = np.array(singleVarMarginals)
        preds.append(''.join([AllChars[x] for x in np.argmax(singleVarMarginals, axis=1)]))
    
    #### Character Level accuracy ########
    tp = 0
    total = 0
    for pred_id, pred in enumerate(preds):
        testWordLabel = np.array([x for x in test_labels[pred_id]])
        pred = np.array([x for x in pred])
        tp += len(np.where(testWordLabel == pred)[0])
        total += len(testWordLabel)
    
    print("Character Level Test-Accuracy:", tp/total*100, "%")
    return tp/total*100


# In[373]:


_training_times_bfgs = {}
_weights_bfgs = {}
_accuracies_bfgs = {}
_testLL_bfgs = {}
_trainLL_bfgs = {}


# In[374]:


for _data_size in range(50, 401, 50):
    ### train with given data ###
    x0 = np.concatenate( (np.zeros((len(AllChars)*featSize)), np.zeros((len(AllChars)*len(AllChars)))) )
    startTime = time.time()
    res = sciop.minimize(fun=nll, x0=x0, args=(_data_size, AlltrainWords, train_labels), jac=gradnll,                              method='BFGS', options={'disp': True, 'maxiter': 40})
    endTime = time.time()
    training_time = endTime - startTime
    
    print("Training Time:", training_time)
    
    _training_times_bfgs[_data_size] = training_time
    
    _weights_bfgs[_data_size] = res.x
    
    _accuracies_bfgs[_data_size] = getAccuracy(res.x)
    
    _testLL_bfgs[_data_size] = loglikelihood( _weights_bfgs[_data_size], 200, AlltestWords, test_labels )
    
    _trainLL_bfgs[_data_size] = loglikelihood(_weights_bfgs[_data_size], _data_size, AlltrainWords, train_labels)


# In[398]:


import matplotlib.pyplot as plt
fig = plt.figure()

lists = sorted(_training_times_bfgs.items())
x, y = zip(*lists)
# ax1 = fig.add_subplot(311)
plt.plot(x, y)
plt.grid(True)
plt.xlabel('Training Set Size')
plt.ylabel('Training Time (s)')
for xx, yy in zip(x, y):                                
    plt.annotate('%.2f' % yy, xy=(xx, yy), xytext= (xx, yy), textcoords='data')
plt.savefig('trainingTime.jpg', bbox_inches='tight')
plt.show()

lists = sorted(_accuracies_bfgs.items())
x, y = zip(*lists)
plt.xlabel('Training Set Size')
plt.ylabel('Test Accuracy (%)')
plt.plot(x, y)
plt.grid(True)
for xx, yy in zip(x, y):                                    
    plt.annotate('%.2f' % yy, xy=(xx, yy), xytext= (xx, yy), textcoords='data')
plt.savefig('Accuracies.jpg', bbox_inches='tight')
plt.show()

lists = sorted(_testLL_bfgs.items())
x, y = zip(*lists)
plt.xlabel('Training Set Size')
plt.ylabel('avg. test LogLikelihood')
plt.plot(x, y)
plt.grid(True)
for xx, yy in zip(x, y):                                      
    plt.annotate('%.2f' % yy, xy=(xx, yy), xytext= (xx, yy), textcoords='data') 
plt.savefig('testLogLikelihood.jpg', bbox_inches='tight')
plt.show()

lists = sorted(_trainLL_bfgs.items())
x, y = zip(*lists)
plt.xlabel('Training Set Size')
plt.ylabel('avg. train LogLikelihood')
plt.plot(x, y)
plt.grid(True)
for xx, yy in zip(x, y):                                       
    plt.annotate('%.2f' % yy, xy=(xx, yy), xytext= (xx, yy), textcoords='data')
plt.savefig('trainLogLikelihood.jpg', bbox_inches='tight')
plt.show()


# In[375]:


_training_times_lbfgsb
_weights_bfgs_lbfgsb 
_accuracies_bfgs_lbfgsb 
_testLL_bfgs_lbfgsb 
_trainLL_bfgs_lbfgsb 


# ## Appendix
# ### Compare the runtime of likelihood computation with and without preloaded TrainingData

# In[247]:


_start = time.time()
for i in range(10):
    ll = 0
    for trainWordIdx in range(0, dataSize):
        trainWord = pd.read_csv(PATH + 'data/train_img{}.txt'.format(trainWordIdx+1), header=None, delimiter=" ")
        trainWord = trainWord.values
        trainWord_label = train_labels[trainWordIdx]
        trainWord_labelIdxs = [CharMapping[x] for x in trainWord_label]

        trainWordPhis = computePotentials(trainWord, feature_params)
        trainWordlogM = messageDict(trainWord, feature_params, transition_params)
        marginal = trainWordPhis[:, 0] + trainWordlogM[(2, 1)]
        logZ = logsumexp(marginal)

        trainWordEnergy = computeEnergy(trainWord, trainWord_label, trainWordPhis, transition_params)
        ll += trainWordEnergy - logZ

_end = time.time()
print(_end - _start)


# In[248]:


_start = time.time()
AlltrainWords = []
for trainWordIdx in range(0, dataSize):
    trainWord = pd.read_csv(PATH + 'data/train_img{}.txt'.format(trainWordIdx+1), header=None, delimiter=" ")
    trainWord = trainWord.values
    AlltrainWords.append(trainWord)
    
for i in range(10):
    ll = 0
    for trainWordIdx in range(0, dataSize):
        trainWord = AlltrainWords[trainWordIdx]
        trainWord_label = train_labels[trainWordIdx]

        trainWordPhis = computePotentials(trainWord, feature_params)
        trainWordlogM = messageDict(trainWord, feature_params, transition_params)
        marginal = trainWordPhis[:, 0] + trainWordlogM[(2, 1)]
        logZ = logsumexp(marginal)

        trainWordEnergy = computeEnergy(trainWord, trainWord_label, trainWordPhis, transition_params)
        ll += trainWordEnergy - logZ

_end = time.time()
print(_end - _start)

