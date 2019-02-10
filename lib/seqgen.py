import numpy as np


'''
    Generate sequences of neural actvations, corresponding to words
    * Some neurons may code for nothing
    * A distinct population of neurons codes for each letter. It is equal for each letter
    * All letters over all words are distinct
    * Each letter appears for 1 timestep, immediately followed by next letter
    
    NTot - total number of neurons
    NCoding - number of neurons that code for letters
    WLen_Lst - list of lengths of words. E.g. [2,3] would correspond to words ('ab', 'cde')
    WFreq_Lst - list of frequencies of each word. E.g. [0.1, 0.2] means 1st word will start in ~10% of all time steps, 2nd in ~20%
    TTot - total number of time steps. Some words may be left unfinished if they start late.
'''
def seqGen(NTot, NCoding, WLen_Lst, WFreq_Lst, TTot):
    NWord = len(WLen_Lst)
    NSymb = np.sum(WLen_Lst)
    NNeuPerSymb = int(NCoding / NSymb)
    
    if NCoding % NSymb != 0:
        raise ValueError('Expected integer number of coding neurons per letter. Got', NCoding, '/', NSymb)
        
    # To avoid idx out of bounds, initially make the sequence a little larger
    # to account for the situation when the word starts just before the end
    seq = np.zeros((TTot + np.max(WLen_Lst)-1, NTot))
    
    shiftNeu = 0
    for i in range(NWord):
        # Generate random starting times for each word
        wordStartTimes = np.random.permutation(TTot)[0: int(TTot*WFreq_Lst[i])]
        
        # Activate neurons corresponding to each letter in sequence, for each time the word starts
        for j in range(WLen_Lst[i]):
            neuronIdxsForThisLetter = range(shiftNeu+j*NNeuPerSymb, shiftNeu+(j+1)*NNeuPerSymb)
            sliceIDX = np.ix_(wordStartTimes+j, neuronIdxsForThisLetter)
            seq[sliceIDX] = 1
            
        # Shift to different neurons to code letters of a new word
        shiftNeu += NNeuPerSymb * WLen_Lst[i]
        
    return seq[:TTot, :]  # Truncate back to desired size. Thus, some words may be unfinished