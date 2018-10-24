import numpy as np
import numpy.matlib
import copy
from time import time

from sklearn.preprocessing import normalize
#from scipy.sparse import csr_matrix, hstack, vstack
import scipy.sparse

import matrices_lib


'''
Single Population Discrete Neural Network

Parameters:
    N_EXC            : Number of excitatory neurons in the network
    N_INH            : Number of inhibitory neurons in the network

    T_RANGE_EXC      : Initial range of thresholds for excitatory neurons
    T_RANGE_INH      : Initial range of thresholds for inhibitory neurons
    W_MAG_EXC        : Initial range of excitatory weights for uniform random distribution
    W_MAG_INH        : Initial range of inhibitory weights for uniform random distribution

    P_CONN           : Probability of [EXC_EXC, EXC_INH, INH_EXC, INH_INH] connection

    SP_RATE          : Rate of synaptic plasticity for [EXC_EXC, EXC_INH, INH_EXC, INH_INH] connection
    IP_RATE_EXC      : Rate of intrinsic plasticity for excitatory neurons
    IP_RATE_INH      : Rate of intrinsic plasticity for inhibitory neurons
    V_EQ_EXC         : Equillibrium membrane potential for excitatory neurons
    V_EQ_INH         : Equillibrium membrane potential for inhibitory neurons

    WITH_SYNAPSES    : Whether the population has synapses at all
    WITH_INIT_NORM_INP  : Whether to normalize input weights to each neuron
    
TODO:
    [ ] Accelerate code: Write our own sparse matrix class, just row, col and val
      [ ] Sort indices in the beginning
      [ ] For sum add values
      [ ] For clip clip values
      [ ] For mult try loop or sth like rez[rowidx] += mult(v[colidx], val) 
    [ ] Implement actual sparsity
       [+] Sparse outer product
       [+] Sparse clipping
       [ ] Sparse normalization
    [ ] Implement adequate alternative to weight clipping during Hebbian update
    [ ] Implement different types of plasticity:
        [] Synaptic
            [] STD (Synaptic Depression)
            [] log-STDP
            [] SYN-HP (Homeostatic Plasticity)
        [] Somatic
            [] CREB-IP (Excitability-enhancement)
            [] SOMA-HP (Homeostatic plasticity)  Rename current IP to HP
    [ ] Introduce adequate timescales for each plasticity
    [ ] Consider improving performance using stochastic update (once we understand it)
'''


class SinglePopulationDiscrete:
    def __init__(self, p):
        self.p = p   # Properties, unchanged during simulation
        self.v = {}  # Variables, changed during simulation

        ##################################
        ## Populations
        ##################################
        
        self.p['POP_KEYS'] = ['EXC', 'INH']
        
        for rowPopKey in self.p['POP_KEYS']:
            ##################################
            ## Neurons
            ##################################
            N_ROW = p['N_' + rowPopKey]

            # Initial synaptic potential
            self.v['V_' + rowPopKey] = np.zeros(N_ROW)

            # Threshold values for each neuron
            TRangeRow = p['T_RANGE_' + rowPopKey]
            self.v['T_' + rowPopKey] = np.random.uniform(TRangeRow[0], TRangeRow[1], N_ROW)

            ##################################
            ##synaptic connections matrix
            ##################################

            if p['WITH_SYNAPSES']:
                for colPopKey in self.p['POP_KEYS']:
                    N_COL = p['N_' + colPopKey]
                    rckey = rowPopKey + '_' + colPopKey
                    WKey = 'W_' + rckey
                    WIdxKey = 'W_IDX_' + rckey
                    
                    # Create weight matrix
                    # Note the swap of N_ROW and N_COL, because in order for the matrix to be named FROM_TO,
                    # its rows must be TO, and its cols must be FROM
                    WMagRow = p['W_MAG_' + rowPopKey]
#                     self.v[WKey], self.p[WIdxKey] = matrices_lib.spRandUMatBase((N_COL, N_ROW), WRangeRow, p['P_CONN'][rckey])
                    self.v[WKey] = WMagRow * matrices_lib.spRandUMat((N_COL, N_ROW), [0, 1], p['P_CONN'][rckey])
                    
                    # Excract its sparse connectivity indices.
                    # Note that we do not intend to change neuron connectivity during simulation,
                    # so connectivity is a parameter, not a variable
                    WThisCoo = self.v[WKey].tocoo()
                    self.p[WIdxKey] = [np.array(WThisCoo.row), np.array(WThisCoo.col)]
                    
                    # Normalize the matrix wrt rows (all input weights to a given )
                    if p['WITH_INIT_NORM_INP']:
                        self.v[WKey] = normalize(self.v[WKey], norm='l1', axis=1) * abs(WMagRow)

            # TODO: Problem is that normalization is over two different matrices. Would need to take norm wrt given axis, add sqrt(EE^2 + EI^2), then normalize each row manually
            # synaptic normalization - normalize all inputs to each neuron
            # Note: sklearn normalize is smart enough not to normalize zero-rows in case some neurons have no input
            #if p['WITH_INIT_NORM_INP']:
            #    self.W[self.idxE, :] = normalize(self.W[self.idxE, :], axis=1, norm='l1') * p['W_NORM_EXC']
            #    self.W[self.idxI, :] = normalize(self.W[self.idxI, :], axis=1, norm='l1') * p['W_NORM_INH']
        
        
        ##################################
        ## Statistics
        ##################################
        
        # Storage for activities
        self.V_Lst = []
        self.statsDictInit = {key : [] for key in self.v.keys()}
        self.statsDictInit.update({'D'+key : [] for key in self.v.keys()})
        

    # Calculate the next membrane potential
    def nextState(self, rowPopKey, VI, param_sim, vnew):
        N_ROW = self.p['N_' + rowPopKey]
        T_ROW = self.v['T_' + rowPopKey]

        # Add external input
        # NOTE: Discrete model does not reuse old state of V in the same neuron
        V_PRE = VI.copy()

        # Add noise if it is requested
        noiseKey = 'SOMA_NOISE_' + rowPopKey
        if noiseKey in param_sim.keys():
            noiseParam = param_sim[noiseKey]
            V_PRE += np.random.normal(noiseParam[0], noiseParam[1], N_ROW)

        # Add synaptic input from other neurons (colPop) to these neurons (rowPop)
        if self.p['WITH_SYNAPSES']:
            for colPopKey in self.p['POP_KEYS']:
                V_COL = self.v['V_' + colPopKey]
                W = self.v['W_' + colPopKey + '_' + rowPopKey]
                V_PRE += W.dot(V_COL)

        # Compute value after thresholding
        vnew['V_' + rowPopKey] = (V_PRE > T_ROW).astype(int)
    
    
    # Calculate change in threshold due to intrinsic plasticity
    def nextThresholds(self, rowPopKey, vnew):        
        V_ROW = self.v['V_' + rowPopKey]
        T_ROW = self.v['T_' + rowPopKey]
        IP_RATE_ROW = self.p['IP_RATE_' + rowPopKey]
        V_EQ_ROW = self.p['V_EQ_' + rowPopKey]
        T_RANGE_ROW = self.p['T_RANGE_' + rowPopKey]

        # Update threshold
        vnew['T_' + rowPopKey] = T_ROW + IP_RATE_ROW * (V_ROW - V_EQ_ROW)

        # Clip all thresholds to allowed values
        matrices_lib.clipMat(vnew['T_' + rowPopKey], T_RANGE_ROW)
        
        
    # Calculate change in weight due to synaptic plasticity due to additive STDP
    def nextWeights(self, rowPopKey, colPopKey, vnew):
        
        # Precompute keys
        WKey = 'W_' + rowPopKey + '_' + colPopKey
        WIdxKey = 'W_IDX_' + rowPopKey + '_' + colPopKey
        SP_RATE_THIS = self.p['SP_RATE'][rowPopKey + '_' + colPopKey]

        # Shorthand for old weights
        WThis = self.v[WKey]

        if SP_RATE_THIS != 0:
            # Find the row and col indices of sparse entries in the connectivity matrix
            rowWIdx = self.p[WIdxKey][0]
            colWIdx = self.p[WIdxKey][1]
            
            # Find new and old values of membrane potential corresponding to those sparse entries
            V0row = self.v['V_' + rowPopKey][rowWIdx]
            V0col = self.v['V_' + colPopKey][colWIdx]
            V1row = vnew['V_' + rowPopKey][rowWIdx]
            V1col = vnew['V_' + colPopKey][colWIdx]

            # Compute correlation matrix between previous and new time step activities
            # Hebb matrix is the difference between causal and anti-causal correlations in activities
            HebbVal = SP_RATE_THIS * (np.multiply(V0row, V1col) - np.multiply(V1row, V0col))
            #HebbSpMat = scipy.sparse.csr_matrix((HebbVal, (rowWIdx, colWIdx)), shape=WThis.shape, dtype=WThis.dtype)
            vnew[WKey] = WThis.copy()
            WThis.data += HebbVal
#             vnew[WKey] = WThis + HebbSpMat
#             vnew[WKey] = WThis + HebbVal
            
            
            # Clip all excitatory weights to their allowed ranges
            # matrices_lib.clipMat(vnew[WKey], self.p['W_RANGE_' + rowPopKey])
        else:
            vnew[WKey] = WThis

        # TODO: Problem is that normalization is over two different matrices. Would need to take norm wrt given axis, add sqrt(EE^2 + EI^2), then normalize each row manually
        # synaptic normalization - normalize all inputs to each neuron
        # Note: sklearn normalize is smart enough not to normalize zero-rows in case some neurons have no input
        #if p['WITH_INIT_NORM_INP']:
        #    WNew[self.idxE, :] = normalize(WNew[self.idxE, :], axis=1, norm='l1') * self.W_NORM_EXC
        #    WNew[self.idxI, :] = normalize(WNew[self.idxI, :], axis=1, norm='l1') * self.W_NORM_INH
        
    
    # Compute magnitudes and change-magnitudes for all variables of the simulation
    def updateStats(self, iStep, vnew):
        for key in self.v.keys():
            # Record this variable and its change, if the variable is being updated between timesteps
            if key in vnew.keys():
                if type(self.v[key]) is numpy.ndarray:
                    self.statsDict[key][iStep] = np.linalg.norm(self.v[key])
                    self.statsDict['D'+key][iStep] = np.linalg.norm(vnew[key] - self.v[key])
                elif type(self.v[key]) is scipy.sparse.csr.csr_matrix:
                    self.statsDict[key][iStep] = scipy.sparse.linalg.norm(self.v[key])
                    self.statsDict['D'+key][iStep] = scipy.sparse.linalg.norm(vnew[key] - self.v[key])
                else:
                    raise ValueError('Unexpected Variable Data Type: ', type(self.v[key]))



    '''
    VI_MAT_EXC  : Input voltage matrix [timesteps x N_EXC]
    WITH_IP     : If intrinsic plasticity is on
    WITH_SP     : If synaptic plasticity is on
    SOMA_NOISE_EXC  : Mean and variance of excitatory noise (default is None)
    SOMA_NOISE_INH  : Mean and variance of inhibitory noise (default is None)
    ''' 
    def run(self, VI_MAT_EXC, param_sim):
        
        # Get number of timesteps
        N_STEP = VI_MAT_EXC.shape[0]
        
        # Sanity check
        if not self.p['WITH_SYNAPSES'] and param_sim['WITH_SP']:
            raise ValueError('Requested SP but no synapses defined!')
        
        # Initialize statsDict
        self.V_Lst = []
        self.statsDict = copy.deepcopy(self.statsDictInit)
        for key in self.statsDict.keys():
            self.statsDict[key] = np.zeros(N_STEP)
                    
        timeCount = {'State' : 0, 'IP' : 0, 'STDP' : 0, 'STAT' : 0, 'UPDATE' : 0}
                
        # Run the simulation
        for iStep in range(N_STEP):
            if iStep % 20 == 0:
                print('Doing step', iStep, {k : round(v,2) for k,v in timeCount.items()})
            
            # Initialize temporary variable dictionary
            vnew = {}
            
            # Update neuronal activity and threshold
            
            for rowPopKey in self.p['POP_KEYS']:
                # Compute new state
                t = time()
                N_ROW = self.p['N_'+rowPopKey]
                VI = VI_MAT_EXC[iStep] if rowPopKey == 'EXC' else np.zeros(N_ROW)
                self.nextState(rowPopKey, VI, param_sim, vnew)
                timeCount['State'] += time() - t

                # Compute new threshold
                if param_sim['WITH_IP']:
                    t = time()
                    self.nextThresholds(rowPopKey, vnew)
                    timeCount['IP'] += time() - t
                
            # Compute new weight matrix
            # Note that STDP requires new values for membrane potential to be calculated beforehand
            if param_sim['WITH_SP']:
                t = time()
                for rowPopKey in self.p['POP_KEYS']:
                    for colPopKey in self.p['POP_KEYS']:
                        self.nextWeights(rowPopKey, colPopKey, vnew)
                timeCount['STDP'] += time() - t
            
            # Store statistics
            t = time()
            self.V_Lst.append(self.v['V_EXC'])
            self.updateStats(iStep, vnew)
            timeCount['STAT'] += time() - t
                
            # Update variables
            t = time()
            self.v.update(vnew)
            timeCount['UPDATE'] += time() - t
            