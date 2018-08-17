import numpy as np
import numpy.matlib
import copy

from sklearn.preprocessing import normalize
from scipy.sparse import csr_matrix, hstack, vstack

import aux


'''
Single Population Discrete Neural Network

Parameters:
    N_EXC            : Number of excitatory neurons in the network
    N_INH            : Number of inhibitory neurons in the network

    T_RANGE_EXC      : Initial range of thresholds for excitatory neurons
    T_RANGE_INH      : Initial range of thresholds for inhibitory neurons
    W_RANGE_EXC      : Initial range of excitatory weights for uniform random distribution
    W_RANGE_INH      : Initial range of inhibitory weights for uniform random distribution

    P_CONN           : Probability of [exc-exc, exc-inh, inh-exc, inh-inh] connection

    SP_RATE          : Rate of synaptic plasticity for [exc-exc, exc-inh, inh-exc, inh-inh] connection
    IP_RATE_EXC      : Rate of intrinsic plasticity for excitatory neurons
    IP_RATE_INH      : Rate of intrinsic plasticity for inhibitory neurons
    V_EQ_EXC         : Equillibrium membrane potential for excitatory neurons
    V_EQ_INH         : Equillibrium membrane potential for inhibitory neurons

    WITH_SYNAPSES    : Whether the population has synapses at all
    WITH_INIT_NORM_INP  : Whether to normalize input weights to each neuron
    
TODO:
    [ ] Implement actual sparsity
       [ ] Sparse outer product
       [ ] Sparse clipping
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
        self.p = p
        
        ##################################
        ## Neurons
        ##################################
        
        # Initial synaptic potential (EXC and INH)
        self.V = {
            'EXC' : np.zeros(p['N_EXC']),
            'INH' : np.zeros(p['N_INH'])}
        
        # Threshold values for each neuron (EXC and INH)
        self.T = {
            'EXC' : np.random.uniform(p['T_RANGE_EXC'][0], p['T_RANGE_EXC'][1], p['N_EXC']),
            'INH' : np.random.uniform(p['T_RANGE_INH'][0], p['T_RANGE_INH'][1], p['N_INH'])}
        
        ##################################
        ##synaptic connections matrix
        ##################################
        
        if p['WITH_SYNAPSES']:
            # Create weight matrix
            self.W = {
                'EE' : spRandUMat((p['N_EXC'], p['N_EXC']), p['W_RANGE_EXC'], p['P_CONN']['EE']),
                'EI' : spRandUMat((p['N_EXC'], p['N_INH']), p['W_RANGE_EXC'], p['P_CONN']['EI']),
                'IE' : spRandUMat((p['N_INH'], p['N_EXC']), p['W_RANGE_INH'], p['P_CONN']['IE']),
                'II' : spRandUMat((p['N_INH'], p['N_INH']), p['W_RANGE_INH'], p['P_CONN']['II'])}
            
            # Excract its sparse connectivity indices 
            self.Widx = {
                'EE' : [np.array(self.W['EE'].tocoo().row), np.array(self.W['EE'].tocoo().col)],
                'EI' : [np.array(self.W['EI'].tocoo().row), np.array(self.W['EI'].tocoo().col)],
                'IE' : [np.array(self.W['IE'].tocoo().row), np.array(self.W['IE'].tocoo().col)],
                'II' : [np.array(self.W['II'].tocoo().row), np.array(self.W['II'].tocoo().col)]}

            # TODO: Problem is that normalization is over two different matrices. Would need to take norm wrt given axis, add sqrt(EE^2 + EI^2), then normalize each row manually
            # synaptic normalization - normalize all inputs to each neuron
            # Note: sklearn normalize is smart enough not to normalize zero-rows in case some neurons have no input
            #if p['WITH_INIT_NORM_INP']:
            #    self.W[self.idxE, :] = normalize(self.W[self.idxE, :], axis=1, norm='l1') * p['W_NORM_EXC']
            #    self.W[self.idxI, :] = normalize(self.W[self.idxI, :], axis=1, norm='l1') * p['W_NORM_INH']
        else:
            self.W = None
        
        
        ##################################
        ## Statistics
        ##################################
        
        # Storage for activities
        self.V_Lst = []
        
        self.statsDictInit = {
            'V_EXC' : [],
            'V_INH' : [],
            'T_EXC' : [],
            'T_INH' : [],
            'W_EE' : [],
            'W_EI' : [],
            'W_IE' : [],
            'W_II' : [],
            
            'DV_EXC' : [],
            'DV_INH' : [],
            'DT_EXC' : [],
            'DT_INH' : [],
            'DW_EE' : [],
            'DW_EI' : [],
            'DW_IE' : [],
            'DW_II' : []
        }
        
    
    # Create separate noise for excitatory and inhibitory neurons
    def makeNoise(self, noise):
        if noise is None:
            return {
                'EXC' : np.zeros(self.p['N_EXC']),
                'INH' : np.zeros(self.p['N_INH'])}
        else:
            return {
                'EXC' : np.random.normal(noise['EXC'][0], noise['EXC'][1], self.p['N_EXC']),
                'INH' : np.random.normal(noise['INH'][0], noise['INH'][1], self.p['N_INH'])}
    
    
    # Calculate the next membrane potential
    def nextState(self, VI, VNoise, V, T, W):
        if self.p['WITH_SYNAPSES']:
            VSyn = {
                'EXC' : W['EE'].dot(V['EXC']) + W['IE'].dot(V['INH']),
                'INH' : W['EI'].dot(V['EXC']) + W['II'].dot(V['INH'])}
        else:
            VSyn = {
                'EXC' : np.zeros(self.p['N_EXC']),
                'INH' : np.zeros(self.p['N_INH'])}
            
        return {
            'EXC' : (VI['EXC'] + VNoise['EXC'] + VSyn['EXC'] > T['EXC']).astype(int),
            'INH' : (VI['INH'] + VNoise['INH'] + VSyn['INH'] > T['INH']).astype(int)}
    
    
    # Calculate change in threshold due to intrinsic plasticity
    def nextThresholds(self, V, T):
        TNew = {
            'EXC' : T['EXC'] + self.p['IP_RATE_EXC'] * (V['EXC'] - self.p['V_EQ_EXC']),
            'INH' : T['INH'] + self.p['IP_RATE_INH'] * (V['INH'] - self.p['V_EQ_INH'])}

        # Clip all thresholds to allowed values
        aux.clipMat(TNew['EXC'], self.p['T_RANGE_EXC'])
        aux.clipMat(TNew['INH'], self.p['T_RANGE_INH'])
        
        return TNew
        
        
    # Calculate change in weight due to synaptic plasticity
    def nextWeights(self, VOld, V, W):
        
        #additive STDP
        WNew = {}
        for key in self.W.keys():
            if self.SP_RATE[key] != 0:
                rowidx = self.Widx[key][0]
                colidx = self.Widx[key][1]
                
                # Compute correlation matrix between previous and new time step activities
                # Hebb matrix is the difference between causal and anti-causal correlations in activities
                HebbVal = np.multiply(VOld[rowidx], V[colidx]) - np.multiply(V[rowidx], VOld[colidx])
                WNew[key] = W[key] + self.SP_RATE[key] * csr_matrix((HebbVal, (idxrow, idxcol)), shape=W[key].shape, dtype=W[key].dtype)
            
                # Clip all excitatory weights to their allowed ranges
                rangeKey = 'W_RANGE_EXC' if (key == 'EE' or key == 'EI') else 'W_RANGE_INH'
                aux.clipMat(WNew[key], self.p[rangeKey])
            else:
                WNew[key] = W[key]

        # TODO: Problem is that normalization is over two different matrices. Would need to take norm wrt given axis, add sqrt(EE^2 + EI^2), then normalize each row manually
        # synaptic normalization - normalize all inputs to each neuron
        # Note: sklearn normalize is smart enough not to normalize zero-rows in case some neurons have no input
        #if p['WITH_INIT_NORM_INP']:
        #    WNew[self.idxE, :] = normalize(WNew[self.idxE, :], axis=1, norm='l1') * self.W_NORM_EXC
        #    WNew[self.idxI, :] = normalize(WNew[self.idxI, :], axis=1, norm='l1') * self.W_NORM_INH
        
        return WNew
        
    
    def updateStats(self, iStep, V, VNew, T, TNew, W = None, WNew = None):
        # Calculate magnitudes of potentials, weights and thresholds for excitatory and inhibitory populations
        self.statsDict['V_EXC'][iStep] = np.linalg.norm(V['EXC'])
        self.statsDict['V_INH'][iStep] = np.linalg.norm(V['INH'])
        self.statsDict['T_EXC'][iStep] = np.linalg.norm(T['EXC'])
        self.statsDict['T_INH'][iStep] = np.linalg.norm(T['INH'])
        if self.p['WITH_SYNAPSES']:
            self.statsDict['W_EE'][iStep] = np.linalg.norm(W['EE'])
            self.statsDict['W_EI'][iStep] = np.linalg.norm(W['EI'])
            self.statsDict['W_IE'][iStep] = np.linalg.norm(W['IE'])
            self.statsDict['W_II'][iStep] = np.linalg.norm(W['II'])
        
        # Calculate changes of potentials, weights and thresholds for excitatory and inhibitory populations
        self.statsDict['DV_EXC'][iStep] = np.linalg.norm(VNew['EXC'] - V['EXC'])
        self.statsDict['DV_INH'][iStep] = np.linalg.norm(VNew['INH'] - V['INH'])
        self.statsDict['DT_EXC'][iStep] = np.linalg.norm(TNew['EXC'] - T['EXC'])
        self.statsDict['DT_INH'][iStep] = np.linalg.norm(TNew['INH'] - T['INH'])
        if self.p['WITH_SYNAPSES']:
            self.statsDict['DW_EE'][iStep] = np.linalg.norm(WNew['EE'] - W['EE'])
            self.statsDict['DW_EI'][iStep] = np.linalg.norm(WNew['EI'] - W['EI'])
            self.statsDict['DW_IE'][iStep] = np.linalg.norm(WNew['IE'] - W['IE'])
            self.statsDict['DW_II'][iStep] = np.linalg.norm(WNew['II'] - W['II'])


    '''
    VI_MAT_EXC  : Input voltage matrix [timesteps x N_EXC]
    WITH_IP     : If intrinsic plasticity is on
    WITH_SP     : If synaptic plasticity is on
    SOMA_NOISE  : Mean and variance of excitatory and inhibitory noise (default is None)
    ''' 
    def run(self, VI_MAT_EXC, param_sim):
        
        # Get number of timesteps
        N_STEP = VI_MAT_EXC.shape[0]
        
        # Sanity check
        if not self.p['WITH_SYNAPSES'] and param_sim['WITH_SP']:
            raise ValueError('Requested SP but no synapses defined!')
        
        # Initialize statsDict
        self.statsDict = copy.deepcopy(self.statsDictInit)
        for key in self.statsDict.keys():
            self.statsDict[key] = np.zeros(N_STEP)
        
        # Run the simulation
        for iStep in range(N_STEP):
            # Compute synaptic noise for this step
            VNoise = self.makeNoise(param_sim['SOMA_NOISE']) if 'SOMA_NOISE' in param_sim.keys() else self.makeNoise(None)
            
            # Preformat input
            VI = {
                'EXC' : VI_MAT_EXC[iStep],
                'INH' : np.zeros(self.p['N_INH'])}  # There is no inhibitory input, but we keep the option for symmetry
            
            # Compute new state
            VNew  = self.nextState(VI, VNoise, self.V, self.T, self.W)
            
            # Compute new threshold
            TNew = self.T if not param_sim['WITH_IP'] else self.nextThresholds(self.V, self.T)
                
            # Compute new weight matrix
            WNew = self.W if not param_sim['WITH_SP'] else self.nextWeights(self.V, VNew, self.W)
            
            # Store statistics
            # self.V_Lst.append(VNew)
            self.updateStats(iStep, self.V, VNew, self.T, TNew, self.W, WNew)
                
            # Update variables
            self.V = VNew
            self.T = TNew
            self.W = WNew
            
