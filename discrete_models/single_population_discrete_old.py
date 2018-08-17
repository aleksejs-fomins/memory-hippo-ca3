import numpy as np
import numpy.matlib
import copy

from sklearn.preprocessing import normalize

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
    WITH_SOMATIC_PL  : Whether there is somatic plasticity at all
    WITH_INIT_NORM_INP  : Whether to normalize input weights to each neuron
    
TODO:
    [ ] Implement actual sparsity
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
        self.p['N_TOT'] = p['N_EXC'] + p['N_INH']
        
        ##################################
        ## Neurons
        ##################################
        
        # Define indices for the EXC and INH parts of the neuron vector
        self.idxE = np.arange(0,          p['N_EXC'])
        self.idxI = np.arange(p['N_EXC'], p['N_TOT'])
        
        # Initial synaptic potential
        self.V = np.zeros(p['N_TOT'])
        
        # Threshold values for each neuron
        self.T = np.hstack((
            np.random.uniform(p['T_RANGE_EXC'][0], p['T_RANGE_EXC'][1], p['N_EXC']),
            np.random.uniform(p['T_RANGE_INH'][0], p['T_RANGE_INH'][1], p['N_INH'])))
        
        if p['WITH_SOMATIC_PL']:
            # Intrinsic plasticity for each neuron
            self.IP_RATE = np.hstack((
                p['IP_RATE_EXC']*np.ones(p['N_EXC']),
                p['IP_RATE_INH']*np.ones(p['N_INH'])))
            
            # Equillibrium membrane potential for each neuron
            self.V_EQ = np.hstack((
                p['V_EQ_EXC']*np.ones(p['N_EXC']),
                p['V_EQ_INH']*np.ones(p['N_INH'])))
        
        
        ##################################
        ##synaptic connections matrix
        ##################################
        
        if p['WITH_SYNAPSES']:
            # Define indices for the EXC and INH parts of the neuron matrix
            # Note that row index denotes target, but column index denotes source. Thus for matrix indices the labels are flipped
            self.idxEE = np.ix_(self.idxE, self.idxE)
            self.idxEI = np.ix_(self.idxI, self.idxE)
            self.idxIE = np.ix_(self.idxE, self.idxI)
            self.idxII = np.ix_(self.idxI, self.idxI)
            
            # Create connectivity matrix based on provided connection probabilities
            self.CRR = np.zeros((p['N_TOT'], p['N_TOT']))
            self.CRR[self.idxEE] = (np.random.rand(p['N_EXC'], p['N_EXC']) < p['P_CONN'][0]).astype(int)
            self.CRR[self.idxEI] = (np.random.rand(p['N_INH'], p['N_EXC']) < p['P_CONN'][1]).astype(int)
            self.CRR[self.idxIE] = (np.random.rand(p['N_EXC'], p['N_INH']) < p['P_CONN'][2]).astype(int)
            self.CRR[self.idxII] = (np.random.rand(p['N_INH'], p['N_INH']) < p['P_CONN'][3]).astype(int)
            self.N_CONN = np.sum(self.CRR)
            
            # Create weight matrix
            self.W = np.zeros((p['N_TOT'], p['N_TOT']))
            self.W[self.idxEE] = aux.randUMat((p['N_EXC'], p['N_EXC']), p['W_RANGE_EXC'])
            self.W[self.idxEI] = aux.randUMat((p['N_INH'], p['N_EXC']), p['W_RANGE_EXC'])
            self.W[self.idxIE] = aux.randUMat((p['N_EXC'], p['N_INH']), p['W_RANGE_INH'])
            self.W[self.idxII] = aux.randUMat((p['N_INH'], p['N_INH']), p['W_RANGE_INH'])
            self.W = np.multiply(self.W, self.CRR)  # Set all non-existent connections to zero
            
            # synaptic normalization - normalize all inputs to each neuron
            # Note: sklearn normalize is smart enough not to normalize zero-rows in case some neurons have no input
            if p['WITH_INIT_NORM_INP']:
                self.W[self.idxE, :] = normalize(self.W[self.idxE, :], axis=1, norm='l1') * p['W_NORM_EXC']
                self.W[self.idxI, :] = normalize(self.W[self.idxI, :], axis=1, norm='l1') * p['W_NORM_INH']
                
            # Synaptic plasticity rate for each synapse
            # Non-existing synapses will have zero rate
            self.SP_RATE = np.zeros((p['N_TOT'], p['N_TOT']))
            self.SP_RATE[self.idxEE] = p['SP_RATE'][0]
            self.SP_RATE[self.idxEI] = p['SP_RATE'][1]
            self.SP_RATE[self.idxIE] = p['SP_RATE'][2]
            self.SP_RATE[self.idxII] = p['SP_RATE'][3]
            self.SP_RATE = np.multiply(self.SP_RATE, self.CRR)  # Set all non-existent connections to zero
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
    def makeNoise(self, sigExc, sigInh):
        return np.hstack((
            np.zeros(self.p['N_EXC']) if sigExc is None else np.random.normal(0, sigExc, self.p['N_EXC']),
            np.zeros(self.p['N_INH']) if sigInh is None else np.random.normal(0, sigInh, self.p['N_INH'])))
    
    
    # Calculate the next membrane potential
    def nextState(self, VI, VNoise, V, T, W):
        VSyn = W.dot(V) if self.p['WITH_SYNAPSES'] else np.zeros(self.p['N_TOT'])
        return (VI + VNoise + VSyn > T).astype(int)
    
    
    # Calculate change in threshold due to intrinsic plasticity
    def nextThresholds(self, V, T):
        TNew = T + np.multiply(self.IP_RATE, V - self.V_EQ)

        # Clip all thresholds to allowed values
        TNew[self.idxE] = aux.clipMat(TNew[self.idxE], self.p['T_RANGE_EXC'])
        TNew[self.idxI] = aux.clipMat(TNew[self.idxI], self.p['T_RANGE_INH'])
        
        return TNew
        
        
    # Calculate change in weight due to synaptic plasticity
    def nextWeights(self, V_Old, V, W):
        #additive STDP
        CorrM = np.outer(V_Old, V)                   # Correlation matrix between previous and new time step activities
        HebbM = CorrM - np.transpose(CorrM);         # Difference between causal and anti-causal correlations
        WNew = np.multiply(self.SP_RATE, W + HebbM)  # Causal weights strengthened, anti-causal weakened
        
        # Clip all excitatory weights to their allowed ranges
        WNew[:, self.idxE] = aux.clipMat(WNew[:, self.idxE], self.p['W_RANGE_EXC'])
        WNew[:, self.idxI] = aux.clipMat(WNew[:, self.idxI], self.p['W_RANGE_INH'])

        # synaptic normalization - normalize all inputs to each neuron
        # Note: sklearn normalize is smart enough not to normalize zero-rows in case some neurons have no input
        if p['WITH_INIT_NORM_INP']:
            WNew[self.idxE, :] = normalize(WNew[self.idxE, :], axis=1, norm='l1') * self.W_NORM_EXC
            WNew[self.idxI, :] = normalize(WNew[self.idxI, :], axis=1, norm='l1') * self.W_NORM_INH
        
        return WNew
        
    
    def updateStats(self, iStep, V, VNew, T, TNew, W = None, WNew = None):
        # Calculate magnitudes of potentials, weights and thresholds for excitatory and inhibitory populations
        self.statsDict['V_EXC'][iStep] = np.linalg.norm(V[self.idxE])
        self.statsDict['V_INH'][iStep] = np.linalg.norm(V[self.idxI])
        self.statsDict['T_EXC'][iStep] = np.linalg.norm(T[self.idxE])
        self.statsDict['T_INH'][iStep] = np.linalg.norm(T[self.idxI])
        if self.p['WITH_SYNAPSES']:
            self.statsDict['W_EE'][iStep] = np.linalg.norm(W[self.idxEE])
            self.statsDict['W_EI'][iStep] = np.linalg.norm(W[self.idxEI])
            self.statsDict['W_IE'][iStep] = np.linalg.norm(W[self.idxIE])
            self.statsDict['W_II'][iStep] = np.linalg.norm(W[self.idxII])
        
        # Calculate changes of potentials, weights and thresholds for excitatory and inhibitory populations
        self.statsDict['DV_EXC'][iStep] = np.linalg.norm(VNew[self.idxE] - V[self.idxE])
        self.statsDict['DV_INH'][iStep] = np.linalg.norm(VNew[self.idxI] - V[self.idxI])
        self.statsDict['DT_EXC'][iStep] = np.linalg.norm(TNew[self.idxE] - T[self.idxE])
        self.statsDict['DT_INH'][iStep] = np.linalg.norm(TNew[self.idxI] - T[self.idxI])
        if self.p['WITH_SYNAPSES']:
            self.statsDict['DW_EE'][iStep] = np.linalg.norm(WNew[self.idxEE] - W[self.idxEE])
            self.statsDict['DW_EI'][iStep] = np.linalg.norm(WNew[self.idxEI] - W[self.idxEI])
            self.statsDict['DW_IE'][iStep] = np.linalg.norm(WNew[self.idxIE] - W[self.idxIE])
            self.statsDict['DW_II'][iStep] = np.linalg.norm(WNew[self.idxII] - W[self.idxII])
        
    '''
    VI_MAT    : Input voltage matrix [timesteps x neurons]
    WITH_IP   : If intrinsic plasticity is on
    WITH_SP   : If synaptic plasticity is on
    NOISE_MAG_EXC = None
    NOISE_MAG_INH = None
    ''' 
    def run(self, VI_MAT, param_sim):
        
        # Get number of timesteps
        N_STEP = VI_MAT.shape[0]
        
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
            VNoise = self.makeNoise(param_sim['NOISE_MAG_EXC'], param_sim['NOISE_MAG_INH'])
            
            # Compute new state
            VNew  = self.nextState(VI_MAT[iStep], VNoise, self.V, self.T, self.W)
            
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
            
