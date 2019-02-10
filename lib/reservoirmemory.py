import numpy as np
import numpy.matlib

from sklearn.preprocessing import normalize

import aux

class ReservoirMemory:
    def __init__(self, p):
        
        #self.N_SRC = p['N_SRC']
        #self.N_OUT = p['N_OUT']
        self.N_EXC = p['N_EXC']
        self.N_INH = p['N_INH']
        self.N_REZ = self.N_EXC + self.N_INH
        
        ##################################
        ##synaptic connections matrix
        ##################################
        
        # Define indices for the EXC and INH parts of the neuron vector and reservoir matrix
        # Note that row index denotes target, but column index denotes source. Thus for matrix indices the labels are flipped
        self.idxE = np.arange(0,          self.N_EXC)
        self.idxI = np.arange(self.N_EXC, self.N_REZ)
        self.idxEE = np.ix_(self.idxE, self.idxE)
        self.idxEI = np.ix_(self.idxI, self.idxE)
        self.idxIE = np.ix_(self.idxE, self.idxI)
        self.idxII = np.ix_(self.idxI, self.idxI)
        
        # Create connectivity matrix based on provided connection probabilities
        self.CRR = np.zeros((self.N_REZ, self.N_REZ))
        self.CRR[self.idxEE] = (np.random.rand(self.N_EXC, self.N_EXC) < p['P_CONN_EE']).astype(int)
        index_plastic = self.CRR #learning restricted to positive EE synapses
        self.CRR[self.idxEI] = (np.random.rand(self.N_INH, self.N_EXC) < p['P_CONN_EI']).astype(int)
        self.CRR[self.idxIE] = (np.random.rand(self.N_EXC, self.N_INH) < p['P_CONN_IE']).astype(int)
        self.CRR[self.idxII] = (np.random.rand(self.N_INH, self.N_INH) < p['P_CONN_II']).astype(int)
        
        # Create reservoir matrix
        self.WRR = np.zeros((self.N_REZ, self.N_REZ))
        self.WRR[self.idxEE] = aux.randUMat((self.N_EXC, self.N_EXC), p['W_RANGE_EXC'])
        self.WRR[self.idxEI] = aux.randUMat((self.N_INH, self.N_EXC), p['W_RANGE_EXC'])
        self.WRR[self.idxIE] = aux.randUMat((self.N_EXC, self.N_INH), p['W_RANGE_INH'])
        self.WRR[self.idxII] = aux.randUMat((self.N_INH, self.N_INH), p['W_RANGE_INH'])
        self.WRR = np.multiply(self.WRR, self.CRR)  # Set all non-existent connections to zero
        
        # synaptic normalization - normalize all inputs to each neuron
        # Note: sklearn normalize is smart enough not to normalize zero-rows in case some neurons have no input
        self.W_NORM_EXC = p['W_NORM_EXC']
        self.W_NORM_INH = p['W_NORM_INH']
        self.WRR[self.idxE, :] = normalize(self.WRR[self.idxE, :], axis=1, norm='l1') * p['W_NORM_EXC']
        self.WRR[self.idxI, :] = normalize(self.WRR[self.idxI, :], axis=1, norm='l1') * p['W_NORM_INH']
        
        # Create input and output matrices
        #self.WSR = np.zeros((self.N_REZ, self.N_SRC))  # Source    -> Reservoir
        #self.WRO = np.zeros((self.N_OUT, self.N_REZ))  # Reservoir -> Output
        self.WSR = np.identity(self.N_REZ)  # Source    -> Reservoir
        self.WRO = np.identity(self.N_REZ)  # Reservoir -> Output

        ##################################
        ## Neurons
        ##################################
        
        # Initial synaptic potential
        self.V = np.zeros(self.N_REZ)
        
        # threshold values for each neuron
        self.T = np.hstack((
            np.random.uniform(p['THR_RANGE_EXC'][0], p['THR_RANGE_EXC'][1], self.N_EXC),
            np.random.uniform(p['THR_RANGE_INH'][0], p['THR_RANGE_INH'][1], self.N_INH)))
        
        # Intrinsic plasticity equillibrium spiking rate
        self.EQ_RATE_IP = np.hstack((
            p['EQ_RATE_IP_EXC']*np.ones(self.N_EXC),
            p['EQ_RATE_IP_INH']*np.ones(self.N_INH)))
        
        # Intrinsic plasticity timescale
        # Neurons for which intrinsic plasticity not enabled will have 0 prefactor, the rest will have IP_RATE
        self.IP_RATE = np.zeros(self.N_REZ)
        self.IP_RATE[p['IP_ACTIVE_NEURON_RANGE']] = p['IP_RATE']  
        
        # Synaptic plasticity timescale
        # Weights for which synaptic plasticity not enabled will have 0 prefactor, the rest will have SP_RATE
        self.SP_RATE = index_plastic * p['SP_RATE']
        
        # Storage for activities
        self.V_Lst = []
        self.O_Lst = []
        
    
    def nextState(self, VS, V, T, WSR, WRR):
        return (WSR.dot(VS) + WRR.dot(V) > T).astype(int)
    
    
    def nextReadout(self, V, WRO):
        return WRO.dot(V)
    
    
    # Calculate change in threshold due to intrinsic plasticity
    # IP_RATE is non-zero only if IP is allowed for that neuron
    def nextThreshold(self, V_Old, T, IP_RATE, EQ_RATE_IP):
        return T + np.multiply(IP_RATE, V_Old - EQ_RATE_IP);
        
        
    # Calculate change in weight due to synaptic plasticity
    # SP_RATE is non-zero only if SP is allowed for that synapse
    def nextWeights(self, V_Old, V, WRR, SP_RATE):
        #print(V.shape)
        #print(SP_RATE.shape)        
        
        #additive STDP
        CorrM = np.outer(V_Old, V)             # Correlation matrix between previous and new time step activities
        HebbM = CorrM - np.transpose(CorrM);   # Difference between causal and anti-causal correlations
        WRRnew = np.multiply(SP_RATE, WRR + HebbM)  # Causal weights strengthened, anti-causal weakened
        
        # Clip all Excitatory weights to [0, 1], Inhibitory weights to [-1, 0]
        WRRnew[:, self.idxE] = aux.clipMat(WRRnew[:, self.idxE], [0, 1])
        WRRnew[:, self.idxI] = aux.clipMat(WRRnew[:, self.idxI], [-1, 0])

        # synaptic normalization - normalize all inputs to each neuron
        # Note: sklearn normalize is smart enough not to normalize zero-rows in case some neurons have no input
        WRRnew[self.idxE, :] = normalize(WRRnew[self.idxE, :], axis=1, norm='l1') * self.W_NORM_EXC
        WRRnew[self.idxI, :] = normalize(WRRnew[self.idxI, :], axis=1, norm='l1') * self.W_NORM_INH
        
        return WRRnew
        
        
    def evolve(self, VS_Lst, allowIP, allowSP):
        
        for VS in VS_Lst:
            # Compute new state
            VNew  = self.nextState(VS, self.V, self.T, self.WSR, self.WRR)
            
            # Compute new readout
            ONew = self.nextReadout(self.V, self.WRO)
            
            # Compute new threshold
            if allowIP:
                TNew = self.nextThreshold(self.V, self.T, self.IP_RATE, self.EQ_RATE_IP)
            else:
                TNew = self.T
                
            # Compute new weight matrix
            if allowSP:
                WRRnew = self.nextWeights(self.V, VNew, self.WRR, self.SP_RATE)
            else:
                WRRnew = self.WRR
            
            # Update state
            self.V = VNew
            self.T = TNew
            self.WRR = WRRnew
            
            # Store data
            self.V_Lst.append(VNew)
            self.O_Lst.append(ONew)
            
