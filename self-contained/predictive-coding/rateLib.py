import numpy as np

def makeconn(nrow, ncol, mattype):
    if mattype == "dense":
        return np.random.uniform(0, 1, (nrow, ncol))
    elif mattype == 'id':
        assert nrow==ncol, "dimensions of identity matrix must match"
        return np.identity(nrow)
    else:
        raise ValueError("Unexpected matrix type", mattype)

        
'''
RATE NEURON SIMULATOR

[ ] Enable parameter translation
'''
        
class rateNeuronSimulator():
    def __init__(self, p):
        # Parameters
        self.p = p
        
        # Input Containers
        self.iname2idx = {}   # Map population name  -> index
        self.isize = []       # Map population index -> size
        self.ifunc = []       # Map population index -> input as function of time
        
        # Population Containers
        self.pname2idx = {}   # Map population name  -> index
        self.psize = []       # Map population index -> size
        self.ptype = []       # Map population index -> type
        
        # Synapse containers
        self.sname2idx = {}   # Map synapse name  -> index
        self.sconn = []
        self.stype = []
        self.splast = []

        
    # Get index of the population and its type using its name
    def get_idx(self, name):
        if name in self.pname2idx:
            pidx = self.pname2idx[name]
            return pidx, self.ptype[pidx]
        elif name in self.iname2idx:
            return self.iname2idx[name], 'INP'
        else:
            raise ValueError("Unknown population", name)
            
            
    # Get size of the population using its name
    def get_size(self, name):
        if name in self.pname2idx:
            return self.psize[self.pname2idx[name]]
        elif name in self.iname2idx:
            return self.isize[self.iname2idx[name]]
        else:
            raise ValueError("Unknown population", name)
            
        
    def add_input(self, iname, isize, ifunc):
        iidx = len(self.iname2idx)
        self.iname2idx[iname] = iidx
        self.isize += [isize]
        self.ifunc += [ifunc]
        return iidx
        
        
    def add_population(self, pname, ptype, psize):
        pidx = len(self.pname2idx)
        self.pname2idx[pname] = pidx
        self.ptype += [ptype]
        self.psize += [psize]
        return pidx
        
        
    def add_synapse(self, sname, name1, name2, stype, splast):
        idx1, type1 = self.get_idx(name1)   # Use to assert that population with this name exists
        idx2, type2 = self.get_idx(name2)   # Use to assert that population with this name exists
        assert type2 != 'INP', "Input population can not be the target of a synapse"
        
        sidx = len(self.sname2idx)
        self.sconn += [(name1, name2)]
        self.stype += [stype]
        self.splast += [splast]
        return sidx
    
    
    def construct(self):
        # Generating populations
        self.pop = [np.zeros(l) for l in self.psize]
        
        # Generating synapses
        self.syn = []
        for (namepre, namepost), stype in zip(self.sconn, self.stype):
            sizepre  = self.get_size(namepre)
            sizepost = self.get_size(namepost)
            self.syn += [makeconn(sizepost, sizepre, stype)]
            
        # Generating result storage
        self.results = {}
        if self.p['STORE_POP']:
            for pname in self.pname2idx.keys():
                self.results[pname] = []
        if self.p['STORE_SYN']:
            for sname in self.sname2idx.keys():
                self.results[sname] = []
        if self.p['STORE_SYN_NORM']:
            for sname in self.sname2idx.keys():
                self.results[sname+'_norm'] = []
           
        
    # Compute output of a synapse given synaptic weights and presynaptic activities
    def eval_syn_output(self, M, x, typepre):
        if typepre == "INH":
            return -M.dot(x)
        elif (typepre == "EXC") or (typepre == "INP"):
            return M.dot(x)
        else:
            raise ValueError("Unknown synapse type", typepre)
            
            
    # Compute RHS of plasticity ODE depending on plasticity type
    def eval_rhs_plast(self, xpre, xpost, W, splast):
        if splast is None:
            return None            # Return None if synapse is not plastic
        elif splast == "XX_forw":
            return (np.outer(xpost, xpre) - np.outer(xpost, W.T.dot(xpost))) / self.p['TAU_W']
        elif splast == "XX_back":
            return (np.outer(xpost, xpre) - np.outer(W.dot(xpre), xpre)) / self.p['TAU_W']
        else:
            raise ValueError("Unexpected plasticity type", self.splast[sidx])


    # Compute RHS for neuronal and synaptic dynamics
    def get_rhs(self, timenow):
        rhspop = [-x for x in self.pop]     # Initialize population RHS with leak
        rhssyn = []
        for (namepre, namepost), W, splast in zip(self.sconn, self.syn, self.splast):
            idxpre, typepre   = self.get_idx(namepre)
            idxpost, typepost = self.get_idx(namepost)
            
            xpre = self.pop[idxpre] if typepre != 'INP' else self.ifunc[idxpre](timenow)
            xpost = self.pop[idxpost]
            
            # Compute RHS for neuronal dynamics
            rhspop[idxpost] += self.eval_syn_output(W, xpre, typepre) / self.p['TAU_X']
            
            # Compute RHS for synaptic dynamics
            rhssyn += [self.eval_rhs_plast(xpre, xpost, W, splast)]
                
        return rhspop, rhssyn
    
    
    # Update values
    def update(self, timenow):
        rhspop, rhssyn = self.get_rhs(timenow)
        
        # Update neurons
        for i in range(len(self.pop)):
            self.pop[i] += rhspop[i] * self.p['DT']
            
        # Update synapses
        for i in range(len(self.syn)):
            if rhssyn[i] is not None:
                self.syn[i] += rhssyn[i] * self.p['DT']
                
    def run(self, T_MAX):
        # Run simulation
        t_arr = np.arange(0, T_MAX, self.p['DT']) + self.p['DT']        
        for t in t_arr:
            self.update(t_arr)
            
            # Store populations if requested
            if self.p['STORE_POP']:
                for pname, pidx in self.pname2idx.items():
                    self.results[pname] += [self.pop[pidx]]
            if self.p['STORE_SYN']:
                for sname, sidx in self.sname2idx.items():
                    self.results[sname] += [self.syn[sidx]]
            if self.p['STORE_SYN_NORM']:
                for sname, sidx in self.sname2idx.items():
                    self.results[sname+'_norm'] += [np.linalg.norm(self.syn[sidx])]
            
                
                
    # Print the summary of neuronal populations and synapses present in the model
    def print_summary(self):
        print("Input Populations:")
        for k,v in self.iname2idx.items():
            print('   ', v, k, self.isize[v])
        
        print("Populations:")
        for k,v in self.pname2idx.items():
            print('   ', v, k, self.psize[v], self.ptype[v])

        print("Synapses:")
        for k,v in self.pname2idx.items():
            print('   ', v, k, self.sconn[v], self.stype[v], self.splast[v])
            