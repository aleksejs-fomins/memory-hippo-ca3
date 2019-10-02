import numpy as np
#import scipy.integrate as integrate
from integrator_lib import integrate_ode_ord1

def makeconn(nrow, ncol, mattype):
    if mattype == "dense":
        M = np.random.uniform(0, 1, (nrow, ncol))
        M /= np.linalg.norm(M)
        return M
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
        self.peqtype = []     # Map population index -> eq_type
        
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
        
        
    def add_population(self, pname, ptype, peqtype, psize):
        pidx = len(self.pname2idx)
        self.pname2idx[pname] = pidx
        self.ptype += [ptype]
        self.psize += [psize]
        self.peqtype += [peqtype]
        return pidx
        
        
    def add_synapse(self, sname, name1, name2, stype, splast):
        idx1, type1 = self.get_idx(name1)   # Use to assert that population with this name exists
        idx2, type2 = self.get_idx(name2)   # Use to assert that population with this name exists
        assert type2 != 'INP', "Input population can not be the target of a synapse"
        
        sidx = len(self.sname2idx)
        self.sname2idx[sname] = sidx
        self.sconn += [(name1, name2)]
        self.stype += [stype]
        self.splast += [splast]
        return sidx
    
    
    def construct(self):
        # Generating populations
        self.N_POP_TOTAL = np.sum(self.psize)
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
            
            
    # Compute neuronal value given all input to it
    def eval_rhs_pop(self, x, inpx, peqtype):
        if "LRN" in peqtype:    # Leaky rate neuron
            rez = (inpx - x) / self.p['TAU_X']
        elif "SRN" in peqtype:  # Slider rate neuron
            rez = inpx / self.p['TAU_X']
        else:
            raise ValueError("Unexpedted neuronal equation type", peqtype)
            
        # If neuron must be positive, and velocity tries to make it negative, set velocity to zero instead
        if "P" in peqtype:
            rez[x + rez*self.p['DT'] < 0] = 0
        
        return rez
            
            
    # Compute RHS of plasticity ODE depending on plasticity type
    def eval_rhs_plast(self, xpre, xpost, W, splast):
        if splast is None:
            return None            # Return None if synapse is not plastic
        elif splast == "Hebb":
            rez = np.outer(xpost, xpre) / self.p['TAU_W']
        elif splast == "XX_forw":
            rez = np.outer(xpost, xpre - W.T.dot(xpost)) / self.p['TAU_W']
        elif splast == "XX_back":
            rez = np.outer(xpost - W.dot(xpre), xpre) / self.p['TAU_W']
        else:
            raise ValueError("Unexpected plasticity type", self.splast[sidx])
            
        # synapses are not allowed to be negative
        rez[W - rez*self.p['DT'] < 0] = 0
        return rez


    # Compute RHS for neuronal and synaptic dynamics
    def get_rhs(self, timenow, pop, syn):
        inputPerPop = [np.zeros(l) for l in self.psize]
        rhssyn = []
        for (namepre, namepost), W, splast in zip(self.sconn, syn, self.splast):
            idxpre, typepre   = self.get_idx(namepre)
            idxpost, typepost = self.get_idx(namepost)
            
            xpre = pop[idxpre] if typepre != 'INP' else self.ifunc[idxpre](timenow)
            xpost = pop[idxpost]
            
            # Compute RHS for neuronal dynamics
            inputPerPop[idxpost] += self.eval_syn_output(W, xpre, typepre)
            
            # Compute RHS for synaptic dynamics
            rhssyn += [self.eval_rhs_plast(xpre, xpost, W, splast)]
                
        rhspop = [self.eval_rhs_pop(x, inpX, peqtype) for x, inpX, peqtype in zip(pop, inputPerPop, self.peqtype)]
        return rhspop, rhssyn
    
    
    # Pack simulation variables into single array
    # Do not include parameters that are not plastic
    def pack_var(self, pop, syn):
        pop_pack = np.hstack(pop)
        syn_pack = np.hstack([s.flatten() for i, s in enumerate(syn) if self.splast[i] is not None])
        return np.hstack([pop_pack, syn_pack])
    
    
    # Unpack simulation variables from a single array
    # Do not include parameters that are not plastic
    def unpack_var(self, pack):
        pop_pack = pack[:self.N_POP_TOTAL]
        syn_pack = pack[self.N_POP_TOTAL:]
        
        # Shift along 1D array, extract blocks of length psize
        sh = 0
        pop = []
        for psize in self.psize:
            pop += [pop_pack[sh:sh+psize]]
            sh += psize
            
        # Shift along 1D array, extract blocks of length shape1D.
        # If the block is not plastic, it is not present in the 1D array. Reuse generic one instead
        sh = 0
        syn = []
        for i in range(len(self.syn)):
            if self.splast[i] is None:
                syn += [self.syn[i]]
            else:
                shape = self.syn[i].shape
                shape1D = np.prod(shape)
                syn += [syn_pack[sh:sh+shape1D].reshape(shape)]
                sh += shape1D
                
        return pop, syn

                
    def run(self, T_MAX):
        # Run simulation
        N_STEP = int(T_MAX / self.p['DT'])
        #t_arr = np.linspace(0.0, self.p['DT']*N_STEP, N_STEP+1)
            
        # Unpack variables, compute RHS, then pack RHS
        rhs = lambda var, t: self.pack_var(*self.get_rhs(t, *self.unpack_var(var)))
        
        #sol = integrate.odeint(rhs, self.pack_var(self.pop, self.syn), t_arr)
        sol = integrate_ode_ord1(rhs, self.pack_var(self.pop, self.syn),  self.p['DT'], N_STEP, method='rk2')
            
        # Store populations if requested
        if self.p['STORE_POP'] or self.p['STORE_SYN'] or self.p['STORE_SYN_NORM']:
            for packed_step in sol:
                pop, syn = self.unpack_var(packed_step)
                if self.p['STORE_POP']:
                    for pname, pidx in self.pname2idx.items():
                        self.results[pname] += [pop[pidx]]
                if self.p['STORE_SYN']:
                    for sname, sidx in self.sname2idx.items():
                        self.results[sname] += [syn[sidx]]
                if self.p['STORE_SYN_NORM']:
                    for sname, sidx in self.sname2idx.items():
                        self.results[sname+'_norm'] += [np.linalg.norm(syn[sidx])]
                
                
    # Print the summary of neuronal populations and synapses present in the model
    def print_summary(self):
        print("Input Populations:")
        for k,v in self.iname2idx.items():
            print('   ', v, k, self.isize[v])
        
        print("Populations:")
        for k,v in self.pname2idx.items():
            print('   ', v, k, self.psize[v], self.ptype[v], self.peqtype[v])

        print("Synapses:")
        for k,v in self.sname2idx.items():
            print('   ', v, k, self.sconn[v], self.stype[v], self.splast[v])
            