'''
Generalized Predictive Coding Library:

TODO:
[ ] Communicate paramaters to populations (such as tau)
[ ] Allow storing history for some populations
[ ] Formalism to exclude certain populations at first layer
[ ] Extend forwards integration to RK4 or alike
[ ] Plots to compute error, backprop receptive fields, representation values
'''

import numpy as np
import copy

def makeconn(N, M, conntype):
    if conntype == "dense":
        return np.random.uniform(0, 1, (N, M))
    elif conntype == 'id':
        assert N==M, "dimensions of identity matrix must match"
        return np.identity(N)
    else:
        raise ValueError("Unexpected conn type", conntype)
    

class RaoBallardExtended():
    def __init__(self, param):
        assert "N_LAYERS" in param.keys(), "Number of layers must be specified"
        assert "DT"       in param.keys(), "Time step must be specified"
        self.p = param
        self.p['pop'] = {}
        self.p['syn'] = {}
        self.p['inppop'] = {}
        self.p['inpsyn'] = {}
    
    
    # Get parameters of a neuronal population
    def getpopparam(self, pname):
        return self.p['pop'][pname] if pname in self.p['pop'] else self.p['inppop'][pname]

    
    # Get size of a neuronal population at a given layer, or all sizes if layer is not specified
    def getsize(self, pname, layer=None):
        param = self.getpopparam(pname)
        return param['size'][layer] if layer is not None else param['size']
    
    
    # Add new population to the network
    # Each input population is present only once
    # All other populations are present once on each layer
    def add_population(self, pname, ptype, size):
        assert ptype in ["EXC", "INH", "INP"], "Unexpected population type " + ptype
        if ptype == "INP":
            assert type(size) == int, "Input population requires single integer size"
            self.p['inppop'][pname] = {'ptype' : ptype, 'size' : size}
        else:
            assert len(size) == self.p['N_LAYERS'], "Population " + pname + " quantity does not match layer count"
            self.p['pop'][pname] = {'ptype' : ptype, 'size' : size}
        
        
    # Add synapses
    # Both connected populations must already be specified
    # Again, input synapses are present only in one layer, all other synapses are the same for each layer
    def add_synapses(self, pname1, pname2, geomtype, conntype, plasticity):
        assert (pname1 in self.p['pop']) or (pname1 in self.p['inppop']), "Population "+pname1+" unknown"
        assert (pname2 in self.p['pop']) or (pname2 in self.p['inppop']), "Population "+pname2+" unknown"
        assert self.getpopparam(pname2)['ptype'] != 'INP', "Input population can not be the target of a synapse"
        assert geomtype in ['lateral', 'forward', 'backward'], "Unexpected synapse geometry"
        assert conntype in ['dense', 'sparse', 'id'], "Unexpected connectivity type"
        if conntype == 'id':
            assert geomtype == 'lateral', "Only lateral synapses allowed to be identity"
            assert self.getsize(pname1) == self.getsize(pname2), "Population sizes for identity connections must match"
        assert (pname1, pname2, geomtype) not in self.p['syn'].keys(), "Synapses "+str((pname1, pname2, geomtype))+" already exist"
        
        if self.getpopparam(pname1)['ptype'] == 'INP':
            assert geomtype == 'forward', "Input synapses can only be forward"
            self.p['inpsyn'][(pname1, pname2, geomtype)] = {"conntype" : conntype, "plasticity" : plasticity}
        else:
            self.p['syn'][(pname1, pname2, geomtype)] = {"conntype" : conntype, "plasticity" : plasticity}
        
        
    # Initialize neurons and synaptic weights
    def construct(self):        
        # Initialize input population
        assert 'inppop' in self.p, "Simulation does not have specified input population"
        self.inppop = {}
        for pname, v in self.p['pop'].items():
            self.inppop[pname] = np.zeros(v['size'])
        
        # Initializing internal populations
        self.pop = {}
        for pname, v in self.p['pop'].items():
            self.pop[pname] = [np.zeros(l) for l in v['size']]
            
        # Initializing input synapses
        self.inpsyn = {}
        for k, v in self.p['inpsyn'].items():
            pname1, pname2, geomtype = k
            self.inpsyn[(pname1, pname2, "INP")] = makeconn(self.getsize(pname1), self.getsize(pname2, 0), v["conntype"])
        
        # Initializing synapses
        self.syn = {}
        for k, v in self.p['syn'].items():
            pname1, pname2, geomtype = k
            if geomtype == "lateral":
                self.syn[k] = [makeconn(self.getsize(pname1, i), self.getsize(pname2, i), v["conntype"]) for i in range(self.p['N_LAYERS'])]
            elif geomtype == "forward":
                self.syn[k] = [makeconn(self.getsize(pname1, i), self.getsize(pname2, i+1), v["conntype"]) for i in range(self.p['N_LAYERS']-1)]
            elif geomtype == "backward":
                self.syn[k] = [makeconn(self.getsize(pname1, i+1), self.getsize(pname2, i), v["conntype"]) for i in range(self.p['N_LAYERS']-1)]
        
        
    # Add two lists of additive objects together element-wise
    def add_lists(l1, l2):
        return [v1 + v2 for v1, v2 in zip(l1, l2)]
    
    # Add two lists of additive objects together element-wise
    def add_lists_scale(l1, l2, a2):
        return [v1 + a2 * v2 for v1, v2 in zip(l1, l2)]
        
        
    # Compute effect of synaptic propagation depending on direction
    def propagateMV(trg, xlst, Mlst, geomtype):
        if geomtype == 'lateral':
            trg = add_lists(trg, [M.dot(x) for M, x in zip(Mlst, xlst)])
        elif geomtype == 'forward':
            trg[1:] = add_lists(trg[1:], [M.dot(x) for M, x in zip(Mlst, xlst[:-1])])
        elif geomtype == 'backward':
            trg[:-1] = add_lists(trg[:-1], [M.dot(x) for M, x in zip(Mlst, xlst[1:])])
        else:
            raise ValueError("Unexpected geometry type", geomtype)
        return trg
    
    def makeplasticity(xpre, xpost, W, geomtype, plasticity):
        
        if plasticity == "XX_forw":
            np.outer(xpost, xpre) - np.outer(xpost, W.T.dot(xpost))
        elif plasticity == "XX_back":
            np.outer(xpost, xpre) - np.outer(W.dot(xpre), xpre)
    
        
    # Update neuronal and synaptic values by one time-step
    def update(self, inp):
        # Initialize RHS of all internal populations with leak
        popRHS = {pname : [-x for x in val] for pname, val in self.pop.items()}
        
        # Update populations due to input
        for ksyn, mat in self.inpsyn.items():
            pname1, pname2, geomtype = ksyn
            popRHS[pname2][0] += mat.dot(inp)
        
        # Update populations due to internal synapses
        for ksyn, mat_lst in self.syn.items():
            pname1, pname2, geomtype = ksyn
            popRHS[pname2] = self.propagateMV(popRHS[pname2], self.pop[pname1], mat_lst, geomtype)

        # Compute first order forwards differences
        for pname, xold_lst in self.pop.items():
            self.pop[pname] = self.add_lists_scale(xold_lst, popRHS[pname], self.p['DT'])
        
        # Update input weights
        for ksyn, v in self.p['inpsyn'].items():
            if v["plasticity"] is not None:
                pname1, pname2, geomtype = ksyn
                rhs = self.makeplasticity(self.inppop[pname1], self.pop[pname2][0], self.inpsyn[ksyn], geomtype, v["plasticity"])
                self.inpsyn[ksyn] += rhs * self.p['DT']
        
        # Update weights
        synRHS = {}
        for ksyn, v in self.p['syn'].items():
            if v["plasticity"] is not None:
                pname1, pname2, geomtype = ksyn
                rhs = self.makeplasticity(self.pop[pname1], self.pop[pname2], self.syn[ksyn], geomtype, v["plasticity"])
                self.syn[ksyn] += rhs * self.p['DT']
                
        
        
    # Print the summary of neuronal populations and synapses present in the model
    def print_summary(self):
        print("Number of layers", self.p['N_LAYERS'])
        print("Input Populations:")
        for k,v in self.p['inppop'].items():
            print("  ", k, ':', v)
        print("Populations:")
        for k,v in self.p['pop'].items():
            print("  ", k, ':', v)
        
        print("Input Synapses:")
        for k,v in self.p['inpsyn'].items():
            print("  ", k, ':', v)
        print("Synapses:")
        for k,v in self.p['syn'].items():
            print("  ", k, ':', v)
