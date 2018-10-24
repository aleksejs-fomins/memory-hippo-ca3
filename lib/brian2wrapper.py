from brian2 import *

def NeuronGroupLIF(N, V0, VMAX, TAU):
    return NeuronGroup(N, 'dv/dt = (V0-v)/TAU : volt', threshold='v > VMAX', reset='v = V0', method='exact')


def SynapsesPlastic(G1, G2, plasticity_model):
    
    if plasticity_model['TYPE'] == 'STDP':
        # Extract parameters
        TAU_PRE = plasticity_model['TAU_PRE']
        TAU_POST = plasticity_model['TAU_POST']
        DW_FORW = plasticity_model['DW_FORW']
        DW_BACK = plasticity_model['DW_BACK']
        DV_SPIKE = plasticity_model['DV_SPIKE']
        REL_W_MIN = plasticity_model['REL_W_MIN']
        REL_W_MAX = plasticity_model['REL_W_MAX']
        
        
        # Two auxiliary variables track decaying trace of
        # presynaptic and postsynaptic spikes
        syn_eq = '''
        w : 1
        dzpre/dt = -zpre/TAU_PRE : 1 (event-driven)
        dzpost/dt = -zpost/TAU_POST : 1 (event-driven)
        '''

        # On spike increase decaying variable by fixed amount
        # Increase weight by the value of the decaying variable
        # from the other side of the synapse
        # Truncate weight if it exceeds maximum
        syn_pre_eq = '''
        zpre += 1
        w = clip(w + DW_FORW * zpost, REL_W_MIN, REL_W_MAX)
        v_post += DV_SPIKE * w
        '''

        syn_post_eq = '''
        zpost += 1
        w = clip(w + DW_BACK * zpre, REL_W_MIN, REL_W_MAX)
        '''

        return Synapses(G1, G1, syn_eq, on_pre=syn_pre_eq, on_post=syn_post_eq)
    else:
        raise ValueError('Unexpected Plasticity type', plasticity_model['TYPE'])