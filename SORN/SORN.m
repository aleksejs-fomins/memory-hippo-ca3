% Self Organizing Recurrent Network
% Random recurrent network is trained via plasticity (STDP, SN and IP)
% http://www.frontiersin.org/computational_neuroscience/10.3389/neuro.10/023.2009/abstract

function [H, Network_samples] = SORN(Network, Input, sampling_rate)

% Network Parameters:

N = Network.N;  % network size;

% Network.W0  = initial configuration of EE weights;
% Network.Wei = configuration of IE weights (stays stable);
% Network.Wei = configuration of EI weights (stays stable);
% Network.T0  = initial configuration of E thresholds;

% Network.STDP     = 1 if STDP is on, 0 otherwise; 
% Network.eta_STDP = time scale STDP rule;
% Network.IP       = 1 if IP is on, 0 otherwise; 
% Network.eta_IP   = time scale IP rule;
% Network.rate_IP  = IP stabilizes the mean firing rate of each neuron towards a mean value rate_IP;

% Network.u         = input drive;
% Network.U_neurons = input neurons;

% Input   = vector of discrete values e.g. [1,2,2,3,1,3];
% 0       = no input;

trainTime = size(Input,2);
H         = zeros(N,trainTime);  % complete state history

% Initial network structure
W = Network.W0;    % excitatory weights
idx_W = find(W>0); % index initial weights, only these synapses learn
T = Network.T0;    % excitatory thresholds

% Random initial network activation
X       =  zeros(Network.N,1);              % excitatory units
Y       =  zeros(20/100*Network.N,1);       % inhibitory units

idx = 1; % sampling index;
nrSamples = floor(trainTime/sampling_rate);
Network_samples.W = zeros(nrSamples, N,N);
Network_samples.T = zeros(nrSamples, 1,N);

for t = 1:trainTime
    
    lastX  = X;
    
    % Input
    U = zeros(N,1);
    U(Network.U_neurons(Input(t),:))  = Network.u; 

    % Network activation
    X        = sign(sign(W*X-Network.Wei*Y-T+U)+1);   
    Y        = sign(sign(Network.Wie*X-Network.Ti)+1); 
    H(:,t)   = X;   % save network activition

    % Plasticity
    if (Network.IP==1)
             T = T + Network.eta_IP*(X-Network.rateIP);  % intrinsic plasticity
    end;

    if (Network.STDP==1)
            A          = lastX*X';
            delta      = Network.eta_SP*(A'-A);
            W(idx_W)   = W(idx_W) + delta(idx_W);  %additive STDP
            
            W   = min(W, ones(N,N));   % clip weights to [0,1] 
            W   = max(W, zeros(N,N));
            
            W= W./repmat(sum(W,2),1,N); % synaptic normalization
            
            % TAH rule Gutig, Sompolinsky, depends on mu_SP;
            % STDP additive(bimodal weights) or multiplicative(unimodal weights) 
            % delta   = Network.eta_SP*abs(A'-A).*(power(1-W,mu_SP).*A'-power(W,mu_SP).*A);
           
    end
    
    % save network state
    if (mod(t,sampling_rate)==0)
        Network_samples.W(idx,:,:) = W;
        Network_samples.T(idx,:)   = T;
        idx =idx +1;
    end
 
end
