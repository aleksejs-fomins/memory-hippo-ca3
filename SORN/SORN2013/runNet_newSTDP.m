function [Hout,W0,T0,Lyap] = runNet_newSTDP(input, traintime, wndsnapshot, NetParams, STDP, IP)

N             = NetParams.N; 
k             = NetParams.k;
input_p       = NetParams.input_p;
u             = NetParams.u;
eta_IP        = NetParams.eta_IP;
eta_SP        = NetParams.eta_SP;
mu_SP         = NetParams.mu_SP;
W             = NetParams.W0;
index_plastic = NetParams.index_plastic;
T             = NetParams.T;
Wei           = NetParams.Wei;
Wie           = NetParams.Wie;
Ti            = NetParams.Ti;



Xvect       = zeros(N, traintime); % complete state history
Rvect       = zeros(N, traintime); % complete state history
%Svect       = zeros(N, traintime); % complete state history
%Mvect       = zeros(N, traintime); % complete state history

Y       =  zeros(size(Ti));  % inhibitory units

%initiate 1 random state with k neurons active
X         =  kWTA(rand(N, 1), k);      % excitatory units

%define subpopulations receiving input
pop = zeros(input_p,max(input(1,:)));
for ID  =  1:size(pop,2)
    pop(:,ID)   = ((ID-1)*input_p+1 : ID*input_p)';
end

snapshot_idx = 0;
nrsnapshots  = round(traintime/wndsnapshot);
W0   = zeros(nrsnapshots,N,N);
T0   = zeros(nrsnapshots,N);
Lyap = zeros(nrsnapshots,1);
HammingDist  = 0;
ONES  = ones(size(index_plastic));
ZEROS = zeros(size(index_plastic));

for t = 1:traintime
    
    last  = X;
    
    % if input is not 0 the corresponding neurons receive the drive u
    I = zeros(N,1);
    if (input(1,t)>0)  
        if (input(1,t)==10)
            input_neurons = [pop(1:2,1) ;pop(3:10,2)];
        end
        if (input(1,t)==11)
            input_neurons = [pop(1:5,1) ;pop(1:5,2)];
        end
        if (input(1,t)==12)
            input_neurons = [pop(1:8,1) ;pop(9:10,2)];
        end
        if (input(1,t)<10)
            input_neurons = pop(:,input(1,t));
        end
        I(input_neurons)  = u; 
    end

%     % 1 bit perturbation for vect X
%     pX        = X;
%     idx       = floor(1 + N* rand(1,1));  % value between 1 and N
%     pX(idx)   = not(last(idx));
%     pNext     = sign(sign(W*pX -Wei*Y-T+I)+1);
    
    % Network activation
    %Svect(:,t)   = W*X-Wei*Y-T;
    %Mvect(:,t)   = W*X-Wei*Y;
    Rvect(:,t)   = sign(sign(W*X-Wei*Y-T)+1);  
    X            = sign(sign(W*X-Wei*Y-T+I)+1);    
    Y            = sign(sign(Wie*X-Ti)+1);   
  
    Xvect(:,t)   =  X;

    % perturbation at next state
    %HammingDist        = HammingDist + sum(xor(X,pNext));   

    %PLASTICITY
    
    if (IP==1)
             T = T + eta_IP*(X-k/N);  % intrinsic plasticity
    end;
    
    if (STDP==1)
            A = last*X';
            
            % STDP additive(bimodal) or multiplicative(unimodal) weights  - TAH rule Gutig, Sompolinsky
            %asym = eta_SP*abs(A'-A).*(power(1-W,mu_SP).*A'-power(W,mu_SP).*A);
            %W(index_plastic) = W(index_plastic) + asym(index_plastic);
            
            aux=eta_SP*(A'-A);
            W(index_plastic) = W(index_plastic) +aux(index_plastic) ;
            
            W(index_plastic) = min(W(index_plastic), ONES);   % clip weights to [0,1] --> still necessary
            W(index_plastic) = max(W(index_plastic), ZEROS);
            
            aux = sum(W,2);
            aux(aux==0) = 1;
            W= W./repmat(aux,1,N); % synaptic scaling (brutal)
  end

    %save weights and thresholds at snapshot times
    if (mod(t,wndsnapshot)==0)
        snapshot_idx             = snapshot_idx+1;
        W0(snapshot_idx,:,:)     = W;
        T0(snapshot_idx,:)       = T;
        %Lyap(snapshot_idx)       = HammingDist/wndsnapshot;  %mean distance
        %HammingDist              = 0;
    end
end


Hout.Xvect  = Xvect;
Hout.Rvect  = Rvect;  
%Hout.Svect  = Svect;  
%Hout.Mvect  = Mvect;  

