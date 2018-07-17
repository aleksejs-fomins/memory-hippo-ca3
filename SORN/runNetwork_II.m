%function features=runNetwork(N,TEmax,TImax,Wmax,lagVect)    
clear all; close all;

for run= 1:1
    
    N=200;
   
    lagVect = -10:1:10;  % input shift => memory or prediction

    %% declare parameters 

    NE = 80/100*N; %number of excitatory neurons
    NI = 20/100*N; %number of inhibitory neurons
    indexE = 1:NE;
    indexI = NE+1:N;
    
    Time = 50000; % ms
    
    pEE = 0.05;
    pIE = 1;
    pEI = 0.2; 
    pII = 1; % prob of connection between inhibitory-inhibitory neurons
    
    
    % set initial firing rates
    TEmax = 0.75;
    TImax = 0.2;
    T  = [rand(1,NE)*TEmax rand(1,NI)*TImax]';  %threshold values for each neuron
    T0 = T;
    
    % firing rates with IP
    rateE = 0.1;  %target value for IP rule
    rateI = 0.3;
    H_IP    = [repmat(rateE,NE,1); repmat(rateI,NI,1)];
    
    eta_IP  = 0.001;  % time scale IP
    eta_SP  = 0.001;  % time scale SP
    
   
    %% synaptic connections matrix
    
    % Wij  - connection from i to j
    W       = zeros(N,N);
    Wmax    = 0.5;
    
    W(indexE,indexE)  =  Wmax*rand(NE,NE).*(rand(NE,NE)<pEE);
    index_plastic     = (W>0);  % learning restricted to positive EE synapses
    
    W(indexE,indexI)  =  Wmax*rand(NE,NI).*(rand(NE,NI)<pEI);
    W(indexI,indexE)  =  Wmax*rand(NI,NE).*(rand(NI,NE)<pIE);
    W(indexI,indexI)  =  Wmax*rand(NI,NI).*(rand(NI,NI)<pII);

    W(indexI,:) = -W(indexI,:);

    % synaptic normalization
    normE = 1;
    normI = 1/2;
    W(indexE,:) = W(indexE,:)./repmat(sum(W(indexE,:),1),NE,1)*normE;
    W(indexI,:) = -W(indexI,:)./repmat(sum(W(indexI,:),1),NI,1)*normI;

    W(isnan(W)) = 0 ;
    
    W0 = W;
 

    %% two input sequences: abc, def

    input1 = randperm(Time);
    input1(Time/10+1:end)=[]; % 10th of the time, input1 is present
    
    input2 = randperm(Time);
    input2(Time/10+1:end)=[]; % 10th of the time, input2 is present

    % U input drive
    
    U   = zeros(N,Time);  
    nrU = 10; seqLength = 3;
    
    NU  =  nrU*seqLength*2;
    
    for j = 1:seqLength
        U((j-1)*nrU+1:j*nrU, input1+j) = 1;
        U(((j-1)*nrU+1:j*nrU)+nrU*seqLength, input2+j) = 1;
    end
        
    U(:,Time+1:end) = [];
   
    indexRes = NU+1:NE;
      

    %% output vector
    % out = input1&input2

    aux  = max(sum(U(1:NU/2,:))>0,(sum(U(NU/2+1:NU,:))>0)*2);
    
    
    
    output = zeros(length(lagVect),Time);
    for j = 1:length(lagVect) %j is the delay used to look into the past

          output(j,(max(lagVect)+1:(Time-max(lagVect))))= aux((max(lagVect)+1:(Time-max(lagVect)))+lagVect(j));
       
    end
    

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    %% network dynamics
    
    	   % the activity matrix
    X = zeros(N, Time);   % the activity matrix
    Xfade   = zeros(N, Time);   % the activity matrix + perturbations
    dHamm   = zeros(1,Time); % the Hamming distance vector = criticality vector

        
    for t = 2:Time

        
        X(:,t) = ((X(:,t-1)' * W)' + U(:,t-1))>T;
        
        if t>10000
            T(indexRes) = T(indexRes) + eta_IP*(X(indexRes, t-1)-H_IP(indexRes));  % intrinsic plasticity
        end 
        
        if t>20000
            
            %additive STDP
            A                = X(:,t-1)*X(:,t)';  
            delta            = eta_SP*(A-A');
            W(index_plastic) = W(index_plastic) + delta(index_plastic) ; 
            
            % clip weights to [0,1] --> still necessary
            % W(index_plastic) = min(W(index_plastic),1);   
            W(index_plastic) = max(W(index_plastic), 0); 
            
            % synaptic normalization
            W(indexE,:) = W(indexE,:)./repmat(sum(W(indexE,:),1),NE,1)*normE; 
            %W(indexI,:) = -W(indexI,:)./repmat(sum(W(indexI,:),1),NI,1);
        end
        
        flip    = ceil(rand*length(indexRes)); 
        Xp      = X(:,t-1);
        Xp(indexRes(flip)) = not(Xp(indexRes(flip))); % flip reservoir bit at time t-1
        Xp      = ((Xp'*W)' +U(:,t-1))>T;
        
               
        dHamm(t)= sum(abs(Xp-X(:,t))); % Hamming distance at time t
                                            
    end
    
    
    dHamm(1) = dHamm(2);  % dHamm(1) is not defined
      
   
        
    %% Bayes classifier performance 
%     perf = zeros(3,length(lagVect));
    idx  = 5001:10000; 
    idx2 = 15001:20000;
    idx3 = 25001:30000;
%     for i=1:length(lagVect)
%         perfRnd(i)  = BayesClassif(X(indexRes,idx)', output(i,idx));
%         perfIP(i)   = BayesClassif(X(indexRes,idx2)', output(i,idx2));
%         perfSORN(i) = BayesClassif(X(indexRes,idx3)', output(i,idx3));
%     end
    
 
    %% FIGURES

     
    figure()
    
    % network activity
    subplot(5,2,1:2)
    imagesc(X)
    
    
    % Hamming distance
    subplot(5,2,3:4)
    plot(mean(reshape(dHamm,250,Time/250)))
    
    % Performance on Memory/Prediction task
%     subplot(5,2,5)
%     plot(perfRnd','k'); hold on;
%     plot(perfIP','g'); hold on;
%     plot(perfSORN','r')
%     ylim([0.4,1])
    
    
    % Rate excitatory/inhibitory
    subplot(5,2,6)
    bar([mean(mean(X(indexE, idx))),mean(mean(X(indexE,idx2))),mean(mean(X(indexE,idx3)));mean(mean(X(indexI,idx))),mean(mean(X(indexI,idx2))),mean(mean(X(indexI,idx3)))]')
    
    % Thresholds
    subplot(5,2,7:8)
    plot(T,'r'); hold on;
    plot(T0,'k')
    
    % Weights
    subplot(5,2,9)
    hist(W0(index_plastic)); hold on;
    subplot(5,2,10)
    hist(W(index_plastic))
    
        
    % network rate
    mean(mean(X(indexE,:)))
    mean(mean(X(indexI,:)))
    
    figure()
    index=9000:9100;
    plot(mean(X(indexE, index)),'-o'); hold on;
    plot(mean(X(indexI, index)),'-rs');
    
end
    
    
%end
    