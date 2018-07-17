% Elman task, 2 long sequences, occluder
% inhibitory neurons (Dale's low)
%--------------------------------------------------------------------------

clear all; 
close all;
% load('3X_5t_STDP_IP_quick_5000.mat')
% driveVect4=Result.driveVect4
% driveVect5=Result.driveVect5


NetParams.N             = 200;     % network size  
NetParams.p_connect     = 0.05;     % prob. of a connection between any 2 neurons

% Plasticity parameters:
NetParams.STDP          = 1;        % STDP on/off
NetParams.IP            = 1;        % IP on/off

NetParams.eta_IP        = 0.001;    % IP learning rate
NetParams.eta_SP        = 0.001;    % STDP learning rate
NetParams.mu_SP         = 0;        % mu = 0 additive STDP rule, unstable,bimodal
                                    % mu = 1 multiplicative, stable, unimodal 
                                    
% Input parameters:
NetParams.input_p         = 10;                       % N^U nr neurons for each input symbol 
NetParams.u               = 1;                        % input stregth;
NetParams.wordLength      = 12;                       % word length   abbbbbbbbbc === n+2
NetParams.k               = 2*NetParams.input_p;      % rate H_IP 
NetParams.trainTime       = 50000;

% Threshold settings give initial rate
NetParams.Ti_max = 1 %2; %0.6    % low Ti  -> high inhib -> low rates
                            % high Ti -> low inhib  -> high rates 
NetParams.T_max  = 0.5 %1;%0.3

for index_run =1:3

% Inhibitory interneurons
N                = NetParams.N;
Ninhib           = N * 0.2;            % number of inhibitory units

step = NetParams.Ti_max/Ninhib;
NetParams.Ti = [0.5:Ninhib]'*step;     %(step/2:step:Tmax)'; 


Wie             = unifrnd(0, 1, Ninhib,N);      % excitatory- inhibitory conections 
Wie             = Wie./repmat(sum(Wie,2),1,N);  % scaled 

Wei             = unifrnd(0, 1, N, Ninhib);          % inhibitory-excitatory connections
Wei             = Wei./repmat(sum(Wei,2),1,Ninhib);  % scaled 

NetParams.Wei           = Wei;
NetParams.Wie           = Wie;

% Excitatory neurons 
step = NetParams.T_max/N;
NetParams.T    = [0.5:N]'*step; 
NetParams.T    = NetParams.T(randperm(N));

for i = 1:N
    W0(:,i)      = unifrnd(0, 0.2, 1, N).*(unifrnd(0.0, 1.0, 1, N) < (NetParams.p_connect));
end

% %strange:
% for i = 2:N
%     W0(:,i)      = unifrnd(0, 0.2, 1, N).*(unifrnd(0.0, 1.0, 1, N) < (NetParams.p_connect/2));
%     %W0(:,i)      = repmat( [0.2], 1, N).*(unifrnd(0.0, 1.0, 1, N) < (NetParams.p_connect/2));
% end
% for i = 1:N
%     W0(i,:)      = max(W0(i,:),unifrnd(0, 0.2, 1, N).*(unifrnd(0.0, 1.0, 1, N) < (NetParams.p_connect/2)));
%     %W0(i,:)      = max(W0(i,:),repmat([0.2], 1, N).*(unifrnd(0.0, 1.0, 1, N) < (NetParams.p_connect/2)));
% end

W0(1:N+1:end)   = 0;        % set diagonal elements to zero (linear indexing)
W0              = W0./(repmat(sum(W0,2),1,N)); %scale
NetParams.W0    = W0;
NetParams.index_plastic   = find(NetParams.W0>0);  %index of initial weights (affected by plasticity)
 
%--------------------------------------------------------------------------
   tic

    % TRAINING WITH PLASTICITY 

    [seqTrain(1,:)  seqTrain(2,:)] =  seqGen(NetParams.trainTime);  % generate input

    toc
    [H ,Wvect,Tvect,Lvect]  =  runNet_newSTDP(seqTrain,NetParams.trainTime, 1000, NetParams,NetParams.STDP,NetParams.IP); 
    toc
    NetParams.nrLetters =  max(seqTrain(:,1:200)');  %nr of input letters (used in Test) 
    
    
    % Figures 1 and 2
    W = squeeze(Wvect(end,:,:));
    T = Tvect(end,:)';
   
    figure(1)
    subplot(2,1,1)
    imagesc(H.Xvect)
    subplot(2,1,2)
    imagesc(H.Rvect)
    ylabel('Unit');
    xlabel('Activity')
    L=Lvect;
  
    figure(2)
    subplot(2,4,1)
    weights = reshape(Wie, size(Wie,1)*size(Wie,2), 1);
    hist(weights,20);
    xlabel('weight E-I');
    ylabel('frequency');
   
    subplot(2,4,2)
    weights = reshape(Wei, size(Wei,1)*size(Wei,2), 1);
    hist(weights,20);
    xlabel('weight I-E');
    ylabel('frequency');

    % scatter plot of thresholds vs. sum of incoming weights of a neuron
    subplot(2,4,3)
    sumOfIncomingWeights = sum(Wie,2);
    scatter(sumOfIncomingWeights, NetParams.Ti);
    xlabel('incoming weights (Wie)');
    ylabel('threshold Ti');
    % scatter plot of thresholds vs. sum of incoming weights of a neuron
    subplot(2,4,4)
    sumOfIncomingWeights = sum(W,2)- sum(Wei,2);
    scatter(sumOfIncomingWeights, T);
    xlabel('incoming weights Wee-Wei');
    ylabel('threshold Te');
    
    % plot W hist before and after learning
    subplot(2,4,5)
    hist(reshape(NetParams.W0,N*N,1),20)
    xlabel('weight W0');
    ylabel('frequency');
    
    subplot(2,4,6)
    hist(reshape(W,N*N,1),20)
    xlabel('weight W');
    ylabel('frequency');

    % plot E-E weights before and after learning
    subplot(2,4,7)
    imagesc(NetParams.W0,[0,1]);
    xlabel('weight W0');
    subplot(2,4,8)
    imagesc(W,[0,1]);
    xlabel('weight W');

    
    
    % save trained weights and thresholds 
    Result.Wvect     = Wvect;
    Result.Tvect     = Tvect;
    Result.NetParams = NetParams;
    
    %Wchange(Wvect,[1:5:500,500],500,Result)
    
    % TEST
    indexOfInterest  = [0,10:20:50] %, 10:10:50];%[2:2:10,20:10:Result.NetParams.trainTime/1000];
    %indexOfInterest  = [0,50];
    
    % M = 4;
    % N = 5;
    [drive4, drive5, fit1] = similarity(Result, indexOfInterest);
    
    driveVect4(index_run,:,:) =  drive4
    driveVect5(index_run,:,:) =  drive5
end
Result.driveVect4 =driveVect4;
Result.driveVect5 =driveVect5;

figure()
plot(squeeze(mean(driveVect4(:,1,:),1))); hold on;
plot(squeeze(mean(driveVect5(:,1,:),1)))
plot(squeeze(mean(driveVect4(:,4,:),1)),'r'); hold on;
plot(squeeze(mean(driveVect5(:,4,:),1)),'r')

