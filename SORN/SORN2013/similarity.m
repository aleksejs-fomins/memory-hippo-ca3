

function [drive4 drive5 rw] = similarity(Result,indexOfInterest)
%similarity


NetParams = Result.NetParams;
Wvect     = Result.Wvect;
Tvect     = Result.Tvect;
t=-1;

 
 for t = 1:length(indexOfInterest)
        timeidx = indexOfInterest(t);  
        %Train sequence
        [seqTest1 seqTestUniq1]     =  seqGen(5000);
        %Test sequence = ambigous stimulus
        [seqTest2 seqTestUniq2]     =  seqGenTest(5000);
        
        if (timeidx==0)     %for t==0 we test initial network previuos any training
            [H1,Wvect1,Tvect1,Lvect1]   =  runNet_newSTDP(seqTest1,length(seqTest1),length(seqTest1), NetParams, 0, 0);
            [H2,Wvect2,Tvect2,Lvect2]   =  runNet_newSTDP(seqTest2,length(seqTest2),length(seqTest2), NetParams, 0, 0);
        else
            NetParams.W0 =  squeeze(Wvect(timeidx,:,:)); 
            NetParams.T  =  squeeze(Tvect(timeidx,:)');
            [H1,Wvect1,Tvect1,Lvect1]   =  runNet_newSTDP(seqTest1,length(seqTest1),length(seqTest1), NetParams, 0, 1);
            [H2,Wvect2,Tvect2,Lvect2]   =  runNet_newSTDP(seqTest2,length(seqTest2),length(seqTest2), NetParams, 0, 1);
        
        end
    
     
        indexdel = seqTestUniq1>14; %14
        seqTest1(indexdel)=[];
        H1.Rvect(:,indexdel)=[];

        nrR = max(seqTest1); time=length(seqTest1);
        R  = zeros(nrR, time); 
        R(seqTest1+(0:time-1)*nrR) = 1;
        
        % pseudoinverse -> readout weights 
        indexReservoir  =  NetParams.nrLetters(1)*NetParams.input_p+1:NetParams.N;
        rw  = R*pinv(H1.Rvect(indexReservoir,:));
        
        outputVector       =  rw*H2.Rvect(indexReservoir,:);  % only reservoir units
  
        %[a,i] = max(outputVector);
       
       
        size(outputVector)
        symbvect = [2,10,11,12,1];
        for symbindex=1:5
                symb=symbvect(symbindex);

                indexSymb = find(seqTest2==symb);
                indexSymb = indexSymb+6;  % time steps
                indexSymb(indexSymb>5000)=[];
              
                drive4(t,symbindex) = mean(outputVector(4,indexSymb));
                drive5(t,symbindex) = mean(outputVector(5,indexSymb));

        end
 end

end
        
