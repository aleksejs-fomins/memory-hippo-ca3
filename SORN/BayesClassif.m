% Bayes classifier

function performance = BayesClassif(M,Out)
%M=X;
%Out=O;
  idxDel = Out==0; 

  M(idxDel,:) = [];
  Out(idxDel) = [];
  
  class1=1;
  class2=2;
  
    index1  = find(Out==class1);
    index2  = find(Out==class2);
    [nr,index] = min([sum(Out==class2),sum(Out==class1)]);
    aux = max([sum(Out==class2),sum(Out==class1)])-min([sum(Out==class2),sum(Out==class1)]);
    if (index==1)
        indexDel = randperm(length(index2)); indexDel(aux+1:end)=[];
        Out(index2(indexDel)) = [];
        M(index2(indexDel),:) = [];
    else
        indexDel = randperm(length(index1)); indexDel(aux+1:end)=[];
        Out(index1(indexDel)) = [];
        M(index1(indexDel),:) = [];
    end
    
    Time = size(Out,2);
    indexTrain = 1:round(Time*3/4);
    indexTest = round(Time*3/4)+1:Time;

    %removes the input neurons
    M(:,1:20)=[];
    %removes the neurons that do not change their activity
    idxdel = or(or(std(M(Out(indexTrain)==class2,:))==0,std(M(Out(indexTrain)==class1,:))==0),or(std(M(Out(indexTest)==class2,:))==0,std(M(Out(indexTest)==class1,:))==0));
    M(:,idxdel)=[]; 
  
%     nb = NaiveBayes.fit(double(M(indexTrain,:)),Out(indexTrain));
    nb = fitcnb.fit(double(M(indexTrain,:)),Out(indexTrain));

    pre = predict(nb,double(M(indexTest,:)));

    performance = mean(pre == Out(indexTest)');
    
end