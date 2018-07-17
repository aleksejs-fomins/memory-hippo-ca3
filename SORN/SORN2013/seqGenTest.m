
% n desired sequence length

function [seq seqU] = seqGenTest(n)

word(1,:) = [1 3 3 3 3 3 4];
word(2,:) = [2 3 3 3 3 3 5];
word(3,:) = [10 3 3 3 3 3 5];
word(4,:) = [10 3 3 3 3 3 4];
word(5,:) = [11 3 3 3 3 3 5];
word(6,:) = [11 3 3 3 3 3 4];
word(7,:) = [12 3 3 3 3 3 5];
word(8,:) = [12 3 3 3 3 3 4];

%repeated letters are treated distinctly
wordU   = word';
wordU(wordU>0) = 1:length(wordU(wordU>0));
wordU   = wordU';

%constructs a sequence of randomly alternated words
m=size(word,2);
seq  = zeros(1,floor(n*m/(m-1)+50)); 
seqU = zeros(1,floor(n*m/(m-1)+50));
rndindex = ceil(rand(n,1)*size(word,1));
%while (length(seq)<n+10) 
for i= 1: (n/(m-1))
   seq(((i-1)*m+1):i*m)  = word(rndindex(i),:); 
   seqU(((i-1)*m+1):i*m) =  wordU(rndindex(i),:); 
end
seq(seq==0) = [];
seqU(seqU==0)=[];
%bring the sequence to the exact length
if (length(seq)>n)
    seq(n+1:end) = [];
    seqU(n+1:end) = [];
end



end