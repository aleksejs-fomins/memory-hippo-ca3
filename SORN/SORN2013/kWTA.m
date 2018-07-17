%
% kWTA.m
%
% out = kWTA(in, k)
%
% in  : real input vector
% k   : number of "winners"
% out : binary output vector
%
% implements k-Winner-Take-All function
%
% Jochen Triesch, October 2005
%

function out = kWTA(in, k)

[sorted, index] = sort(in);
out = zeros(size(in));
out(index(length(in)-k+1:length(in)))=1;
