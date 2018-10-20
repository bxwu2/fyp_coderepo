close all
clear all
clc

n = 10;

% d = randi([0,1],[n,1]);
A = randi([0,1],n); % generate matrix
% NO NEED FOR DIAG????? diag is for repeats

Atriag = triu(A,1);
M = Atriag + Atriag';
% at least 2 rows need at least 2
% no empty column/row
i = 0;

G = graph(M);
numEdges = numedges(G);
while (sum(~sum(M)))&&(numEdges<n-1)
    zeroA = zeros(n);
    indZero = find(~sum(M));
    zeroA(indZero,randi([1,n])) = 1;
    zeroUp = triu(zeroA,1);
    zeroUpSum = sum(sum(zeroUp));
        if (~zeroUpSum)
            Atriag = Atriag + zeroA';
        else
            Atriag = Atriag + zeroA;
        end
    Atriag = triu(Atriag,1)
    M = Atriag + Atriag'
    G = graph(M);
    i=i+1
end

% Atriag1 = triu(A,1)
% M1 = Atriag1+Atriag1'
% Atriag1(randi([1,n]),indZero)=1
% Atriag1 = triu(Atriag1,1)
% M2 = Atriag1+Atriag1'

% sumAtriag = sum(Atriag(:,2:end));
% while (nnz(~sumAtriag)>1)
%     fprintf('here we go\n')
%     indAtriag = find(~sumAtriag);
%     pickAtriag = indAtriag(randi([1,nnz(~sumAtriag)]));
%     Atriag(pickAtriag,randi([1,n])) = 1;  
%     sumAtriag = sum(Atriag(:,2:end));
% end

% M = Atriag + Atriag'
G = graph(M);
plot(G)
title('Generated Network of Nodes')


