close all
clear all
clc

% new problem is to make sure we're not excluding any network types
% if sum(column) ==0, then randomly put in a 1 and symmetric it (locate and
% invert the indices?) 
% OR MAYBE columns 2:end need at least 1

n = 5;
numTrials = 1000;
% check = [0 1 0 0 1;  1 0 1 0  0; 0 1 0 1 0 ; 0 0 1 0 1; 1 0 0 1 0];
% all = ones(n) - diag(ones(1,n));
centre = zeros(n);
centre(2:end,1) = 1;
centre(1,2:end) = 1;
for trials = 1:numTrials
    d = randi([0,1],[n,1]); % The diagonal values
    A = randi([0,1],n);
    
        Atriag = triu(A,1);
        for self = 1:n-1
            while sum(Atriag(self,:))==0
                    Atriag(self,randi([self+1,n]))=1;
            end
        end

    % problem is, still get isolated networks 
    M = diag(d) + Atriag+ Atriag';
    for self = 1:n
        M(self,self) =0; 
    end

    % A = randi([0,1],n);
    G = graph(M);
    plot(G)
    numNeighbor = zeros(1,n);
    arrNeighbor = zeros(n);
    for i = 1:n
            N = neighbors(G,i);    % node neighbours
            numNeighbor(i) = length(n); % number of neighbours in node
            for j = 1:numNeighbor(i)    % adding to array of node neighbours
                arrNeighbor(j,i) = N(j);
            end
    end
    if (isequal(M,centre))
        break
    end
end
%    d = randi([0,1],[N,1]); % The diagonal values
%    t = triu(bsxfun(@min,d,d.').*randi(N),1); % The upper trianglar random values
%    M = diag(d)+t+t.'; % Put them together in a symmetric matrix


