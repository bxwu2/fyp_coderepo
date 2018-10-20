close all
% clear all
clc

% Barry Wu
% 2599 3534
% Date created: 17-04-2018
% Date modified: 30-04-2018

% Commute Time

%% GENERATE NETWORK OF n NODES
n = 10;

A = randi([0,1],n); % generate matrix

Atriag = triu(A,1); % upper triangular (remove diagonals)
M = Atriag + Atriag'; % create adjacency matrix

i = 0;

G = graph(M); % initialise network graph
numEdges = numedges(G); % intialise number of edges (to ensure no multiple networks)
while (sum(~sum(M)))||(numEdges<n-1)	% while there are no empty columns/rows (no isolated nodes), no multiple networks with the numEdges rule
    i = i+1;
    zeroA = zeros(n);					% if need to add into the empty column row, construct the matrix
    indZero = find(~sum(M));			% find the index of the empty column/row
    zeroA(indZero,randi([1,n])) = 1;	% add 1 to random index of the empty column/row
    zeroUp = triu(zeroA,1);				
    zeroUpSum = sum(sum(zeroUp));		% check if the 1 landed in upper or lower triangle, because the adjacency matrix is formed using triangle matrices
        if (~zeroUpSum)
            Atriag = Atriag + zeroA';
        else
            Atriag = Atriag + zeroA;
        end
    Atriag = triu(Atriag,1)
    M = Atriag + Atriag';
    G = graph(M);
    numEdges = numedges(G);
    if (sum(sum(prevM == M)))== n^2
        A = randi([0,1],n); % generate matrix

        Atriag = triu(A,1); % upper triangular (remove diagonals)
        M = Atriag + Atriag'; % create adjacency matrix
        prevM = M;
    end 
end

G = graph(M);
subplot(1,2,1)
plot(G)
title('Generated Network of Nodes')

%% COMMUTE TIME

arrNeighbor = zeros(n-1,n); % array for each node's neighbours
numNeighbor = zeros(1,n);   % number of neighbors in each node
% setting up array of node neighbours
    for i = 1:n
        N = neighbors(G,i);    % node neighbours
        numNeighbor(i) = length(N); % number of neighbours in node
        for j = 1:numNeighbor(i)    % adding to array of node neighbours
            arrNeighbor(j,i) = N(j);
        end
    end
nodeDst = randi([1,n]); % node to hit
nodeSrc = randi([1,n]); % node to start from

numTrials = 1000;
counterNode = zeros(1,numTrials);
for trials = 1:numTrials
    node = nodeSrc;
    counterHit = 0;
    counterReturn = 0;

    while (~counterReturn)
        for k = 1:n
            if (node==k)    % if at this node
              p = randi([1,numNeighbor(k)]);  % pick random neighbor node
              node = arrNeighbor(p,k);        % hop there
              counterNode(trials) = counterNode(trials) +1;   % number of hops so far
              if (node == nodeDst)
                  counterHit = 1;
              end
              
              if (counterHit) && (node==nodeSrc)
                  counterReturn = 1;
              end   
            end
        end
    end 
end

subplot(1,2,2)
numHopsU = length(unique(counterNode));
hist(counterNode,numTrials)
% hist(counterNode,numHopsU)
title('Hit Time')

%%  Computes Unique Hop Counts & Frequency of that Hop Count
numHopsUnique = unique(counterNode);       % temp vector of vals
numHopsSort = sort(counterNode);          % sorted input aligns with temp (lowest to highest)
numHopsArr = zeros(size(numHopsUnique)); % vector for freqs
% frequency for each value
for histo = 1:length(numHopsUnique)
    numHopsArr(histo) = sum(counterNode == numHopsUnique(histo));
end

figure(2)
plot(numHopsUnique, numHopsArr, 'b-')