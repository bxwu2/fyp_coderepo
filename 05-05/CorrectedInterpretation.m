close all
% clear all
clc

% Barry Wu
% 2599 3534
% Date created: 17-04-2018
% Date modified: 30-04-2018

% Hit Time

%% GENERATE NETWORK OF n NODES
% arrControl = [ 0 0 0 1 1 0; 0 0 0 0 1 0; 0 0 0 1 1 0; 1 0 1 0 1 0; 1 1 1 1 0 1; 0 0 0 0 1 0];
arrControl=[0 0 0 1 1 1; 0 0 0 0 1 1; 0 0 0 1 0 1; 1 0 1 0 0 0; 1 1 0 0 0 0; 1 1 1 0 0 0]';
controlCount = 0;
controlFlag = 1;

while controlFlag >0
    n = 6;

A = randi([0,1],n); % generate matrix

Atriag = triu(A,1); % upper triangular (remove diagonals)
M = Atriag + Atriag'; % create adjacency matrix

controlFlag = sum(sum(arrControl ~= M))
controlCount = controlCount + 1
end

prevM = M;
i = 0;

G = graph(M); % initialise network graph
numEdges = numedges(G); % intialise number of edges (to ensure no multiple networks)
while (sum(~sum(M))||numEdges<n-1)	% while there are no empty columns/rows (no isolated nodes), no multiple networks with the numEdges rule
% while (sum(~sum(M)))||(numEdges<n-1)
    i=i+1;  
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
    Atriag = triu(Atriag,1);
    prevM = M;
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

%% HIT TIME

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
arrNeighborNum = sum(arrNeighbor~=0);

nodeDst = randi([1,n]); % node to hit
nodeSrc = randi([1,n]); % node to start from

while (nodeDst == nodeSrc)  % to ensure desired node is not same as start
    nodeDst = randi([1,n]);
end

numTrials = 1000;
counterNode = zeros(1,numTrials);
for trials = 1:numTrials
    node = nodeSrc;
    counterHit = 0;

    while (~counterHit)
        for k = 1:n
            if (node==k)    % if at this node
              p = randi([1,numNeighbor(k)]);  % pick random neighbor node
              node = arrNeighbor(p,k);        % hop there
              counterNode(trials) = counterNode(trials) +1;   % number of hops so far
              if (node == nodeDst)
                  counterHit = 1;
              end

            %             dataY(idxData) = node;     % for plotting
            %             idxData = idxData + 1;      
            end
        end
    end 
end

subplot(1,2,2)
numHopsU = length(unique(counterNode));
hist(counterNode,numHopsU)
title('Hit Time')

%%  Computes Unique Hop Counts & Frequency of that Hop Count
% numHopsUnique = unique(counterNode);       % temp vector of vals
% numHopsSort = sort(counterNode);          % sorted input aligns with temp (lowest to highest)
% numHopsArr = zeros(size(numHopsUnique)); % vector for freqs
% % frequency for each value
% for histo = 1:length(numHopsUnique)
%     numHopsArr(histo) = sum(counterNode == numHopsUnique(histo));
% end
% 
% figure(2)
% plot(numHopsUnique, numHopsArr, 'b-')

%% Hit Time

% create Transition Matrix

% create transitional probabilities
arrHopProb = zeros(1,n);
for probHop = 1:n
    arrHopProb(probHop) = 1/arrNeighborNum(probHop);
end

% create transition matrix
transMatrix = zeros(n);
for transHop = 1:n
    for transHopNeighbor = 1:arrNeighborNum(transHop)
        transMatrix(transHop,arrNeighbor(transHopNeighbor,transHop))=arrHopProb(transHop);
    end
end

% create D matrix
D = transMatrix;
D(nodeDst,:) = 0;
D(nodeDst,nodeDst) = 1;

% create D^t matrix
t = 100;
hitCDF= zeros(1,t+1);
for tCDF = 1:t
    Dt = D^tCDF
    hitCDF(tCDF+1) = Dt(nodeSrc,nodeDst);
end

% plot CDF of Hit Time
figure(2)
subplot(1,2,1)
plot(0:t,hitCDF,'*')
title('CDF of Hit Time')
ylabel('Probability')
xlabel('t (hops)')

% find PDF of Hit Time
hitPDF = zeros(1,t+1);
for tPDF = 1:t
    hitPDF(tPDF+1) = hitCDF(tPDF+1)-hitCDF(tPDF);
end
subplot(1,2,2)
plot(0:t,hitPDF,'*')
title('PDF of Hit Time')
ylabel('Probability')
xlabel('t (hops)')    
   


% write high level description for cover


