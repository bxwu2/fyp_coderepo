close all
clear all
clc

% Barry Wu
% 2599 3534
% Date created: 17-04-2018
% Date modified: 26-07-2018

%% GENERATE NETWORK OF n NODES
n = 6; % number of nodes in network

% 1 particular network
% 2 random network of size n
choice = 1;
switch choice
    case 1 
        % arrControl = [ 0 0 0 1 1 0; 0 0 0 0 1 0; 0 0 0 1 1 0; 1 0 1 0 1 0; 1 1 1 1 0 1; 0 0 0 0 1 0];
        arrControl=[0 0 0 1 1 1; 0 0 0 0 1 1; 0 0 0 1 0 1; 1 0 1 0 0 0; 1 1 0 0 0 0; 1 1 1 0 0 0]';
        M = arrControl;
           

    case 2
        A = randi([0,1],n); % generate matrix
        Atriag = triu(A,1); % upper triangular (remove diagonals)
        M = Atriag + Atriag'; % create adjacency matrix

        prevM = M;
%         i = 0;
        testCon = 3;
        testCon2 = 1;
        
        while (testCon>2||testCon2)	% while there are no empty columns/rows (no isolated nodes), no multiple networks with the numEdges rule
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
            testCon = nnz(~(sum(M)-1)); % test for disconnectedness
            testCon2 = nnz(~sum(M));
        end
end

G = graph(M);
subplot(1,2,1)
plot(G)
title('Generated Network of Nodes')

%% EXPERIMENTAL HIT TIME
numTrials = 50000;


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


counterNode = zeros(1,numTrials);
for trials = 1:numTrials
    node = nodeSrc;
    counterHit = 0;

    while (~counterHit)
         k = find(1:n==node);
%              if (node==k)    % if at this node
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

subplot(1,2,2)
uniqueHops = unique(counterNode);
numHopsU = length(uniqueHops);
% hist(counterNode,numHopsU+1)

title('Hit Time')

uniqueArr = zeros(1,numHopsU);
for uniqueIdx = 1:numHopsU
    uniqueMatch = uniqueHops(uniqueIdx)== counterNode;
    uniqueMatchCount = sum(uniqueMatch);
    uniqueArr(uniqueIdx) = uniqueMatchCount;
end

bar(uniqueHops,uniqueArr/numTrials);
title('Histogram for Number of Hops')
ylabel('Probability')
xlabel('t (hops)')    
xticks(0:2:100)
hold on

%% SIMULATED HIT TIME

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

%% CDF of Simulated Hit Time
t = 100; % end time

hitCDF= zeros(1,t+1);
for tCDF = 1:t
    Dt = D^tCDF;
    hitCDF(tCDF+1) = Dt(nodeSrc,nodeDst);
end

% plot CDF of Hit Time
% figure(2)
% subplot(1,2,1)
% plot(0:t,hitCDF,'*')
title('CDF of Hit Time')
ylabel('Probability')
xlabel('t (hops)')

%% PDF

% find PDF of Hit Time
hitPDF = zeros(1,t+1);
for tPDF = 1:t
    hitPDF(tPDF+1) = hitCDF(tPDF+1)-hitCDF(tPDF);
end
% subplot(1,2,2)
% plot(0:t,hitPDF,'*')
title('PDF of Hit Time')
ylabel('Probability')
xlabel('t (hops)')    

%% COMPARING EXPERIMENTAL WITH SIMULATION

figure(2)
% bar(uniqueHops,uniqueArr/numTrials);
uniqueProb = uniqueArr/numTrials;
plot(uniqueHops,uniqueProb,'--o');
title('Experimental v Simulation for PMF')
ylabel('Probability')
xlabel('t (hops)')
xticks(0:2:100)
hold on

nonZero = ~(hitPDF==0);
xPDF  = 0:t;
hitPDFomit = hitPDF(nonZero);

plot(xPDF(nonZero),hitPDFomit,'--*')

legend('Experimental','Simulation')

figure(1)
hold on
subplot(1,2,2)
plot(uniqueHops,uniqueProb,'--o')

%% COMPARISON WITH CDF

figure(4)
uniqueProbCDF = uniqueProb;
for idx = 1:length(uniqueHops)-1
    uniqueProbCDF(idx+1) = uniqueProbCDF(idx)+uniqueProbCDF(idx+1);
end

plot(uniqueHops,uniqueProbCDF,'--o');
title('Experimental vs Simulation for CDF')
ylabel('Probability')
xlabel('t (hops)')
xticks(0:4:100)
hold on

hitCDF = hitPDFomit;
for idx = 1:length(xPDF(nonZero))-1
    hitCDF(idx+1) = hitCDF(idx)+hitCDF(idx+1);
end

plot (xPDF(nonZero),hitCDF,'--*')

legend('Experimental','Simulation')

figure(10)
plot(uniqueHops,uniqueProbCDF,'--o');
title('Experimental PMF of Hitting Time')
ylabel('p')
xlabel('Number of hops')



