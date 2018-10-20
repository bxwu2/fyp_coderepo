close all
clear all
clc

% Barry Wu
% 2599 3534
% Date created: 13-08-2018
% Date modified: 13-08-2018

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

%% EXPERIMENTAL Cover Time (histogram)
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



counterTrials = zeros(1,numTrials);
for trials = 1:numTrials
    node = nodeSrc;
    counterNode = zeros(1,n);
    counterNode(node)= 1;
    counterCover = 0;
    while (~counterCover)
         k = find(1:n==node);
%              if (node==k)    % if at this node
              p = randi([1,numNeighbor(k)]);  % pick random neighbor node
              node = arrNeighbor(p,k);        % hop there
              counterNode(node) = counterNode(node) +1;   % number of hops so far
              counterTrials(trials) = counterTrials(trials) + 1;
              if (~sum(~counterNode))
                  counterCover = 1;
              end

            %             dataY(idxData) = node;     % for plotting
            %             idxData = idxData + 1;      
        
    end 
end


uniqueHops = unique(counterTrials);
numHopsU = length(uniqueHops);
% hist(counterNode,numHopsU+1)

uniqueArr = zeros(1,numHopsU);
for uniqueIdx = 1:numHopsU
    uniqueMatch = uniqueHops(uniqueIdx)== counterTrials;
    uniqueMatchCount = sum(uniqueMatch);
    uniqueArr(uniqueIdx) = uniqueMatchCount;
end

subplot(1,2,2)
bar(uniqueHops,uniqueArr/numTrials);
title('Histogram for Number of Hops')
ylabel('Probability')
xlabel('t (hops)')    
xticks(0:2:100)



%% Comparing Experimental with Simulation/Formula for PDF
figure(2)
bar(uniqueHops,uniqueArr/numTrials);
uniqueProb = uniqueArr/numTrials;
filler = [0:uniqueHops(1)-1];
uniqueHops = [filler uniqueHops];
uniqueProb = [zeros(1,length(filler)), uniqueProb];
plot(uniqueHops,uniqueProb,'--o');
title('Experimental v Simulation for Cover PDF')
ylabel('Probability')
xlabel('t (hops)')
xticks(0:2:100)
hold on


%% Comparing Experimental with Simulation/Formula for CDF
figure(3)
uniqueProbCDF = uniqueProb;
for idx = 1:length(uniqueHops)-1
    uniqueProbCDF(idx+1) = uniqueProbCDF(idx)+uniqueProbCDF(idx+1);
end

plot(uniqueHops,uniqueProbCDF,'--o');
title('Experimental vs Simulation for Cover CDF')
ylabel('Probability')
xlabel('t (hops)')
xticks(0:4:100)
hold on

%% Formula for Cover PDF

% CREATE TRANSITION MATRIX
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

t = 100;

hitCDF= zeros(n,t+1);
hitPDF = zeros(n,t+1);
%% Term 1
for nodeDst = 1:n
    if (nodeDst == nodeSrc)
        continue
    end
    
    % create D matrix
    D = transMatrix;
    D(nodeDst,:) = 0;
    D(nodeDst,nodeDst) = 1;

    % create D^t matrix
    t = 100; % end time

    for tCDF = 1:t
        Dt = D^tCDF;
        hitCDF(nodeDst,tCDF+1) = Dt(nodeSrc,nodeDst);
    end

    % find PDF of Hit Time
    
    for tPDF = 1:t
        hitPDF(nodeDst,tPDF+1) = hitCDF(nodeDst, tPDF+1)-hitCDF(nodeDst,tPDF);
    end
end

% Term 1
cover1 = sum(hitCDF); % sum of hit times from src to all nodes (i=/=z); vector contains sum for t hops (t entries in vector)
cover_test = cover1(2:end);
%% Term n
%     cover_n = zeros(n-2,t);
    combSum=zeros(1,t);
    for unionIdx = 2:n-1
        nchoosek(1:n,unionIdx);
    % -------------------------------------------
        combArr = nchoosek(1:n,unionIdx);
%             combSum = 0;

            for combIndex = 1:size(combArr,1)
                testSrc = ismember(combArr(combIndex,:),nodeSrc); % exclude nodeSrc
                if(sum(testSrc))
                    continue
                end
    % -------------------------------------------
                transMatrixAbs = transMatrix;
                transMatrixAbs(combArr(combIndex,:),:)= 0;
                absIdx = sub2ind(size(transMatrix),combArr(combIndex,:),...
                    combArr(combIndex,:));
                transMatrixAbs(absIdx) = 1;
                
                %% 
                for coverIdx = 1:t
                    transMatrixT = transMatrixAbs^coverIdx;
                    combProbs = transMatrixT(nodeSrc,combArr(combIndex,:)); % asking supervisor
                    combSum(coverIdx) = combSum(coverIdx) +(-1)^(unionIdx-1)*sum(combProbs);
                end
            end
    end
    
    %% Last term
    transMatrixAbs = eye(n);
    combEnd=zeros(1,t);
    for sumIdx = 1:n
        for endIdx = 1:t
            transMatrixT_end = transMatrixAbs^endIdx;
            combEnd(endIdx) = combEnd(endIdx) + transMatrixT_end(nodeSrc,sumIdx);
        end
    end
    
    %% Total Cover PDF sum
    
    figure(4)
    coverTotal = cover_test + combSum +(-1)^(n-1)*ones(size(cover_test));
    plot(coverTotal)