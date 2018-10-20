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

%% EXPERIMENTAL COMMUTE TIME
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
    counterRet = 0;

    while (~counterRet)
         k = find(1:n==node);
%              if (node==k)    % if at this node
              p = randi([1,numNeighbor(k)]);  % pick random neighbor node
              node = arrNeighbor(p,k);        % hop there
              counterNode(trials) = counterNode(trials) +1;   % number of hops so far
              if ((node == nodeDst)&&(~counterHit))
                  counterHit = 1;
              end
              
              if (counterHit)   % IF ON THE WAY BACK
                  if (node == nodeSrc)  % returns home!!
                      counterRet =1; %END LOOP
                  end
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
    uniqueMatch = (uniqueHops(uniqueIdx)==counterNode);
    uniqueMatchCount = sum(uniqueMatch);
    uniqueArr(uniqueIdx) = uniqueMatchCount;
end

bar(uniqueHops,uniqueArr/numTrials);
title('Histogram for Number of Hops')
ylabel('Probability')
xlabel('t (hops)')    
xticks(0:2:100)

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

D2 = transMatrix;
D2(nodeSrc,:) = 0;
D2(nodeSrc,nodeSrc) = 1;


% create D^t matrix


%% CDF of Simulated Commute Time

% Dstar matrix
Dstar = transMatrix;
Dstar(nodeDst,:) = 0;
% Oj matrix
Oj = zeros(n);
Oj(nodeDst,nodeDst) = 1;
% O matrix
O = zeros(n);
% Di matrix
Di = transMatrix;
Di(nodeSrc,:) = 0;
Di(nodeSrc,nodeSrc) = 1;
% create C matrix
C = [Dstar Oj; O Di];

% create C^t matrix
t = 100;
commuteCDF= zeros(1,t);
for tCDF = 1:t
    Ct = C^tCDF;
    commuteCDF(tCDF) = Ct(nodeSrc,nodeSrc+n);
end


%% COMPARING EXPERIMENTAL WITH SIMULATION PDF
figure(2)
% bar(uniqueHops,uniqueArr/numTrials);
uniqueProb = uniqueArr/numTrials;
plot(uniqueHops,uniqueProb,'--o');
title('Simulation vs Mathematical Commute PDF')
ylabel('Probability')
xlabel('t (hops)')
xticks(0:2:100)
hold on

% CONVOLUTION

% find CDF of Hit Time 
hitCDF= zeros(1,t);
hitCDFij = hitCDF;
hitCDFji = hitCDF;
for tCDF = 1:t
    Dt = D^tCDF;
    Dt2 = D2^tCDF;
    hitCDFij(tCDF+1) = Dt(nodeSrc,nodeDst);
    hitCDFji(tCDF+1) = Dt2(nodeDst,nodeSrc);
end
% find PDF of Hit Time (h_ij)
hitPDF = zeros(1,t);
for tPDF = 1:t
    hitPDFij(tPDF+1) = hitCDFij(tPDF+1)-hitCDFij(tPDF);
    hitPDFji(tPDF+1) = hitCDFji(tPDF+1)-hitCDFji(tPDF);
end
% nonZeroij = ~(hitPDFij==0);
% nonZeroji = ~(hitPDFji==0);
% 
% hitPDFij2 = hitPDFij(nonZeroij);
% hitPDFji2 = hitPDFji(nonZeroji);
% convolve h_ij * h_ji
% convPDF = zeros(1,length(hitPDFij2)+1);
% for idxPDF = 1:length(hitPDFij2)
%     for idxConv = 1:idxPDF-1
%         convPDF(idxPDF) = convPDF(idxPDF) + hitPDFij2(idxConv)*hitPDFji2(length(hitPDFij2)-idxConv);
%     end
% end
convPDF = conv(hitPDFij(1:end),hitPDFji(1:end));
convPDF(1:length(convPDF)-1) = convPDF(2:length(convPDF));
xConv = 1:length(convPDF);
nonZeroConv = ~(convPDF==0);

plot(xConv(nonZeroConv),convPDF(nonZeroConv),'--*')

legend('Simulation','Formula')




%% COMPARING EXPERIMENTAL WITH SIMULATION CDF
% plot CDF of Commute Time
% get rid of 0 increments for CDF
figure(3)

% GETTING RID OF THE ODD ZEROS (plateaus) 
commutePDF = zeros(1,length(commuteCDF)-1);
for idx = 1:length(commutePDF)-1
    commutePDF(idx+1)=commuteCDF(idx+1)-commuteCDF(idx);      
end
nonZeroPDF = ~(commutePDF==0); 

xCDF = 0:t-1;

plot(xCDF(nonZeroPDF),commuteCDF(nonZeroPDF),'--*')
% title('CDF of Commute Time')
title(sprintf('CDF of Commute_%d_%d',nodeSrc,nodeDst))
ylabel('Probability')
xlabel('t (hops)')
hold on

uniqueProbCDF = uniqueProb;
for idx = 1:length(uniqueHops)-1
    uniqueProbCDF(idx+1) = uniqueProbCDF(idx)+uniqueProbCDF(idx+1);
end
plot(uniqueHops,uniqueProbCDF,'--o');
title('Mathematical vs Simulation for Commute CDF')
ylabel('Probability')
xlabel('t (hops)')
xticks(0:4:100)

legend('Mathematical','Simulation')


