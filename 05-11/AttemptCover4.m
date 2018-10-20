close all
% clear all
clc

% Barry Wu
% 2599 3534
% Date created: 17-04-2018
% Date modified: 09-05-2018

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
% controlFlag=0;
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
numTrials = 1000;

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

% numTrials = 1000;
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

% subplot(1,2,2)
% numHopsU = length(unique(counterNode));
% hist(counterNode,numHopsU)
% title('Hit Time')

%% Cover Time
for trials = 1:numTrials

    counterCover = zeros(1,n); % counting hops to node
    node=nodeSrc; % start at source node
    while (sum(~counterCover)>0)   % while not all nodes covered
        for k = 1:n
            if (node==k)    % if at this node
                p = randi([1,numNeighbor(k)]);  % pick random neighbor node
                node = arrNeighbor(p,k);        % hop there
                counterCover(node) = counterCover(node)+1;       % increase node covered
    %             dataY(idxData) = node;     % for plotting
    %             idxData = idxData + 1;      
            end
        end
    end 

    numHops(trials) = sum(counterCover); % total number of hops
end

% dataX = 1:numHops; % x axis of plot
% figure(2)
% plot(dataX,dataY,'o-');
% title('Node for each hop')
%     fprintf('Cover time = %i hops\n',numHops(trials))
% figure(2)
subplot(1,2,2)
numHopsU = length(unique(numHops));
% hist(numHops,numHopsU)
histogram(numHops,'Normalization','pdf','BinWidth',0.5)

title('Histogram (PDF) for cover time')
ylabel('Frequency')
xlabel('Number of hops')

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
    Dt = D^tCDF;
    hitCDF(tCDF+1) = Dt(nodeSrc,nodeDst);
end

% % plot CDF of Hit Time
% figure(2)
% subplot(1,2,1)
% plot(0:t,hitCDF,'*')
% title('CDF of Hit Time')
% ylabel('Probability')
% xlabel('t (hops)')

% find PDF of Hit Time
hitPDF = zeros(1,t+1);
for tPDF = 1:t
    hitPDF(tPDF+1) = hitCDF(tPDF+1)-hitCDF(tPDF);
end
% subplot(1,2,2)
% plot(0:t,hitPDF,'*')
% title('PDF of Hit Time')
% ylabel('Probability')
% xlabel('t (hops)')    
   
%% Commute Time

% % Dstar matrix
% Dstar = transMatrix;
% Dstar(nodeDst,:) = 0;
% % Oj matrix
% Oj = zeros(n);
% Oj(nodeDst,nodeDst) = 1;
% % O matrix
% O = zeros(n);
% % Di matrix
% Di = transMatrix;
% Di(nodeSrc,:) = 0;
% Di(nodeSrc,nodeSrc) = 1;
% % create C matrix
% C = [Dstar Oj; O Di];
% 
% % create C^t matrix
% % t = 100;
% commuteCDF= zeros(1,t+1);
% for tCDF = 1:t
%     Ct = C^tCDF;
%     commuteCDF(tCDF) = Ct(nodeSrc,nodeSrc+n);
% end
% 
% % plot CDF of Commute Time
% figure(4)
% subplot(1,2,1)
% plot(0:t,commuteCDF,'*')
% % title('CDF of Commute Time')
% title(sprintf('CDF of Commute_%d_%d',nodeSrc,nodeDst))
% ylabel('Probability')
% xlabel('t (hops)')
% 
% % convolution to find PDF of commute
% 
% % find hit_ji
% % create Dji matrix
% Dji = transMatrix;
% Dji(nodeSrc,:) = 0;
% Dji(nodeSrc,nodeSrc) = 1;
% 
% % create Dji^t matrix
% hitCDFji= zeros(1,t+1);
% for tCDF = 1:t
%     Djit = Dji^tCDF;
%     hitCDFji(tCDF+1) = Djit(nodeDst,nodeSrc);
% end
% 
% % % plot CDF of Return Hit Time
% % figure(3)
% % subplot(1,2,1)
% % plot(0:t,hitCDFji,'*')
% % title('CDF of Return Hit Time')
% % ylabel('Probability')
% % xlabel('t (hops)')
% 
% % find PDF of Return Hit Time
% hitPDFji = zeros(1,t+1);
% for tPDF = 1:t
%     hitPDFji(tPDF+1) = hitCDFji(tPDF+1)-hitCDFji(tPDF);
% end
% % subplot(1,2,2)
% % plot(0:t,hitPDFji,'*')
% % title('PDF of Return Hit Time')
% % ylabel('Probability')
% % xlabel('t (hops)')  
% 
% 
% % % now CONVOLVE
% % % sum(tau = 1 to t) hij(tau) hji(t-tau)
% %  commuteConv = zeros(1,t+1);
% %  commuteConvPDF = commuteConv;
% commuteConvPDF = zeros(1,t+1);
% % commuteConv = 0;
%  for commuteHop = 1:t
%      commuteConvSum = 0;
%      for tau = 1:commuteHop
% %          commuteConv = zeros(1,commuteHop+1);
% %          commuteConv(tau+1) = hitPDF(tau+1)*hitPDFji(t-tau+1);
%         commuteConv = hitPDF(tau+1)*hitPDFji(commuteHop-tau+1);
%         commuteConvSum = commuteConvSum + commuteConv;
%      end
% %      commuteConvPDF(tau+1) = sum(commuteConv(tau+1)+commuteConv(tau);
% %     commuteConvPDF(commuteHop+1) = sum(commuteConv);
%         commuteConvPDF(commuteHop+1) = commuteConvSum;
%  end
% figure(4)
% subplot(1,2,2)
%  plot(1:t+1,commuteConvPDF,'*')
%  
%  
% % % figure(5)
% % subplot(2,2,4)
% % plot(1:t+1,commuteConvPDF,'*')
% % title('PDF of Commute Time')
% title(sprintf('PDF of Commute_%d_%d',nodeSrc,nodeDst))
% ylabel('Probability (%)')
% xlabel('t (hops)')
% % % need to test on t*2 hops probably?? because commute > hit time

%% write high level description for cover THIS COMMENT WAS MADE IN mAY 9
% CDF_cover(t) = sum 1 to n (from Src, i =/= src) Fxi(t)
% find hitCDF from Src to ALL n (1 -> n, excl Src)
% sum them all for a value of t

% for each term in equation, +(-1)^(0 to n) for nth term but last term is
% n-1 index?

% 2nd term of equation is - probCDF z to i, times probCDF of z to j
% (???....)

% PERHAPS not needed to purely look at formula
% maybe look at the jotter pad... ONLY CONSIDER NEIGHBOURS? because other
% propbabilities will be 0???

%%
% term1
term1 = zeros(1,t);

for coverIndex = 1:t
%     transMatrixCover = transMatrix^t;
    for coverHop = 1:n
        if (coverHop==nodeSrc)
            continue
        end
        transMatrixAbs1 = transMatrix;
        transMatrixAbs1(coverHop,:)=0;
        transMatrixAbs1(coverHop,coverHop) = 1;
        transMatrixCover = transMatrixAbs1^coverIndex;
        term1(coverIndex) = term1(coverIndex) + transMatrixCover(nodeSrc,coverHop);
%         term1(coverIndex) = transMatrixCover(nodeSrc,coverHop);
    end
end

transMatrixN = transMatrix; % transition Matrix for Nth term (to power of t)
transMatrixAbs = transMatrix; %add Absorbing nodes for UNION
coverTermNSum = zeros(1,n); % the sums 
coverCDF = zeros(1,t);
termNsum = zeros(t,n);

combNsum = 0; % for 3 terms (iUjUk, there are 6C3 combinations to sum up)
% GET terms 2:n 
for coverIndex = 1:t
    arrUnion = zeros(1,n);
    transMatrixAbs = transMatrix;
%     termNsum = 0;
%     termNsum = zeros(t,n);
    for termN = 2:n
        for unionNumber = 1:termN
            if (unionNumber==nodeSrc)
                continue
            end

            transMatrixAbs(unionNumber,:) = 0;
            transMatrixAbs(unionNumber,unionNumber) =1;
        end
        transMatrixT = transMatrixAbs^coverIndex;

        transMatrixT(nodeSrc,nodeSrc) =0; % to not include z

        termNsum(coverIndex,termN) = sum(transMatrixT(nodeSrc,1:termN));
%         arrUnion(t,termN) = termNsum(t,termN);
    end

                                               
        coverCDF(coverIndex) = term1(coverIndex);
        for coverSum = 1: n-1
            coverCDF(coverIndex) = coverCDF(coverIndex) +(-1)^coverSum... 
                *termNsum(t,coverSum+1);
        end
end

figure(6)
subplot(1,2,2)
plot(1:t,coverCDF,'*')
title('CDF of Cover_i_j')
ylabel('Probability')
xlabel('Number of hops')

subplot(1,2,1)
plot(uniqueC,coverCDFsim,'*')
title('Simulated CDF of Cover_i_j')
ylabel('Probability')
xlabel('Number of hops')
axis([0 100 0 1])
% axis([0 70 0 1.2])
% histogram(numHops,'Normalization','cdf','BinWidth',0.3)

%%
% simulated coverHops PMF and CDF
uniqueHops = unique(numHops);
% numUnique = length(uniqueHops);
[uniqueC,ia,ic] = unique(numHops);
uniqueCount = accumarray(ic,1)';
coverPDFsim = uniqueCount/sum(uniqueCount);

figure(7)
subplot(1,2,1)
plot(uniqueC,coverPDFsim,'*')
title('Simulated PMF of Cover_i_j')
ylabel('Probability')
xlabel('Number of hops')
coverCDFsim = zeros(1,length(uniqueC));
for plotIndex = 1:length(uniqueC)
    coverCDFsim(plotIndex) = sum(coverPDFsim(1:plotIndex));
end



% PMF from CDF of cover formula
coverPDFnew = zeros(1,t);
for newIndex = 2:t
    coverPDFnew(newIndex) = coverCDF(newIndex) - coverCDF(newIndex-1);
end
figure(7)
subplot(1,2,2)
plot(1:t,coverPDFnew,'*')
title('PMF of Cover_i_j')
ylabel('Probability')
xlabel('Number of hops')
axis([0 60 0 1])






