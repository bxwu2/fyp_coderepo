%% COMBINED
clear all 
close all
clc

%% GENERATE NETWORK OF n NODES
n = 3; % number of nodes in network

% 1 particular network
% 2 random network of size n
choice = 2;
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


%% EXPERIMENTAL COMMUTE TIME
numTrials = 50000;

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

figure(5)
subplot(1,2,1)
plot(G)
title('Generated Network of Nodes')
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
figure(6)
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

figure(5)
hold on
subplot(1,2,2)
plot(uniqueHops,uniqueProb,'--o')




%% COMPARING EXPERIMENTAL WITH SIMULATION CDF
% plot CDF of Commute Time
% get rid of 0 increments for CDF
figure(7)

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

% nodeDst = randi([1,n]); % node to hit
% nodeSrc = randi([1,n]); % node to start from
% nodeSrc = 1;
% nodeDst = 4;
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

% subplot(1,2,2)
% bar(uniqueHops,uniqueArr/numTrials);
% title('Histogram for Number of Hops')
% ylabel('Probability')
% xlabel('t (hops)')    
% xticks(0:2:100)

%% Comparing Experimental with Simulation/Formula for PDF
figure(8)
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
figure(9)
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

figure(1)
hold on
plot(uniqueHops,uniqueProb,'--o');




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

% Term 1
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

% Term 1 for PDF and CDF
cover1_PDF = sum(hitPDF(:,2:end)); % sum of hit times from src to all nodes (i=/=z); vector contains sum for t hops (t entries in vector)
cover1_CDF = sum(hitCDF(:,2:end));

% Term 2 to n
    combSumCDF=zeros(1,t);
    combSumPDF=zeros(1,t);
    for unionIdx = 2:n                                                    % for the 2nd term to the n-1 th term
        combArr = nchoosek(1:n,unionIdx);                                   % union combinations
            for combIndex = 1:size(combArr,1)                               % cycle through the union combos
                testSrc = ismember(combArr(combIndex,:),nodeSrc);           % exclude combinations with starting node
                if(sum(testSrc))
                    continue
                end
                
                % find Union Hit Time CDF
                transMatrixAbs = transMatrix;                               
                transMatrixAbs(combArr(combIndex,:),:)= 0;
                absIdx = sub2ind(size(transMatrix),combArr(combIndex,:),...
                    combArr(combIndex,:));
                transMatrixAbs(absIdx) = 1;
                
                for coverIdx = 1:t                                                           % for 1-100 hops
                    transMatrixT1 = transMatrixAbs^(coverIdx-1);
                    transMatrixCDF = transMatrixAbs^(coverIdx);
                    transMatrixPDF = transMatrixCDF - transMatrixT1;
                    combProbsCDF = transMatrixCDF(nodeSrc,combArr(combIndex,:));                  % union probabilities
                    combProbsPDF = transMatrixPDF(nodeSrc,combArr(combIndex,:));
                    combSumCDF(coverIdx) = combSumCDF(coverIdx) +(-1)^(unionIdx-1)*sum(combProbsCDF);   % add to nth term sum sequence
                    combSumPDF(coverIdx) = combSumPDF(coverIdx) +(-1)^(unionIdx-1)*sum(combProbsPDF);   % add to nth term sum sequence
                end
            end
    end
    
    %% Last term
%     transMatrixAbs = transMatrix;
%     for idx = 1:n
%         transMatrixAbs(idx,:) = 0;
%         transMatrixAbs(idx,idx) = 1;
%     end
%     combEnd=zeros(1,t);
%     for sumIdx = 1:n
%         for endIdx = 1:t
%             transMatrixT_end = transMatrixAbs^endIdx;
%             combEnd(endIdx) = combEnd(endIdx) + transMatrixT_end(nodeSrc,sumIdx);
%         end
%     end
       
    %% Total Cover Time for both PDF and CDF
    coverTotalCDF = cover1_CDF + combSumCDF;
    coverTotalPDF = cover1_PDF + combSumPDF;

    % Superimposing the Cover Time PDF and CDF formula plots with the
    % experimental plots    
    figure(8)
    hold on
    plot(0:100, [0 coverTotalPDF],'--*')
    axis([0 100 0 max(coverTotalPDF)*1.1])
    legend('Experimental', 'Mathematical Formula')
    
    figure(9)
    hold on
    plot(0:100, [0 coverTotalCDF],'--*')
    axis([0 100 0 1.05])
    legend('Experimental', 'Mathematical Formula')