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

% %%
% % PDF
% % pick node 2; start = 2;
% % select the node 2's neighbors; nodeN = neighbors(G,2);
% % how many neighbors? numN = nnz(nodeN);
% % set the probability of which node it jumps to
% % loop for every node (into an array)
% 
% % THIS BIT WAS ORIGINALLY IN START
% % arrProb = zeros(1,n);
% % for nodeProb = 1:n
% %     nodeN = neighbors(G,nodeProb);
% %     numN = nnz(nodeN);
% %     arrProb(nodeProb) = 1/numN;
% % end
% % ^ NEED THIS IN AN ARRAY SO DONT HAVE TO DO THIS LOOP AGAIN
% 
% arrNeighborNum = sum(arrNeighbor~=0);
% % startNode = 3;
% % finishProb = zeros(1,n);
% t = 5;
% arrRecur = zeros(t+1,n);
% arrRecur(1,:) = arrProb;
% %     recurIndex = 0;
%     for recurIndex = 1:5
%         for nodeIndex = 1:n
%             probCalc = 0;
%             for nodeHop = neighbors(G,nodeIndex)
%                 probCalc = probCalc + 1/arrNeighborNum(nodeIndex)*arrRecur(recurIndex,nodeIndex);
%             end
%             arrRecur(recurIndex,nodeIndex) = probCalc;
%         end
%     end 
% 
% % given the set probabilities as above
% % pick a starting node; start = 2;
% % prob2 = 1/numN * prob
% % probCalc = 0;
% % for nodeN = neighbors(G,2)
% %   probCalc = prob2 + 1/numN*prob3
% % end
% 
% 
% % NEED TO ENSURE ABSORBING NODE endNode = 1

%% PDF seemingly works???

arrHopProb = zeros(1,n);
for nodeIndex = 1:n
    arrHopProb(nodeIndex) = 1/arrNeighborNum(nodeIndex); 
end

t = 100;
arrRecur = zeros(t+1,n);

arrRecur(1,nodeSrc) = 1;
for iter = 1:t
    for nodeDstIndex = 1:n
        for prevHop = 1:arrNeighborNum(nodeDstIndex)
%             iter
%             arrNeighbor(1:arrNeighborNum(nodeIndex))
            if (arrNeighbor(prevHop,nodeDstIndex)==nodeDst)
                continue
            end
            arrRecur(iter+1,nodeDstIndex) = arrRecur(iter+1,nodeDstIndex)+arrHopProb(arrNeighbor(prevHop,nodeDstIndex))...
                *arrRecur(iter,arrNeighbor(prevHop,nodeDstIndex))
        end
    end
end
figure(2)
subplot(2,2,1)
plot(0:t,arrRecur(1:t+1,nodeDst),'*')
title('PDF of Hit_i_j')
xlabel('t (hops)')
ylabel('Probability (%)')

%% 
% CDF
arrRecurCDF = arrRecur;
for cdfIndex = 3:t
    arrRecurCDF(cdfIndex,nodeDst) = arrRecurCDF(cdfIndex,nodeDst)+arrRecurCDF(cdfIndex-1,nodeDst);    
end
% figure(3)
subplot(2,2,2)
plot(0:t, arrRecurCDF(:,nodeDst),'o')
title('CDF of Hit_i_j')
ylabel('Probability (%)')
xlabel('t (hops)')

%% COMMUTE CDF??? you can find commute pdf with convolution of hit times ij to ji
% transition matrix C (for cover time stuff???)
commuteDst = 4;
commuteSrc = 1;
transMatrix = zeros(n);
for transHop = 1:n
    for transHopNeighbor = 1:arrNeighborNum(transHop)
        transMatrix(transHop,arrNeighbor(transHopNeighbor,transHop))=arrHopProb(transHop);
    end
end
transDstar = transMatrix;
transDstar(commuteDst,:) = 0;
transO = zeros(n); % possibly not needed
transO(commuteDst,commuteDst) = 1;
transDi = transMatrix;
transDi(commuteSrc,:) = 0;
transDi(commuteSrc,commuteSrc) = 1;
matrixC = [transDstar transO; zeros(n) transDi];
% C^t_i,i+n = kij - 1
commuteTime = matrixC(commuteSrc, commuteSrc+n) + 1;

%% COMMUTE PDF???
%  hij(tau) hji(t-tau) 
% find Hit time of Src -> Dst
% find Hit time of Dst -> Src

arrHopProb = zeros(1,n);
for nodeIndex = 1:n
    arrHopProb(nodeIndex) = 1/arrNeighborNum(nodeIndex); 
end

t = 100;
arrRecurReturn = zeros(t+1,n);

arrRecurReturn(1,nodeDst) = 1;
for iter = 1:t
    for nodeSrcIndex = 1:n
        for prevHop = 1:arrNeighborNum(nodeSrcIndex)
%             iter
%             arrNeighbor(1:arrNeighborNum(nodeIndex))
            if (arrNeighbor(prevHop,nodeSrcIndex)==nodeSrc)
                continue
            end
            arrRecurReturn(iter+1,nodeSrcIndex) = arrRecurReturn(iter+1,nodeSrcIndex)+arrHopProb(arrNeighbor(prevHop,nodeSrcIndex))...
                *arrRecurReturn(iter,arrNeighbor(prevHop,nodeSrcIndex))
        end
    end
end
% figure(4)
subplot(2,2,3)
plot(0:t,arrRecurReturn(1:t+1,nodeSrc),'*')
title('PDF of Hit_j_i')
xlabel('t (hops)')
ylabel('Probability (%)')

% hit_ji done
% now CONVOLVE
% sum(tau = 1 to t) hij(tau) hji(t-tau)
commuteConv = zeros(1,t+1);
commuteConvPDF = commuteConv;
for tau = 1:t
    commuteConv(tau+1) = arrRecur(tau+1,nodeDst)*arrRecurReturn(t-tau+1,nodeSrc);
    commuteConvPDF(tau+1) = commuteConv(tau+1)+commuteConv(tau);    
end



% figure(5)
subplot(2,2,4)
plot(1:t+1,commuteConvPDF,'*')
title('PDF of Commute_i_j')
ylabel('Probability (%)')
xlabel('t (hops)')
% need to test on t*2 hops probably?? because commute > hit time




% % recur 
% index = 1 t =0         
% index = 2 t =1     
% index = 3 t =2
% 2



% title(sprintf('x_{%d}',a))
% title( num2str(a))

