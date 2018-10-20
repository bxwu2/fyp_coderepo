close all
clear all
clc

n = 5
numTrials = 1000;
numHops = zeros(1,numTrials);
% setting up a 4 node network
d = randi([0,1],[n,1]); % The diagonal values
    A = randi([0,1],n);

    Atriag = triu(A,1);
    for self = 1:n-1
        while sum(Atriag(self,:))==0
                Atriag(self,randi([self+1,n]))=1;
    %             Atriag = triu(A,1);
        end
    end

    % problem is, still get isolated networks 
    M = diag(d) + Atriag+ Atriag';
    for self = 1:n
        M(self,self) =0; 
    end
G = graph(M); 
subplot(1,2,1)
plot(G)
title('Network of Nodes')

% n = length(M); % how many nodes in the network

arrNeighbor = zeros(n-1,n); % array for each node's neighbours
numNeighbor = zeros(1,n);   % number of neighbors in each node

for trials = 1:numTrials
    % setting up array of node neighbours
    for i = 1:n
        N = neighbors(G,i);    % node neighbours
        numNeighbor(i) = length(N); % number of neighbours in node
        for j = 1:numNeighbor(i)    % adding to array of node neighbours
            arrNeighbor(j,i) = N(j);
        end
    end

    % arrAdj
    % indAdj
    counterNode = zeros(1,n); % counting hops to node
    node=1; % start at node 1

    % % plotting the hop movement
    % dataX = 0; 
    % dataY = 0;
    % idxData = 1;

    % picking a node to hop to
    while (sum(~counterNode)>0)   % while not all nodes covered
        for k = 1:n
            if (node==k)    % if at this node
                p = randi([1,numNeighbor(k)]);  % pick random neighbor node
                node = arrNeighbor(p,k);        % hop there
                counterNode(node) = counterNode(node)+1;       % increase node covered
    %             dataY(idxData) = node;     % for plotting
    %             idxData = idxData + 1;      
            end
        end
    end 

    numHops(trials) = sum(counterNode); % total number of hops
    
    % dataX = 1:numHops; % x axis of plot
    % figure(2)
    % plot(dataX,dataY,'o-');
    % title('Node for each hop')
    fprintf('Cover time = %i hops\n',numHops(trials)) 
end
% figure(2)
subplot(1,2,2)
hist(numHops, numTrials)
title('Histogram for cover time')
ylabel('Frequency')
xlabel('Number of hops')

