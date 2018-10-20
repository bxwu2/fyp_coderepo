close all
clear all
clc

numTrials = 1000;
numHops = zeros(1,numTrials);
% setting up a 4 node network
A = [1 1 2 2 3]; 
B = [2 3 3 4 4];
% test = 4;
% A = 1:test;
% A = randperm(test);
% B = randperm(test);
G = graph(A,B); 
subplot(1,2,1)
plot(G)
title('Network of Nodes')

n = max([A B]); % how many nodes in the network

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

