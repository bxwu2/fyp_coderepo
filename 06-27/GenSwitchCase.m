close all
% clear all
clc

% Barry Wu
% 2599 3534
% Date created: 17-04-2018
% Date modified: 30-04-2018

n = 6; % number of nodes in network

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
        i = 0;
        testCon = 3;
        testCon2 = 1;
        
        while (testCon>2||testCon2)	% while there are no empty columns/rows (no isolated nodes), no multiple networks with the numEdges rule
%             (sum(~sum(M))
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
            testCon = nnz(~(sum(M)-1)); % test for disconnectedness
            testCon2 = nnz(~sum(M));
%             testCon = sum(M)-1
            i= i+1
%             testCon3 = nnz((sum(M)-1));
        end
end

G = graph(M);
subplot(1,2,1)
plot(G)
title('Generated Network of Nodes')
