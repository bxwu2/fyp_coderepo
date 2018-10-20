close all
clear all
clc

% Barry Wu
% 2599 3534
% Date created: 17-04-2018
% Date modified: 26-07-2018

%% GENERATE NETWORK OF n NODES
n = 6; % number of nodes in network
numTrials = 10000;
% 1 particular network
% 2 random network of size n
choice = 2;
% 0 random nodeSrc, nodeDst
% 1 same nodeSrc, nodeDst used
sameNode = 1;
switch choice
    case 1 
        % arrControl = [ 0 0 0 1 1 0; 0 0 0 0 1 0; 0 0 0 1 1 0; 1 0 1 0 1 0; 1 1 1 1 0 1; 0 0 0 0 1 0];
        arrControl=[0 0 0 1 1 1; 0 0 0 0 1 1; 0 0 0 1 0 1; 1 0 1 0 0 0; 1 1 0 0 0 0; 1 1 1 0 0 0]';
%           arrControl=[0 1 1 1 0 0;1 0 0 0  1 0; 1 0 0 0 1 0 ; 1 0 0 0 1 0; 0 1 1 1 0 1; 0 0 0 0 1 0]
        M = arrControl;
           

    case 2
%         testCon = nnz(~sum(M));
        testCon = 1;
        testCon2 = 0;
%         plot(graph(M))
%         while (testCon>n-1||testCon2~=10)	% while there are no empty columns/rows (no isolated nodes), no multiple networks with the numEdges rule
          while(testCon>0||testCon2==0)
                A = randi([0,1],n); % generate matrix
                Atriag = triu(A,1); % upper triangular (remove diagonals)
                M = Atriag + Atriag'; % create adjacency matrix
            
        	plot(graph(M))
            testCon = nnz(~sum(M))
            
            isoArr = ismember(1,sum(M)) % find sum(col)=1
            if (isoArr)
                isoArrLen = find(sum(M)==1)
                for isoIdx = isoArrLen
                    %Mcol = M(:,isoIdx)			% find iso column
                    %Madj = find(Mcol == 1)	% find iso's neighbour
                    Mcheck = 0;
                    Minf = 0;
                    while(~Mcheck) && (Minf < n)	% while neighbour is iso & not repeating
                        Minf = Minf + 1
                        Mcol = M(:,isoIdx)      	% find iso column
                        Madj = find(Mcol==1)		% find iso's neighbour
                        Mcheck = (sum(M(:,Madj))>1)	% check neighbour is iso
                        testCon2 =1
                    end
                    if(~testCon2)
                        break;
                    end
                end
            end
        end
end

G = graph(M);
% subplot(4,3,2)
h = plot(G);
title('Generated Network of Nodes')

return;

if (choice == 1)||(sameNode==1)
    nodeDst = 6;
    nodeSrc = 3;
else
    nodeDst = randi([1,n]); % node to hit
    nodeSrc = randi([1,n]); % node to start from

    % nodeSrc = 1;
    while (nodeDst == nodeSrc)  % to ensure desired node is not same as start
        nodeDst = randi([1,n]);
    end
end

highlight(h,nodeSrc,'NodeColor','r')
highlight(h,nodeDst,'NodeColor','g','Marker','*')
labelnode(h,[nodeSrc nodeDst],{'Source' 'Destination'})

if (n==2)
    % Hitting time
    subplot(4,3,4) % Histogram
        bar(1,1)
        hold on
        plot(0:1,0:1,'--o') 
        title('Experimental Histogram & PMF')
        ylabel('Probability')
        xlabel('t (hops)')   
            subplot(4,3,5)  % PDF
            plot(0:1,0:1,'--o',0:1,0:1,'--*')
            axis([0 3 0 1])
            title('Comparing Hitting Time PMF')
            ylabel('Probability')
            xlabel('t (hops)')   
                subplot(4,3,6)  % CDF
                plot(0:100,[0 ones(1,100)],'--o',0:100,[0, ones(1,100)],'--*')
                axis([0 10 0 1])
                title('Comparing Hitting Time CDF')
                ylabel('Probability')
                xlabel('t (hops)')   
    % Commute time            
    subplot(4,3,7)
        bar(2,1)    %  histogram
        hold on
        plot(0:2,[0 0 1],'--o')
        title('Experimental Histogram & PMF')
        ylabel('Probability')
        xlabel('t (hops)')   
            subplot(4,3,8)
                plot(0:2,[0 0 1],'--o',0:2,[0 0 1],'--*')
                axis([0 3 0 1])
                title('Comparing Commute Time PMF')
                ylabel('Probability')
                xlabel('t (hops)')   
                    subplot(4,3,9)
                    plot(0:100,[0 0 ones(1,99)],'--o',0:100,[0 0 ones(1,99)],'--*')   
                    axis([0 10 0 1])
                    title('Comparing Commute Time CDF')
                    ylabel('Probability')
                    xlabel('t (hops)')   
    % Cover time
    subplot(4,3,10) % histogram
        bar(1,1)
        hold on
        plot(0:1,0:1,'--o')
        title('Experimental Histogram & PMF')
        ylabel('Probability')
        xlabel('t (hops)')   
            subplot(4,3,11) % PDF
            plot(0:1,0:1,'--o',0:1,0:1,'--*')
            axis([0 3 0 1])
            title('Comparing Cover Time PMF')
            ylabel('Probability')
            xlabel('t (hops)')   
                    subplot(4,3,12) % CDF
                    plot(0:100,[0 ones(1,100)],'--o',0:100,[0,ones(1,100)],'--*')
                    axis([0 10 0 1])
                    title('Comparing Cover Time CDF')
                    ylabel('Probability')
                    xlabel('t (hops)')   
    return;
end
   
    


%% EXPERIMENTAL HIT TIME
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



counterNode = zeros(1,numTrials);
for trials = 1:numTrials
    node = nodeSrc;
    counterHit = 0;

    while (~counterHit)
              k = find(1:n==node);
              p = randi([1,numNeighbor(k)]);  % pick random neighbor node
              node = arrNeighbor(p,k);        % hop there
              counterNode(trials) = counterNode(trials) +1;   % number of hops so far
              if (node == nodeDst)
                  counterHit = 1;
              end
    end 
end

subplot(4,3,4)
uniqueHops = unique(counterNode);
numHopsU = length(uniqueHops);
% title('Hit Time')

uniqueArr = zeros(1,numHopsU);
for uniqueIdx = 1:numHopsU
    uniqueMatch = uniqueHops(uniqueIdx)== counterNode;
    uniqueMatchCount = sum(uniqueMatch);
    uniqueArr(uniqueIdx) = uniqueMatchCount;
end

bar(uniqueHops,uniqueArr/numTrials);
uniqueArr = 0:uniqueHops(end);
for fillIdx= 1:uniqueHops(end)+1
    uniqueMatch = uniqueArr(fillIdx)== counterNode;
    uniqueMatchCount = sum(uniqueMatch);
    uniqueArr(fillIdx) = uniqueMatchCount;
end
uniqueProb = uniqueArr/numTrials;
onestep = ismember(1,uniqueProb);
if(onestep)
    uniqueProb = [uniqueProb zeros(1,10)];
    uniqueHops = 0:length(uniqueProb)-1;
end

hold on
plot(0:uniqueHops(end),uniqueProb,'--o');
title('Histogram for Experimental Hitting Time')
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

%% CDF of Simulated Hit Time
t = 100; % end time

hitCDF= zeros(1,t+1);
for tCDF = 1:t
    Dt = D^tCDF;
    hitCDF(tCDF+1) = Dt(nodeSrc,nodeDst);
end

%% PDF

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

%% COMPARING EXPERIMENTAL WITH SIMULATION
% uniqueProb = uniqueArr/numTrials; % experimental prob

% experimental overlaying the histogram
hold on
subplot(4,3,4)
% plot(uniqueHops,uniqueProb,'--o')

% figure(2)
% bar(uniqueHops,uniqueArr/numTrials);
subplot(4,3,5)
% onestep = ismember(1,uniqueProb);
% if(onestep)
%     uniqueProb = [uniqueProb zeros(1,10)];
%     uniqueHops = 0:length(uniqueProb)-1;
% end

plot(0:uniqueHops(end),uniqueProb,'--o');
title('Hitting Time PMF')
ylabel('Probability')
xlabel('t (hops)')
% xticks(0:2:50)
% xlim([0 50])
hold on


nonZero = ~(hitPDF==0);
xPDF  = 0:t;
hitPDFomit = hitPDF(nonZero);
% plot(xPDF(nonZero),hitPDFomit,'--*')
plot(xPDF,hitPDF,'--*')

legend('Experimental','Constructed')

boundary = find(hitPDF<0.001);
    limitIdx = 1;
    limit = boundary(limitIdx);
    flag = 0;
%     onestep = (find(hitPDF==1));
%     onestep = ismember(1,hitPDF);
    while (~flag && ~onestep)
        if(hitPDF(limit+1)<hitPDF(limit))
            flag = 1;
        else
            limitIdx = limitIdx+1;
            limit = boundary(limitIdx);
        end
    end
    xlim([0 limit+10])


%% COMPARISON WITH CDF
% figure(3)
subplot(4,3,6)
% experimental CDF
uniqueProbCDF = uniqueProb;
for idx = 1:length(uniqueProbCDF)-1
    uniqueProbCDF(idx+1) = uniqueProbCDF(idx)+uniqueProbCDF(idx+1);
end

plot(0:uniqueHops(end),uniqueProbCDF,'--o');
title('Hitting Time CDF')
ylabel('Probability')
xlabel('t (hops)')
xticks(0:4:100)
hold on

% overlay the simulation with experimental CDF
% hitCDF = hitPDFomit;
hitCDF = hitPDF;
for idx = 1:length(xPDF)-1
    hitCDF(idx+1) = hitCDF(idx)+hitCDF(idx+1);
end

% plot (xPDF(nonZero),hitCDF,'--*')
plot(xPDF,hitCDF,'--*')
legend('Experimental','Constructed')
subplot(4,3,4)
xlim([0 limit+10])
subplot(4,3,6)
xlim([0 limit+10])


%%
%%
%% HIT ABOVE
%% COMMUTE BELOW
%%
%%
%%

%% EXPERIMENTAL COMMUTE TIME
counterNode = zeros(1,numTrials);
for trials = 1:numTrials
    node = nodeSrc;
    counterHit = 0;
    counterRet = 0;

    while (~counterRet)
         k = find(1:n==node);
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
    end 
end
% subplot(4,3,7)
% plot(G)
% title('Generated Network of Nodes')


uniqueHops = unique(counterNode);
numHopsU = length(uniqueHops);

uniqueArr = zeros(1,numHopsU);
for uniqueIdx = 1:numHopsU
    uniqueMatch = (uniqueHops(uniqueIdx)==counterNode);
    uniqueMatchCount = sum(uniqueMatch);
    uniqueArr(uniqueIdx) = uniqueMatchCount;
end
subplot(4,3,7)
bar(uniqueHops,uniqueArr/numTrials);
title('Experimental Histogram for Commute Time')
ylabel('Probability')
xlabel('t (hops)')    
xticks(0:2:100)

%% SIMULATED COMMUTE TIME

% create Transition Matrix

% create transitional probabilities
% arrHopProb = zeros(1,n);
% for probHop = 1:n
%     arrHopProb(probHop) = 1/arrNeighborNum(probHop);
% end

% create transition matrix
% transMatrix = zeros(n);
% for transHop = 1:n
%     for transHopNeighbor = 1:arrNeighborNum(transHop)
%         transMatrix(transHop,arrNeighbor(transHopNeighbor,transHop))=arrHopProb(transHop);
%     end
% end

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
subplot(4,3,7)
hold on
% bar(uniqueHops,uniqueArr/numTrials);
% uniqueProb = uniqueArr/numTrials;
uniqueArr = 0:uniqueHops(end);
for fillIdx= 1:uniqueHops(end)+1
    uniqueMatch = uniqueArr(fillIdx)== counterNode;
    uniqueMatchCount = sum(uniqueMatch);
    uniqueArr(fillIdx) = uniqueMatchCount;
end
uniqueProb = uniqueArr/numTrials;

plot(0:uniqueHops(end),uniqueProb,'--o');
title('Histogram for Experimental Commute Time PMF')
ylabel('Probability')
xlabel('t (hops)')
xticks(0:2:100)







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

subplot(4,3,8)
plot(0:uniqueHops(end),uniqueProb,'--o')
hold on
% plot(xConv(nonZeroConv),convPDF(nonZeroConv),'--*')
plot([0 xConv],[0 convPDF],'--*')

title('Commute Time PMF')
legend('Experimental','Constructed')
ylabel('Probability')
xlabel('t (hops)')

boundary = find(convPDF<0.001);
    limitIdx = 1;
    limit = boundary(limitIdx);
    flag = 0;
    while (~flag)
        if(convPDF(limit+1)<convPDF(limit))
            flag = 1;
        else
            limitIdx = limitIdx+1;
            limit = boundary(limitIdx);
        end
    end
    xlim([0 limit+10])



%% COMPARING EXPERIMENTAL WITH SIMULATION CDF
% plot CDF of Commute Time
uniqueProbCDF = uniqueProb;
for idx = 1:length(uniqueProbCDF)-1
    uniqueProbCDF(idx+1) = uniqueProbCDF(idx)+uniqueProbCDF(idx+1);
end



subplot(4,3,9)
plot(0:uniqueHops(end),uniqueProbCDF,'--o');
title('Commute Time CDF')
ylabel('Probability')
xlabel('t (hops)')
xticks(0:4:100)
hold on
% get rid of 0 increments for CDF

% GETTING RID OF THE ODD ZEROS (plateaus) 
commutePDF = zeros(1,length(commuteCDF)-1);
for idx = 1:length(commutePDF)-1
    commutePDF(idx+1)=commuteCDF(idx+1)-commuteCDF(idx);      
end
nonZeroPDF = ~(commutePDF==0); 

xCDF = 0:t-1;
% subplot(4,3,9)
% plot(xCDF(nonZeroPDF),commuteCDF(nonZeroPDF),'--*')
plot(xCDF,commuteCDF,'--*')
legend('Experimental','Constructed')
xlim([0 limit+10])
subplot(4,3,7)
xlim([0 limit+10])
% title('CDF of Commute Time')
% title(sprintf('CDF of Commute_%d_%d',nodeSrc,nodeDst))
% ylabel('Probability')
% xlabel('t (hops)')
% hold on


% legend('Constructed','Experimental')

% for plotIdx = 0:2;
%     subplot(4,3,7+plotIdx)
%     xlim([0 limit+10])
% end

    
%%
%%
%%
%% Commute above
%% Cover below!
%%
%%
%%

%% EXPERIMENTAL Cover Time (histogram)
counterTrials = zeros(1,numTrials);
for trials = 1:numTrials
    node = nodeSrc;
    counterNode = zeros(1,n);
    counterNode(node)= 1;
    counterCover = 0;
    while (~counterCover)
         k = find(1:n==node);
              p = randi([1,numNeighbor(k)]);  % pick random neighbor node
              node = arrNeighbor(p,k);        % hop there
              counterNode(node) = counterNode(node) +1;   % number of hops so far
              counterTrials(trials) = counterTrials(trials) + 1;
              if (~sum(~counterNode))
                  counterCover = 1;
              end
    end 
end


uniqueHops = unique(counterTrials);
numHopsU = length(uniqueHops);
uniqueArr = 0:uniqueHops(end);
for fillIdx= 1:uniqueHops(end)+1
    uniqueMatch = uniqueArr(fillIdx)== counterTrials;
    uniqueMatchCount = sum(uniqueMatch);
    uniqueArr(fillIdx) = uniqueMatchCount;
end

% uniqueArr = zeros(1,numHopsU);
% for uniqueIdx = 1:numHopsU
%     uniqueMatch = uniqueHops(uniqueIdx)== counterTrials;
%     uniqueMatchCount = sum(uniqueMatch);
%     uniqueArr(uniqueIdx) = uniqueMatchCount;
% end

% figure(3)
% subplot(2,2,1)
% plot(G)
subplot(4,3,10)
% bar(uniqueHops,uniqueArr/numTrials);
% bar(0:uniqueHops(end),uniqueArr/numTrials);


%% Comparing Experimental with Simulation/Formula for PDF
subplot(4,3,10)
bar(0:uniqueHops(end),uniqueArr/numTrials);
uniqueProb = uniqueArr/numTrials;
hold on
plot(0:uniqueHops(end),uniqueProb,'--o');
title('Histogram for Experimental Cover Time')
ylabel('Probability')
xlabel('t (hops)')    
xticks(0:2:100)




subplot(4,3,11)
% filler = 0:uniqueHops(1)-1;
% uniqueHops = [filler uniqueHops];
% uniqueProb = [zeros(1,length(filler)), uniqueProb];
% plot(uniqueHops,uniqueProb,'--o');
plot(0:uniqueHops(end),uniqueProb,'--o');
title('Cover Time PMF')
ylabel('Probability')
xlabel('t (hops)')
xticks(0:2:100)

%% Comparing Experimental with Simulation/Formula for CDF
uniqueProbCDF = uniqueProb;
% for idx = 1:length(uniqueHops)+1
for idx = 1:length(uniqueProbCDF)-1
    uniqueProbCDF(idx+1) = uniqueProbCDF(idx)+uniqueProbCDF(idx+1);
end

subplot(4,3,12)
plot(0:uniqueHops(end),uniqueProbCDF,'--o');
title('Cover Time CDF')
ylabel('Probability')
xlabel('t (hops)')
xticks(0:4:100)



%% Formula for Cover PDF

hitCDF= zeros(n,t+1);
hitPDF = zeros(n,t+1);
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
% Term 1
    t=100;
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
    subplot(4,3,11)
    hold on
    plot(0:100, [0 coverTotalPDF],'--*')
    axis([0 100 0 max(coverTotalPDF)*1.1])
    
%     
%     figure(2)
% plot(0:uniqueHops(end),uniqueProbCDF,'--o')
% hold on 
% plot(0:100, [0 coverTotalCDF],'--*')
% % plot([0 xConv],[0 convPDF],'--*')
% title('CDF of Cover Time')
% ylabel('p')
% xlabel('Number of hops')
% legend('Experimental','Constructed')
% return;
% figure(1)


    boundary = find(coverTotalPDF<0.01);
    minNodes = numnodes(G)-1;
    limitIdx = minNodes;
    limit = boundary(limitIdx);
    flag = 0;
    while (~flag)
        if(coverTotalPDF(limit+1)<coverTotalPDF(limit))
            flag = 1;
        else
            limitIdx = limitIdx+1;
            limit = boundary(limitIdx);
        end
    end
    xlim([0 limit+10])
    legend('Experimental', 'Constructed')
    
    subplot(4,3,12)
    hold on
    plot(0:100, [0 coverTotalCDF],'--*')
%     axis([0 100 0 1.05])
    legend('Experimental', 'Constructed')

%     for plotIdx = 2:4;
%         subplot(4,3,3*plotIdx-1)
%         xlim([0 limit+10])
%         subplot(4,3,3*plotIdx)
%         xlim([0 limit+10])
%     end
    subplot(4,3,10)
    xlim([0 limit+10])
    subplot(4,3,12)
    xlim([0 limit+10])
    
%     figure(2)
%     plot(0:100,[0 coverTotalCDF],'--*')
%     title('CDF of Cover Time')
%     ylabel('p')
%     xlabel('Number of hops')
%     axis([0 100 0 1])


    