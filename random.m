clear all
close all
clc


% so what does this do

% G = graph([1,2],[2,3],4,5);
% plot(G)

% random stuff
% M = ~abs(randi(3,5)-2) 
%     %Locations of the nodes are on a circle
% V = [cosd(linspace(0,180,5));sind(linspace(0,180,5))]' ;
% 
% 
% figure(1) ; clf
% hold on ; axis equal
%     %plot and label the nodes
% plot(V(:,1),V(:,2),'.') ;
% for i = 1 : size(V,1)
%     text(V(i,1),V(i,2),num2str(i)) ;
% end
%     %also plot the conections
% [I,J] = ind2sub(size(M),find(M)) ;
% for ii = 1 : numel(I)  ;
%     i = I(ii) ;
%     j = J(ii) ;
% 
%     plot([V(i,1),V(j,1)]',[V(i,2) V(j,2)]','k-') ;
% end

% A = randi([1,10],1,10) % generate a matrix 1x10, from 1 to 10
n = randi([5,10]);
test = 1:n;
A = randperm(length(test))
B = randperm(length(A))
% A = randi([1,n],[1,n]);
% B = randi([1,n],[1,n]);

% a_rand = a(randperm(length(a))); % randperm(5) = random permutation of 1:5
while(sum(A==B))>0
    A = randperm(length(test))
    B = randperm(length(A))
% A = randi([1,n],[1,n]);
% B = randi([1,n],[1,n]);
end 
% 
% G = graph(A,B);
% plot(G)

% s = [1 2 1 2 2 3];
% t = [2 1 4 5 6 7];
% G = digraph(s,t)
G = graph(A,B)
plot(G)




