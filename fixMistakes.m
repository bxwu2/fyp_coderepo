% sum(M)
onlyOne = find(sum(M)==1);
M(onlyOne(1),randi([1,n]))=1
Atriag = triu(M,1);
M = Atriag + Atriag'
plot(graph(M))
