M = load('ex5data1.data')
A = M(:,1:4)
%[eg, ev] = pca(X)
ev = load('ev.data')
X = A*ev(:,1)
Y = A*ev(:,2)
gscatter(X,Y,M(:,5))