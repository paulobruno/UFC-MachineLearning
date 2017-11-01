load('ex6data2.mat')
%y(37)=1
tic
model = svmTrain(X, y, 1, @gaussianKernel, 1e-8, 20);
toc
%end  
%plot(media)
%plotData(X,y)
%visualizeBoundaryLinear(X,y, model)
visualizeBoundary(X,y,model)