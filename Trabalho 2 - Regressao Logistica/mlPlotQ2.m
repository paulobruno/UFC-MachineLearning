X = dlmread("newdata2.txt")
X = [ones(size(X, 1), 1) X];
y = X(:, 497)

X(:, 497) = []

theta = dlmread("theta.txt")

plotDecisionBoundary(theta, X, y)