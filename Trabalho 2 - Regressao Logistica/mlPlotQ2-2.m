X = dlmread("newdata2.txt")
y = X(:, 497)

X(:, 497]) = []

theta = dlmread("theta.txt")

plotDecisionBoundary(theta, X, y)