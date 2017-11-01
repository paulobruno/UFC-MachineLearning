A = dlmread("ex2data2.txt")
x1 = A(:, 1)
x2 = A(:, 2)
y = A(:, 3)

M = mapFeature(x1, x2)
M = [M y];
M(:, 1) = []

dlmwrite("newdata2.txt", M, 'precision', 6)