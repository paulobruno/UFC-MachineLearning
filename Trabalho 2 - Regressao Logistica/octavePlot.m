A = dlmread("ex2data1.txt")

x = A(:, 1)
y = A(:, 2)
z = A(:, 3)

c = 2.88420941  
b = 3.77987642
a = 3.08822069

#scatter3 (x, y, z, 10, z(:))
mesh (a, b, c)

input("wait")
