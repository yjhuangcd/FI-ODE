# Count how many samples we need for grids on the decision boundary
import numpy as np
import math

n = 10
T = 40

f = [[0]*(n+1) for _ in range(T+1)]

for j in range(T+1):
	for k in range(n+1):
		if(j==0):
			f[j][k] = 1
		elif(k<2 or j==1):
			f[j][k] = 0
		elif(k==2 and np.mod(j,2)==0):
			f[j][k] = 1
		elif(k==2 and np.mod(j,2)==1):
			f[j][k] = 0
		else:
			for l in range(k-1):
				if(j-k+l>=0 and k-l>=0):
					f[j][k] = f[j][k] + f[j-k+l][k-l]*math.comb(k-1,l)

print(f[T][n])