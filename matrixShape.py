import numpy as np 
matrix=np.array([[1,2,3,4,5],[1,2,3,3,3]])
matrix2=matrix.T
print(matrix.shape[1])
print(matrix2.shape[0])

if (matrix.shape[1]!=matrix2.shape[0]):
   print(False)
else:
   print(True)

