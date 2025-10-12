import numpy as np
from scipy.sparse import coo_matrix

# Create COO with duplicate entries
rows = np.array([0, 0, 1])
cols = np.array([1, 1, 2])
data = np.array([1, 2, 3])

M_coo = coo_matrix((data, (rows, cols)), shape=(3, 3))

print("Raw COO entries:")
print("rows:", M_coo.row)
print("cols:", M_coo.col)
print("data:", M_coo.data)
print("Dense array from COO:\n", M_coo.toarray())

# Convert to CSR
M_csr = M_coo.tocsr()
print("\nAfter converting to CSR:")
print(M_csr)
print("Dense array from CSR:\n", M_csr.toarray())

# Convert to CSC
M_csc = M_coo.tocsc()
print("\nAfter converting to CSC:")
print(M_csc)
print("Dense array from CSC:\n", M_csc.toarray())
