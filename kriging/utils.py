import numpy as np
from scipy import linalg
from rdkit.Chem import MolFromSmiles, Descriptors, rdMolDescriptors, RDKFingerprint



# -------------------------------------------------------------------------------
# Checking properties of matrices 
# -------------------------------------------------------------------------------


# Check that the matrix is positive semi-definite indentifying its smallest eigenvalue
def check_psd(matrix) :
    eigenvalues = np.linalg.eigvalsh(matrix)   
    print(f"\nEigenvalue check (matrix should be positive semi-definite):")
    print(f"  Smallest eigenvalue : {eigenvalues.min():.6f}")
    print(f"  Largest eigenvalue  : {eigenvalues.max():.6f}")
    print(f"  Number of eigenvalues < 1e-6: "f"{np.sum(eigenvalues < 1e-6)}")

    if eigenvalues.min() < -1e-6:
        print("\n  WARNING: Matrix has negative eigenvalues — not PSD.")
    else:
        print("\n  Matrix is positive semi-definite.\n  Confirmed!")



# Check that the matrix is positive definite indentifying its smallest eigenvalue
def check_pd(matrix) :
    eigenvalues = np.linalg.eigvalsh(matrix)
    print(f"\nEigenvalue check (matrix should be positive definite):")
    print(f"  Smallest eigenvalue: {eigenvalues.min():.6f}  "
      f"(must be > 0 for Cholesky factorization to succeed)")
    print(f"  Largest eigenvalue : {eigenvalues.max():.6f}")

    if eigenvalues.min() <= 0:
        raise SystemExit(
            "Matrix is not strictly positive definite.\n"
            "Try increasing SIGMA2_N."
        )
    else :
        print("  Matrix is strictly positive definite:\n  Confirmed!")



# Check that this is a symmetric matrix with simple assertions
def check_unit_symmetry(matrix) :
    assert np.allclose(matrix, matrix.T), "Matrix is not symmetric!"
    assert np.allclose(np.diag(matrix), 1.0), "Diagonal is not 1.0!"
    print("\nSymmetry check  : passed!")
    print("Diagonal check  : passed!")



# Print all values, then the diagonal, and some off-diagonal metrics
def echo_matrix(matrix) :
    print("\nTanimoto similarity matrix:")

    n = matrix.shape[0]

    col_names = "\t\t" + "  ".join(f"mol_{j+1:02d}" for j in range(n))
    print(col_names)

    for i in range(n):
        row_name = f"mol_{i+1:02d}"
        row_vals = "  ".join(f"{matrix[i,j]:.3f}" for j in range(n))
        print(f"{row_name}\t\t{row_vals}")


    print(f"\nDiagonal entries (should all be 1.000):")
    print("  " + "  ".join(f"{v:.3f}" for v in np.diag(matrix)))


    upper = matrix[np.triu_indices(n, k=1)]
    print(f"\nOff-diagonal summary:")
    print(f"  Min  : {upper.min():.4f}")
    print(f"  Max  : {upper.max():.4f}")
    print(f"  Mean : {upper.mean():.4f}")



def factorize(matrix) :
    matrix_factor, lower = linalg.cho_factor(matrix, lower=True)
    lower_tri = np.tril(matrix_factor)

    matrix_rebuilt = lower_tri @ lower_tri.T
    max_error = np.abs(matrix_rebuilt - matrix).max()
    print(f"  Verification if matrix is rebuilt by 'lower_triangle * upper_triangle':")
    print(f"  Max absolute entry-wise error: {max_error:.2e}")
    assert max_error < 1e-8, "Cholesky factorization failed."
    print("  Cholesky factorization check passed!")

    return matrix_factor, lower_tri 



############################################################################################################
