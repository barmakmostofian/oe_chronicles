import sys
import numpy as np
import pandas as pd

from utils import *

from rdkit.Chem         import MolFromSmiles
from rdkit.DataStructs  import BulkTanimotoSimilarity



# ------------------------------------------------------------------
# Define constants — edit these to match your file and preferences
# ------------------------------------------------------------------

CSV_FILE    = "example_compounds.csv"   # path to your input CSV file
SMI_COL     = "Compound Structure"      # column name for smiles strings
OBS_COL     = "pic50"                   # column name for observed values
FP_SIZE     = 1024                      # number of bits in the fingerprint
MIN_PATH    = 1                         # minimum path length (bonds)
MAX_PATH    = 7                         # maximum path length (bonds)



# ------------------------------------------------------------------
# Read data
# ------------------------------------------------------------------

print(f"Reading data from '{CSV_FILE}' ...")
try:
    df = pd.read_csv(CSV_FILE)
except FileNotFoundError:
    print(f"  File '{CSV_FILE}' not found.")

print(f"  Loaded {len(df)} compounds.")
print(f"  Columns found: {list(df.columns)}")


# Convert data to arrays for downstream use
smiles_list = df[SMI_COL].tolist()
obs_values    = df[OBS_COL].to_numpy(dtype=float)
n           = len(smiles_list)



# ------------------------------------------------------------------
# Compute path-based fingerprints (RDKit)
# ------------------------------------------------------------------
# Chem.MolFromSmiles() parses and sanitises the SMILES.
# RDKFingerprint() computes a Daylight-style path-based fingerprint:
#   - enumerates all linear paths from minPath to maxPath bonds
#   - hashes each path into a bit position

print(f"\nComputing RDKit path-based fingerprints "
      f"(fpSize={FP_SIZE}, maxPath={MAX_PATH}) ...")

mols = []
fps  = []
failed = []

for idx, smi in enumerate(smiles_list):
    mol = MolFromSmiles(smi)
    if mol is None:
        print(f"  WARNING: could not parse SMILES at row {idx}: {smi}")
        failed.append(idx)
        continue
    fp = RDKFingerprint(mol, minPath=MIN_PATH, maxPath=MAX_PATH, fpSize=FP_SIZE)
    mols.append(mol)
    fps.append(fp)

if failed:
    # Drop rows with unparseable SMILES
    df          = df.drop(index=failed).reset_index(drop=True)
    obs_values  = np.delete(obs_values, failed)
    n           = len(fps)
    print(f"  {len(failed)} compound(s) dropped due to invalid SMILES.")

print(f"  Done. {n} fingerprints computed.")
print(f"  Bits set in fingerprint 1: {fps[0].GetNumOnBits()} / {FP_SIZE}")



# -------------------------------------------------------------------------------
# Build the Tanimoto similarity matrix, T, prove that it is a kernel, and save it
# -------------------------------------------------------------------------------

print("\nBuilding Tanimoto similarity matrix T ...")

# Compute Tanimoto similarity between any given FP and all other in the list, 
# saving the list of values row-wise in T
T = np.zeros((n, n))
for i in range(n):
    row = BulkTanimotoSimilarity(fps[i], fps)
    T[i, :] = row

print("...done.")


# Check that T is a symmetric matrix with a unit diagonal.
check_unit_symmetry(T)

# Print out matrix values, all diagonal values again, and check som off-diag metrics. 
echo_matrix(T)

# Check that T is positive semi-definite.
check_psd(T)


# Check if there are any identical pairs, i.e. T[i,j]=1 with i!=j.
# This is actually key to the matrix notation of GPs. 
# If identical rows exist (or even such that are linearly dependent), the matrix is singular (has determinant of 0)
# and thus has no inverse. This would confirm that adding noise at this stage is essential to solve the GP equation.

identical_pairs = [(i+1, j+1) for i in range(n) for j in range(i+1, n) if T[i, j] >= 0.9999]

if identical_pairs:
    print(f"\n  Pairs with Tanimoto = 1.000 (identical fingerprints):")
    for a, b in identical_pairs:
        print(f"    mol_{a:02d} and mol_{b:02d}")
else :
    print("\n  There are no pairs of identical structures based on their fingerprints!")


# Save Tanimoto matrix
np.save("tanimoto_matrix.npy", T)


