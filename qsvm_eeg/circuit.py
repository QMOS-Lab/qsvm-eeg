import pennylane as qml
from pennylane.templates import AngleEmbedding
import numpy as np
import time

N_QUBITS = 11
dev_kernel = qml.device("lightning.qubit", wires=N_QUBITS)


@qml.qnode(dev_kernel)
def kernel_circuit(x1, x2):
    AngleEmbedding(x1, wires=range(N_QUBITS))
    qml.adjoint(AngleEmbedding)(x2, wires=range(N_QUBITS))
    return qml.probs(wires=range(N_QUBITS))


def compute_kernel_matrix(X_A, X_B, verbose=True):
    n_A, n_B = len(X_A), len(X_B)
    matrix = np.zeros((n_A, n_B))

    if verbose:
        print(f"   Computing Quantum Kernel ({n_A} x {n_B})...")

    start_t = time.time()
    for i in range(n_A):
        for j in range(n_B):
            matrix[i, j] = kernel_circuit(X_A[i], X_B[j])[0]

        if verbose and (i + 1) % 10 == 0:
            elapsed = time.time() - start_t
            rate = (i + 1) / elapsed
            remaining = (n_A - (i + 1)) / rate
            print(f"   Row {i + 1}/{n_A} | Rate: {rate:.1f} rows/s | ETA: {remaining:.0f}s", end='\r')

    if verbose:
        print(f"\n   Done in {time.time() - start_t:.1f}s")

    return matrix
