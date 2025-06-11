from scipy.sparse import diags, eye, kron, csr_matrix

def build_laplacian(Nx: int, Ny: int, dx: float, dy: float) -> csr_matrix:
    # 1D second derivative matrices
    Dxx = diags([1, -2, 1], [-1, 0, 1], shape=(Nx, Nx)) / dx**2
    Dyy = diags([1, -2, 1], [-1, 0, 1], shape=(Ny, Ny)) / dy**2

    # 2D Laplacian via Kronecker product
    L = kron(eye(Ny), Dxx) + kron(Dyy, eye(Nx))
    return csr_matrix(L)