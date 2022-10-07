import matplotlib.pyplot as plt
import numpy as np
import time as tm
import scipy
from sklearn.cluster import KMeans
from scipy.sparse.linalg import svds

rng = np.random.default_rng()


def quant_convlayer_weights(W, sub_dim, accel, clust_scheme='dl', coeff=None, sparsity_level=None, num_iter=10,
                            num_iter_init=10):
    num_kernels,num_channels,kernel_h,kernel_w=W.shape #Python order
    # kernel_h, kernel_w, num_channels, num_kernels = W.shape  # Matlab order
    W_quant = np.zeros_like(W)
    num_subspaces = round(num_channels / sub_dim);
    num_subvectors = num_kernels * kernel_h * kernel_w;

    subspace_channels_1 = np.arange(0, num_channels, sub_dim)
    subspace_channels_2 = subspace_channels_1 + sub_dim

    K_vq = int(round(num_subvectors / accel))

    print('###############################################')
    stats={}
    print('num_subspaces = {:d}, num_subvectors per subspace = {:d}, accel = {:d}'.format(num_subspaces, num_subvectors,
                                                                                          accel))

    stats['num_subspaces'] = num_subspaces
    stats['num_subvectors'] = num_subvectors
    stats['accel'] = accel

    if clust_scheme == 'dl':
        K_dl = int(round(K_vq * coeff))
        margin = int((sparsity_level * K_dl) / sub_dim)
        num_atoms = K_vq - margin
        if num_atoms > 0:
            print('DL-based quantization: num atoms = {:d}, K_dl = {:d}, sparsity_level = {:d}'.format(num_atoms, K_dl,
                                                                                                       sparsity_level))

            stats['clust_scheme'] = 'dl'
            stats['num_atoms'] =num_atoms
            stats['K_dl'] = K_dl

        else:
            print('Not a valid param combination: num_atoms = {:d}'.format(num_atoms))
            return -1
    elif clust_scheme == 'vq':
        print('kmeans-based quantization: K_vq = {:d}'.format(K_vq))
        stats['clust_scheme'] = 'vq'
        stats['K_vq'] = K_vq

    else:
        print('\'' + clust_scheme + '\'' + ' is not a valid clustering scheme')
        return -1

    tot_tic = tm.time()
    for sub_no in range(num_subspaces):
        sub_tic = tm.time()
        print('Subspace {:2d} of {:2d}: channels {:03d}:{:03d}...'.format(sub_no + 1, num_subspaces,
                                                                          subspace_channels_1[sub_no] + 1,
                                                                          subspace_channels_2[sub_no]), end='')
        W_sub_mat = np.zeros((sub_dim, num_subvectors))
        array_idx = np.zeros((num_subvectors, 3), dtype=int);
        count = 0
        for f1 in range(kernel_h):
            for f2 in range(kernel_w):
                for kernel in range(num_kernels):
                    W_sub_mat[:,count]=W[kernel,subspace_channels_1[sub_no]:subspace_channels_2[sub_no],f1,f2] #Python order
                    # W_sub_mat[:, count] = W[f1, f2, subspace_channels_1[sub_no]:subspace_channels_2[sub_no],kernel]  # Matlab order
                    array_idx[count, :] = np.array([f1, f2, kernel]);
                    count = count + 1
        if clust_scheme == 'dl':
            W_sub_mat_quant, res = dl_subspace_clustering(W_sub_mat, num_atoms, K_dl, sparsity_level, num_iter,
                                                          num_iter_init)
        else:
            W_sub_mat_quant, res = vq_subspace_clustering(W_sub_mat, K_vq)

        for count in range(num_subvectors):
            f1 = array_idx[count, 0]
            f2 = array_idx[count, 1]
            kernel = array_idx[count, 2]
            W_quant[kernel,subspace_channels_1[sub_no]:subspace_channels_2[sub_no],f1,f2]=W_sub_mat_quant[:,count] #Python order
            # W_quant[f1, f2, subspace_channels_1[sub_no]:subspace_channels_2[sub_no], kernel] = W_sub_mat_quant[:,count]  # Matlab order

        sub_toc = tm.time()
        print('Completed in {:4.1f} seconds.'.format(sub_toc - sub_tic))
    tot_toc = tm.time()
    print('Total elapsed time is {:6.1f} seconds.'.format(tot_toc - tot_tic))
    print('###############################################')
    return W_quant,stats


def vq_subspace_clustering(W_sub_mat, K_vq):
    sub_dim, num_subvectors = W_sub_mat.shape
    kmeans = KMeans(n_clusters=K_vq)
    kmeans.fit(W_sub_mat.T)
    I_vq = kmeans.predict(W_sub_mat.T)
    C_vq = kmeans.cluster_centers_

    W_sub_mat_quant = (C_vq[I_vq, :]).T
    res = (np.linalg.norm(W_sub_mat - W_sub_mat_quant, ord='fro') ** 2) / (sub_dim * num_subvectors)
    return W_sub_mat_quant, res


def dl_subspace_clustering(W_sub_mat, num_atoms, K_dl, sparsity_level, num_iter=10, num_iter_init=10, plot_flag=False):
    sub_dim, num_subvectors = W_sub_mat.shape
    norm_W = np.linalg.norm(W_sub_mat, axis=0) ** 2
    norm_W.shape = (1, num_subvectors)
    norm_W_sub_mat = norm_W.T @ np.ones((1, K_dl))

    # Initial Solution
    print('initial solution...', end='')
    init_tic = tm.time()
    D, A, G, I, res_0 = init_solution(W_sub_mat, num_atoms, K_dl, sparsity_level, num_iter_init, norm_W_sub_mat)
    init_toc = tm.time()
    print('done in {:4.1f} seconds. '.format(init_toc - init_tic), end='')

    res = np.zeros((num_iter + 1,))
    res[0] = res_0
    # Solve DL-based SS clustering with Target Sparsity >= 1
    for i in range(num_iter):
        # Sparse Coding
        A, AG = sparse_coding(W_sub_mat, D, G, sparsity_level, K_dl)

        # Dictionary Update
        D = dict_upd(W_sub_mat, D, AG)

        # Assignment vector update
        C_dl = D @ A
        G, I = assignment_upd(W_sub_mat, C_dl, norm_W_sub_mat)

        W_sub_mat_quant = C_dl[:, I]
        res[i + 1] = (np.linalg.norm(W_sub_mat - W_sub_mat_quant, ord='fro') ** 2) / (sub_dim * num_subvectors)

    if plot_flag:
        fig, axs = plt.subplots()
        axs.stem(np.arange(num_iter + 1), res)

        axs.set_xlim(-1, num_iter + 1)
        axs.set_ylim(.9 * min(res), 1.1 * max(res))
        axs.set_xlabel('Iteration')
        axs.set_ylabel('MSE')
        axs.set_title('Mean Sq Error')
        axs.grid(True)
        plt.show()

    return W_sub_mat_quant, res


def init_solution(W_sub_mat, num_atoms, K_dl, sparsity_level, num_iter=10, norm_W_sub_mat=None):
    sub_dim, num_subvectors = W_sub_mat.shape

    kmeans = KMeans(n_clusters=K_dl)
    kmeans.fit(W_sub_mat.T)
    # I=kmeans.predict(W_sub_mat.T)
    C_dl = kmeans.cluster_centers_.T
    # print('')
    # print((np.linalg.norm(W_sub_mat-C_dl[:,I],ord='fro')**2)/(sub_dim*num_subvectors))

    # Solve C_dl ~ D_0*A_0 (OMP & KSVD)
    # Intialize dictionary D
    D = rng.normal(size=(sub_dim, num_atoms))
    D = D / np.linalg.norm(D, axis=0)

    for i in range(num_iter):
        A = myOMP(C_dl, D, sparsity_level)
        D, A = myKSVD(C_dl, D, A)

    C_dl = D @ A
    G, I = assignment_upd(W_sub_mat, C_dl, norm_W_sub_mat)

    W_sub_mat_quant = C_dl[:, I]
    res = (np.linalg.norm(W_sub_mat - W_sub_mat_quant, ord='fro') ** 2) / (sub_dim * num_subvectors)
    return D, A, G, I, res


def sparse_coding(W_sub_mat, D, G, sparsity_level, K_dl):
    (sub_dim, num_atoms) = D.shape
    num_subvectors = W_sub_mat.shape[1]
    A = np.zeros((num_atoms, K_dl))
    AG = np.zeros((num_atoms, num_subvectors))

    Si = np.sum(G, axis=1)
    for j in range(K_dl):
        if Si[j] > 0:
            mean_Wi = np.sum(W_sub_mat[:, G[j, :]], axis=1) / Si[j]
            mean_Wi.shape = (sub_dim, 1)
            r = mean_Wi;
            S_ind = np.zeros((num_atoms,), dtype=bool)
            for s in range(sparsity_level):
                # print('s={:d}'.format(s),end=' ')
                I = ~S_ind
                I_array = np.zeros((num_atoms, 1))
                I_array[I] = 1
                D_corr = D.T @ r
                ind = np.argmax(abs(D_corr * I_array))
                S_ind[ind] = True
                Ds = D[:, S_ind]
                x = scipy.linalg.lstsq(Ds, mean_Wi)[0]

                r = mean_Wi - Ds @ x
            sparsity_level_updated=min(x.shape[0],sparsity_level)
            # print('Sparsity Level Updated: ' +str(sparsity_level_updated))
            x.shape = (sparsity_level_updated,)
            A[S_ind, j] = x
            # Construct A*G without the need for multiplication
            AG_j = A[:, j]
            AG_j.shape = (num_atoms, 1)
            AG[:, G[j, :]] = AG_j @ np.ones((1, Si[j]))
    return A, AG


def dict_upd(W_sub_mat, D, AG):
    (sub_dim, num_atoms) = D.shape
    E = W_sub_mat - D @ AG;
    for j in range(num_atoms):
        Ij = (AG[j, :] != 0)
        sj = sum(Ij)
        if sj > 0:
            E_j = E[:, Ij]
            # E_j.shape=(sub_dim,sj)
            D_j = D[:, j]
            D_j.shape = (sub_dim, 1)

            AG_Ij = AG[j, Ij]
            AG_Ij.shape = (1, sj)

            F = E_j + D_j @ AG_Ij
            d_unorm = F @ AG_Ij.T
            d_norm = d_unorm / np.linalg.norm(d_unorm)

            E[:, Ij] = F - d_norm @ AG_Ij

            d_norm.shape = (sub_dim,)
            D[:, j] = d_norm
    return D


def assignment_upd(W_sub_mat, C_dl, norm_W_sub_mat=None):
    num_subvectors = W_sub_mat.shape[1]
    K_dl = C_dl.shape[1]

    norm_C_dl_ = np.linalg.norm(C_dl, axis=0) ** 2
    norm_C_dl_.shape = (1, K_dl)
    norm_C_dl = np.ones((num_subvectors, 1)) @ norm_C_dl_

    if norm_W_sub_mat is None:
        norm_W = np.sum(np.power(W_sub_mat, 2), axis=0)
        norm_W.shape = (1, num_subvectors)
        norm_W_sub_mat = norm_W.T @ np.ones((1, K_dl))
    dist_mat = norm_W_sub_mat + norm_C_dl - 2 * (W_sub_mat.T @ C_dl)

    I = np.argmin(dist_mat, axis=1)

    G = np.zeros((K_dl, num_subvectors), dtype=bool)
    for j in range(K_dl):
        G[j, :] = (I == j)
    return G, I


def myOMP(Y, D, sparsity_level):
    # Unroll the following for help
    """
    Orthogonal Matching Pursuit.

    Implementation of the algorithim OMP in python.
    The following code solves the problem y=Dx+n, where x
    is an unknown sparse vector.

        x_est=myOMP_func(D,y,sparsity_level,e)

    INPUTS:
      D: (numpy array - 2D matrix): The matrix containing the dictionary (Length of atom X Number of atoms)
      Y: (numpy array - 2D matrix): The observation vectors (Length of atoms x num_obervations)
      sparsity_level: The maximum number of non-zero elements found in x_est. This could
         also be connected to the maximum number of iterations executed by OMP


    OUTPUTS:
      A: (2D numpy array) The estimated sparse vectors (Number of atoms x num_observations)

    """
    num_observations = Y.shape[1]
    len_obs = D.shape[0]
    num_atoms = D.shape[1]

    A = np.zeros((num_atoms, num_observations))
    for obs in range(num_observations):
        y = Y[:, obs]
        y.shape = (len_obs, 1)

        # Set of indices. At most sparsity_level indecies will be produced
        I = np.zeros((num_atoms,), dtype=bool, order='F')

        # Initialize the residual
        r = y.copy()

        # Initialize the estimated sparse vector
        x_est = np.zeros((num_atoms, 1))

        # The OMP iterations

        for i in range(sparsity_level):
            # Get current maximum index
            index = np.argmax(abs(D.T @ r))
            I[index] = True
            # Get new values (Note the 0:1 at x_est. It is used for keeping the 2D nature of x_est.
            x_est[I, :] = np.linalg.pinv(D[:, I]) @ y

            # Update the residual
            r = y - D[:, I] @ x_est[I, :]
        x_est.shape = (num_atoms,)
        A[I, obs] = x_est[I]
    return A


def myKSVD(Y, Dinit, A):
    len_atom = Dinit.shape[0]
    num_atoms = Dinit.shape[1]
    D = Dinit.copy()
    A_loc = A.copy()
    for i in range(num_atoms):
        D[:, i] = 0
        I = (A_loc[i, :] != 0)
        if sum(I) > 0:
            E = Y[:, I] - D @ A_loc[:, I]
            if sum(I) > 1:
                u, s, v = svds(E, k=1)
            else:
                # print(E.shape)
                s = np.array([np.linalg.norm(E)])
                v = np.array([1])
                u = E / s[0]
            u.shape = (len_atom,)
            D[:, i] = u
            v.shape = (sum(I),)
            A_loc[i, I] = s[0] * v
    return D, A_loc








