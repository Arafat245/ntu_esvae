import os
import warnings
os.environ['GEOMSTATS_BACKEND'] = 'pytorch'

import torch
torch.set_default_dtype(torch.float32)
torch.set_default_device('cuda:1')
device = torch.device('cuda:1')
# print('functionsgpu_old.py device:', device)

import geomstats.backend as gs
import numpy as np
from geomstats.geometry.pre_shape import PreShapeSpace
from geomstats.visualization import KendallDisk, KendallSphere
import fdasrsf
import matplotlib.pyplot as plt
from geomstats.geometry.matrices import Matrices
from tqdm.notebook import tqdm
from joblib import Parallel, delayed

preshape = PreShapeSpace(25, 3)
# preshape = PreShapeSpace(3, 2)
preshape.equip_with_group_action("rotations")
preshape.equip_with_quotient()


def OPA_gpu(A_t, B_t, reflect=False):
    
    """ Alignes A to B """       
    if reflect:
        # Correctly transpose A_t and B_t to enable proper matrix multiplication
        A = A_t.permute(2, 1, 0)  # (200, 3, 32)
        B = B_t.permute(2, 1, 0)  # (200, 3, 32)
        
        # Transpose A to get (200, 32, 3)
        A_transposed = A.permute(0, 2, 1)  # (200, 32, 3)

        # Now perform batch matrix multiplication across the 'sample' dimension
        product = B @ A_transposed
        u, sigma, v_t = torch.linalg.svd(product, full_matrices=False)
        R = u @ v_t

        # Apply R to the original A matrix, properly transposed to match dimensions
        aligned = R @ A
        
        # returning numpy array
        return aligned.permute(2, 1, 0)  # Back to (32, 3, 200)
    else:
        # converting (32, 3, 200) to (200, 32, 3)
        A = A_t.permute(2, 0, 1).double()
        B = B_t.permute(2, 0, 1).double()

        # aligning
        aligned =  Matrices.align_matrices(A, B)
        # returning numpy array
        return aligned.permute(1, 2, 0).float()  # Back to (32, 3, 200)


def rotate_trajectory_align_gpu(mu, traj, reflect=False):
    """Batch-aligns trajectories with mu using the modified OPA."""
    # Call the batched OPA directly
    traj_aligned = OPA_gpu(traj, mu, reflect=reflect)
    return traj_aligned
    

def log_gpu(p1, p2):
    # p1: landmarks x ambient x time
    # p2: landmarks x ambient x time
    p1 = p1.permute(2, 0, 1)  # (32, 3, 200) to (200, 32, 3)
    p2 = p2.permute(2, 0, 1)
    # geomstats uses float64 internally; cast to avoid "Float did not match Double"
    result = preshape.quotient.metric.log(p2.double(), p1.double())
    return result.permute(1, 2, 0).float()


def log_gpu_batch(p1, p2):
    # p1: batch x landmarks x ambient x time
    # p2: batch x landmarks x ambient x time
    p1 = p1.permute(0, 3, 1, 2)  # (155, 32, 3, 200) to (155, 200, 32, 3)
    p2 = p2.permute(0, 3, 1, 2)
    # geomstats uses float64 internally; cast to double then back to float
    p1_d = p1.double()
    p2_d = p2.double()
    result = preshape.quotient.metric.log(p2_d, p1_d)
    return result.permute(0, 2, 3, 1).float()


def log_gpu_frechet(p1, p2):
    # p1: (32, 3, 200), a single set of landmarks across time
    # p2: (130, 32, 3, 200), a batch of sets of landmarks across time
    
    # First, add a new dimension at the beginning of p1, then expand to match p2's size
    p1_expanded = p1.unsqueeze(0).expand(p2.shape[0], -1, -1, -1)  # Expand p1 to (130, 32, 3, 200)
    
    # Permute to match the expected dimensions for processing
    p1_expanded = p1_expanded.permute(0, 3, 1, 2).double()  # (130, 200, 32, 3)
    p2 = p2.permute(0, 3, 1, 2).double()  # (130, 200, 32, 3)

    # Calculate log map in a vectorized form across all samples and time points
    result = preshape.quotient.metric.log(p2, p1_expanded)  # Assumes this function can handle the batch dimension
    # Permute back to the original dimensions
    result = result.permute(2, 3, 1, 0).float()  # Back to (32, 3, 200, 130)

    return result


def exp_gpu(p, v):
    # p: landmarks x ambient x time
    p = p.permute(2, 0, 1).double()
    v = v.permute(2, 0, 1).double()
    result = preshape.metric.exp(v, p)
    return result.permute(1, 2, 0).float()

def squared_geodesic_distance(x, y, K, M, T):
    N = x.shape[0]
    x_shapes = x.view(N, K, M, T)
    y_shapes = y.view(N, K, M, T)
    x_shapes = x_shapes.permute(0, 3, 1, 2).double()
    y_shapes = y_shapes.permute(0, 3, 1, 2).double()
    result = preshape.metric.squared_dist(x_shapes, y_shapes)
    return result.float()


def exp_gpu_batch(p, v):
    # p: landmarks x ambient x time
    # v: batch x landmarks x ambient x time
    # First, add a new dimension at the beginning of p, then expand to match v's size
    p_expanded = p.unsqueeze(0).expand(v.shape[0], -1, -1, -1)  # Expand p to (130, 32, 3, 200)
    
    # Permute to match the expected dimensions for processing
    p_expanded = p_expanded.permute(0, 3, 1, 2)  # (130, 200, 32, 3)
    v = v.permute(0, 3, 1, 2)

    # geomstats uses float64 internally; cast to double then back to float
    p_expanded_d = p_expanded.double()
    v_d = v.double()

    result = preshape.metric.exp(v_d, p_expanded_d)
    return result.permute(0, 2, 3, 1).float()


def parallel_gpu(v, p1, p2, n_steps=10):
    v = v.permute(2, 0, 1).double()  # (32, 3, 200) to (200, 32, 3)
    p1 = p1.permute(2, 0, 1).double()
    p2 = p2.permute(2, 0, 1).double()
    result = preshape.quotient.metric.parallel_transport(tangent_vec=v, base_point=p1, end_point=p2, n_steps=2)
    return result.permute(1, 2, 0).float()
    

def preprocess(x):
    """Removes translations and scaling from a k (landmarks) x m (ambient dimension, eg. 2 for 2d shapes)"""
    mu = x.mean(axis=0)
    for i in range(x.shape[0]):
        x[i, :] = x[i, :] - mu
    x = x / np.linalg.norm(x, ord="fro")
    return x
    

def preprocess_temporal(data):
    """Mean centers data and removes scaling for kendall shape space"""
    for t in range(data.shape[2]):
        data[:, :, t] = preprocess(data[:, :, t])

    return data


def cov_der_gpu(beta_t, delta_t, c):
    """Given a function, calculate a derivative in the tangent space of beta(t)"""
    beta_dot_t = torch.zeros_like(beta_t)

    # Compute the log differences for all t except the last one, in parallel
    beta_dot_t[:, :, :-1] = log_gpu(beta_t[:, :, :-1], beta_t[:, :, 1:]) / delta_t

    # Handle the last point separately if required
    pt = parallel_gpu(beta_dot_t[:, :, -2].unsqueeze(2), beta_t[:, :, -2].unsqueeze(2), beta_t[:, :, -1].unsqueeze(2))
    beta_dot_t[:, :, -1] = pt.squeeze(2)

    # Convert the result back to a numpy array
    return beta_dot_t


def parallel_vf_gpu(v, beta, c):
    """Transports the entire vector field v to the tangent spaces at beta to the tangent space of reference point c"""
    c = c.unsqueeze(2)
    
    # Use parallel_gpu to process the entire batch
    parallel_vf = parallel_gpu(v, beta, c)
    
    return parallel_vf


def srvf_gpu(beta_dot_t, delta_t):

    # Calculating norm of each slice along the first two axes
    norms = torch.linalg.norm(beta_dot_t, dim=(0, 1), keepdim=True)

    thresh = 0.0000001
    # Clamping the norm values to avoid division by zero or very small numbers
    norms = torch.clamp(norms, min=thresh)

    # Normalize beta_dot_t by norms
    q_t = beta_dot_t / norms

    return q_t


def tsrvf(beta_t, delta_t, c):
    # beta_dot = cov_der(beta_t, delta_t)

    # beta_parallel = parallel_vf(beta_dot, beta, c)
    beta_dot = cov_der_gpu(beta_t, delta_t, c)

    beta_dot_c = parallel_vf_gpu(beta_dot, beta_t, c)

    q_t = srvf_gpu(beta_dot_c, delta_t)

    return q_t


def compose_gpu(beta, t, gamma):
    # Convert numpy arrays to torch tensors and move to GPU
    gamma = torch.from_numpy(gamma).to(device)

    # Pre-compute indices i for each gamma using searchsorted
    i = torch.searchsorted(t, gamma, right=True)
    i = torch.clamp(i, min=1, max=t.shape[0] - 1)  # Ensure indices are valid

    # Calculate delta_t for all gamma simultaneously
    delta_t = (gamma - t[i - 1]) / (t[i] - t[i - 1])

    # Prepare batch operations
    p1 = beta[:, :, i - 1]  # Points from beta corresponding to i-1
    p2 = beta[:, :, i]      # Points from beta corresponding to i

    # Batch compute the log map of the paths
    log_v = log_gpu(p1, p2)  # Perform batch log map
    v = log_v * delta_t.unsqueeze(0).unsqueeze(0)  # Apply delta_t scaling

    # Batch compute the exponential map
    beta_gamma = exp_gpu(p1, v)  # Perform batch exponential map

    return beta_gamma


def temporal_align(mu, beta, delta_t, q_mu_flat_np=None):
    """ mu, beta are two rotationally aligned trajectories.
    q_mu_flat_np: precomputed flat float64 numpy of tsrvf(mu) (skip recompute if given). """
    c = mu[:, :, 0]
    if q_mu_flat_np is None:
        q_mu = tsrvf(mu, delta_t, c)
        q_mu_flat_np = q_mu.reshape(-1, q_mu.shape[2]).cpu().numpy().astype(np.float64)
    q_beta = tsrvf(beta, delta_t, c)
    q_beta_flat = q_beta.reshape(-1, q_beta.shape[2])
    gamma_inv = fdasrsf.curve_functions.optimum_reparam_curve(
        q_mu_flat_np,
        q_beta_flat.cpu().numpy().astype(np.float64),
        method="DP",
    )
    return gamma_inv



def temporal_rotation_align(mu, beta, t, iterations=10, tol=10 ** (-5), reflect=False):
    prev_error = -10000
    delta_t = t[1] - t[0]
    history = []

    beta_hat = beta

    # mu is fixed across the inner loop, so tsrvf(mu) is too. Compute once.
    c = mu[:, :, 0]
    q_mu = tsrvf(mu, delta_t, c)
    q_mu_flat_np = q_mu.reshape(-1, q_mu.shape[2]).cpu().numpy().astype(np.float64)

    for iteration in range(iterations):
        error = torch.norm(mu - beta_hat)
        error = error.item()
        history.append(error)

        beta_hat = rotate_trajectory_align_gpu(mu, beta_hat, reflect=reflect)

        gamma_inv = temporal_align(mu, beta_hat, delta_t, q_mu_flat_np=q_mu_flat_np)

        beta_hat = compose_gpu(beta_hat, t, gamma_inv)

        if abs(error - prev_error) < tol:
            break
        else:
            prev_error = error

    return beta_hat, gamma_inv, history



def parallel_align(mu, betas, t):
    N = len(betas)

    def align(n):
        return temporal_rotation_align(mu, betas[n], t)

    results = Parallel(n_jobs=-1)(delayed(align)(n) for n in range(N))

    betas_aligned, gammas, temp_histories = zip(*results)

    return list(betas_aligned), list(gammas), list(temp_histories)


def process_kinematic(data, gamma_t):
    pids = data.keys()
    betas_resampled = []

    for i, pid in enumerate(pids):
        beta = preprocess_temporal(data[pid])
        t = torch.linspace(0, 1, steps=beta.shape[2])

        beta_resampled = compose_gpu(torch.from_numpy(beta).to(device), t, gamma_t).cpu().numpy()
        betas_resampled.append(beta_resampled)

    return betas_resampled


def frechet(betas, t, mu_init, iterations=50, plot=True, tol=10 ** (-5)):
    betas_orig = np.copy(betas)

    epsilon = 0.1
    prev_error = -10000

    history = []

    N = len(betas)

    # Quotient translation and scaling
    # for n in range(N):
    # betas[n] = preprocess_temporal(betas[n])

    mu = mu_init
    mu = torch.from_numpy(mu).to(device)
    betas = [torch.from_numpy(beta).to(device) for beta in betas]
    t = torch.from_numpy(t).to(device)

    for iteration in tqdm(range(iterations)):
        betas_aligned, gammas, temp_histories = parallel_align(mu, betas, t)  # we use the original betas here to avoid repeated sampling of same time series, leading to loss of information

        betas_aligned_torch = torch.stack(betas_aligned, dim=0)

    	# Compute all tangent vectors at once
        tangent_vec = log_gpu_frechet(mu, betas_aligned_torch)  # make sure to shoot a vector to the aligned betas
    	# Calculate mean tangent vector across the last dimension (N)
        mean_tangent_vec = torch.mean(tangent_vec, dim=3)

    	# update mu
        mu = exp_gpu(mu, epsilon * mean_tangent_vec)

        error = torch.linalg.norm(mean_tangent_vec) ** 2
        error = error.item()
        history.append(error)

        if abs(error - prev_error) < tol:
            break
        else:
            prev_error = error

        if plot:
            plt.figure()
            plt.plot(history)
            plt.show()

    mu = mu.cpu().numpy()
    betas_aligned = [beta.cpu().numpy() for beta in betas_aligned]
    tangent_vec = tangent_vec.cpu().numpy()
    
    return mu, betas_aligned, gammas, tangent_vec, history

