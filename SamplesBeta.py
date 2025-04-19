# Metropolis-Hastings Sampler for a 2D Gaussian distribution
#
# This script implements the Metropolis-Hastings algorithm to sample from a 
# two-dimensional standard normal distribution (mean = [0, 0], covariance = I).
# The implementation is generalized to allow sampling in any number of dimensions 
# by adjusting the `dim` parameter and target distribution accordingly.
#
# In this implementation:
# - The target distribution is a standard multivariate normal.
# - The proposal distribution is a symmetric Gaussian centered at the current state.
# - We use the log of the target density to avoid numerical underflow when computing ratios.
# - The code records each sample (after each Metropolis step) in an array for analysis.

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal  # incorporation from textbook

# 2D (or d-dimensional) target log-density function for a standard normal distribution.
def target_log_pdf(x, a=2.0, beta=1.0):
    """
    Log-density for 2D double-well potential with adjustble parameters
    """
    log_p = -beta * (x[0]**2 - a**2)**2 #barrier height controlled by beta
    log_p -= 0.5 * x[1]**2 #simple Gaussian in 2D
    return log_p


class MetropolisHastings:
    """Metropolis-Hastings MCMC sampler for a given target distribution."""
    
    def __init__(self, target_params, dim=2, proposal_scale=1.0, initial_state=None, seed=None):
        """
        Initialize the Metropolis-Hastings sampler.
        
        Parameters:
            target_log_pdf (function): Function that computes log-density of target distribution at a given point.
            dim (int): Dimension of the target distribution (e.g., 2 for a bivariate distribution).
            proposal_scale (float): Standard deviation of the Gaussian proposal distribution.
            initial_state (array-like): Optional initial state vector. If None, uses the zero vector.
            seed (int or None): Optional random seed for reproducibility.
        """
        # Save parameters
        self.target_params = target_params
        self.dim = dim
        self.target_log_pdf = lambda x: target_log_pdf(x, **self.target_params)
        self.proposal_scale = proposal_scale
        
        # Set up the random number generator for reproducibility (if seed is provided).
        self.rng = np.random.default_rng(seed)
        
        # Initialize the chain state.
        if initial_state is None:
            # If no initial state is given, start at the origin (zero vector).
            self.current_state = np.zeros(self.dim)
        else:
            # Use the provided initial state (make sure it's a numpy array of correct shape).
            self.current_state = np.array(initial_state, dtype=float)
            if self.current_state.shape[0] != self.dim:
                raise ValueError("Initial state dimension does not match 'dim' parameter.")
        
        # Compute the log-density of the initial state.
        self.current_log_prob = self.target_log_pdf(self.current_state)
    
    def sample(self, num_steps, burn_in=0):  #  added burn_in parameter (incorporated from textbook)
        """
        Run the Metropolis-Hastings algorithm to generate samples.
        
        Parameters:
            num_steps (int): Number of MCMC steps to run (length of the chain to generate).
            burn_in (int): Number of initial samples to discard (textbook-style enhancement)
        """
        #  Handle burn-in period (incorporation from textbook)
        total_steps = num_steps + burn_in
        samples = np.zeros((num_steps, self.dim))
        
        for i in range(total_steps):
            # Propose a new state by sampling from a Gaussian centered at the current state.
            # This is a random-walk proposal: new_state = current_state + N(0, proposal_scale^2 * I).
            proposal = self.current_state + self.rng.normal(0, self.proposal_scale, size=self.dim)
            
            # Compute the log-density of the proposed state.
            proposal_log_prob = self.target_log_pdf(proposal)
            
            # Compute the log acceptance ratio: log(alpha) = log(p(proposal)) - log(p(current)).
            # Since we're using log probabilities, the subtraction gives the log of the ratio.
            log_accept_ratio = proposal_log_prob - self.current_log_prob
            
            # Decide whether to accept the proposal.
            # If log_accept_ratio is positive, exp(log_accept_ratio) > 1.
            # In this case, always accept the proposal (target density increased).
            # If log_accept_ratio is negative, accept the proposal with probability exp(log_accept_ratio).
            if log_accept_ratio >= 0 or self.rng.random() < np.exp(log_accept_ratio):
                # Accept the proposal:
                self.current_state = proposal
                self.current_log_prob = proposal_log_prob
            
            # Textbook-style enhancement: Only store post-burn-in samples
            if i >= burn_in:
                samples[i - burn_in] = self.current_state
        
        return samples


# Textbook-style enhancement: Added diagnostic plotting function
def plot_textbook_diagnostics(samples, a=2.0, beta=1.0):
    """
    Generate textbook-style diagnostic plots for the double-well potential.
    """
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    
    # Trace plots
    ax[0].plot(samples[:, 0], alpha=0.5, label='x0')
    ax[0].plot(samples[:, 1], alpha=0.5, label='x1')
    ax[0].set_title('Trace Plot')
    ax[0].legend()
    
    # Samples vs true density
    x, y = np.mgrid[-4:4:.1, -3:3:.1]
    pos = np.dstack((x, y))
    log_density = -beta * (x**2 - a**2)**2 - 0.5 * y**2
    density = np.exp(log_density - np.max(log_density))  # Normalize
    ax[1].contourf(x, y, density, alpha=0.3)
    ax[1].scatter(samples[:, 0], samples[:, 1], s=5, alpha=0.3, c='red')
    ax[1].set_title('Samples vs True Density')
    
    # Text summary
    sample_mean = np.mean(samples, axis=0).round(2)
    sample_cov = np.cov(samples.T).round(2)
    textstr = '\n'.join((
        f'Sample Mean: {sample_mean}',
        'Sample Cov:',
        np.array2string(sample_cov)
    ))
    ax[2].text(0.5, 0.5, textstr, ha='center', va='center')
    ax[2].axis('off')
    plt.tight_layout()

def track_convergence(samples, true_mean, true_cov):
    """Track error in mean/covariance vs number of samples."""
    mean_errors = []
    cov_errors = []
    
    for n in range(100, len(samples), 100):  # Check every 100 samples
        subsample = samples[:n]
        mean_error = np.linalg.norm(np.mean(subsample, axis=0) - true_mean)
        cov_error = np.linalg.norm(np.cov(subsample.T) - true_cov)
        mean_errors.append(mean_error)
        cov_errors.append(cov_error)
    
    return mean_errors, cov_errors

if __name__ == "__main__":
    # Parameters to test
    betas = [0.5, 1.0, 2.0, 4.0]  # Barrier heights
    a = 2.0                        # Mode separation
    steps = 20000                   # Total samples per run
    burn_in = 2000                  # Discarded samples
    proposal_std = 2.0              # Proposal scale (fixed for comparison)
    seed = 42

    # True statistics for double-well potential (x0 bimodal, x1 Gaussian)
    true_mean = [0.0, 0.0]
    true_cov = [[4.0, 0.0], [0.0, 1.0]]  # Cov(x0, x0) = 4 due to bimodality

    # Track convergence for different betas
    plt.figure(figsize=(10, 6))
    for beta in betas:
        # Initialize sampler
        mh = MetropolisHastings(
            target_params={'a': a, 'beta': beta},
            dim=2,
            proposal_scale=proposal_std,
            seed=seed
        )
        
        # Generate samples
        samples = mh.sample(num_steps=steps, burn_in=burn_in)
        
        # Compute convergence metrics
        mean_errors, cov_errors = track_convergence(samples, true_mean, true_cov)
        
        # Plot mean error vs sample count
        x_vals = 100 * np.arange(1, len(mean_errors) + 1)  
        plt.plot(
            x_vals,
            mean_errors,
            label=f'beta={beta}'
        )

    plt.xlabel("Number of Samples")
    plt.ylabel("Mean Absolute Error")
    plt.title("Convergence Speed vs Barrier Height (beta)")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Diagnostic plots for one beta (e.g., beta=1.0)
    mh = MetropolisHastings(
        target_params={'a': a, 'beta': 1.0},
        dim=2,
        proposal_scale=proposal_std,
        seed=seed
    )
    samples = mh.sample(num_steps=steps, burn_in=burn_in)
    plot_textbook_diagnostics(samples, a=a, beta=1.0)