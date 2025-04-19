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
def target_log_pdf(x):
    """
    Compute the log of the target probability density for a given point x.
    Here, the target is a standard Gaussian N(0, I) in d dimensions (d=2 for this example).
    We return the log of the Gaussian PDF.
    The normalization constant is omitted because it cancels out in the Metropolis-Hastings ratio.
    
    Parameters:
        x (numpy.ndarray): A 1D array of length d representing a point in the state space.
    
    Returns:
        float: The log of the target density at x (proportional to the probability up to a constant).
    """
    # For a standard normal, log PDF = -0.5 * x^T x + constant. We omit the constant term.
    return -0.5 * np.dot(x, x)


class MetropolisHastings:
    """Metropolis-Hastings MCMC sampler for a given target distribution."""
    
    def __init__(self, target_log_pdf, dim, proposal_scale=1.0, initial_state=None, seed=None):
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
        self.target_log_pdf = target_log_pdf
        self.dim = dim
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
def plot_textbook_diagnostics(samples, true_mean, true_cov):
    """
    Generate textbook-style diagnostic plots:
    1. Trace plots of parameters
    2. Samples vs true density comparison
    3. Numerical summary comparing sample/target statistics
    """
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    
    # Trace plots (original code style preserved)
    ax[0].plot(samples[:, 0], alpha=0.5)
    ax[0].plot(samples[:, 1], alpha=0.5)
    ax[0].set_title('Trace Plot (Textbook-style Enhancement)')
    ax[0].set_xlabel('Iteration')
    ax[0].set_ylabel('Parameter Value')
    
    # Samples vs true density (original scatter plot enhanced)
    x, y = np.mgrid[-3:3:.1, -3:3:.1]
    pos = np.dstack((x, y))
    true_dist = multivariate_normal(true_mean, true_cov)
    ax[1].contourf(x, y, true_dist.pdf(pos), alpha=0.3)
    ax[1].scatter(samples[:, 0], samples[:, 1], s=5, alpha=0.3, c='red')
    ax[1].set_title('Samples vs True Density (Enhancement)')
    
    # Text summary preserving original verification style
    sample_mean = np.mean(samples, axis=0).round(2)
    sample_cov = np.cov(samples.T).round(2)
    textstr = '\n'.join((
        f'True Mean: {true_mean}',
        f'Sample Mean: {sample_mean}',
        'True Cov:',
        np.array2string(np.array(true_cov)),
        'Sample Cov:',
        np.array2string(sample_cov)
    ))
    ax[2].text(0.5, 0.5, textstr, ha='center', va='center')
    ax[2].axis('off')
    plt.tight_layout()


# Example usage of the MetropolisHastings sampler:
if __name__ == "__main__":
    # Original comments preserved below
    # Set parameters for the sampler.
    dimension = 2          # We want to sample in 2 dimensions.
    steps = 10000          # Number of MCMC steps to run.
    proposal_std = 1.0     # Standard deviation of the proposal distribution.
    seed = 42              # Random seed for reproducibility.
    
    # Initialize the Metropolis-Hastings sampler for a 2D standard normal target.
    mh_sampler = MetropolisHastings(target_log_pdf=target_log_pdf, dim=dimension,
                                    proposal_scale=proposal_std, initial_state=None, seed=seed)
    
    # Textbook-style enhancement: Added burn-in period
    samples = mh_sampler.sample(num_steps=steps, burn_in=500)
    
    # Original verification code preserved
    # The `samples` array now contains the drawn samples from the target distribution.
    # (Here, we have `steps` points representing the Markov chain after the given number of steps.)
    # For verification, we examine the mean and covariance of the samples.
    # These should be approximately [0,0] for the mean vector and the identity matrix for covariance.
    sample_mean = np.mean(samples, axis=0)
    sample_cov = np.cov(samples, rowvar=False)
    print(f"Sample mean: {sample_mean}")
    print(f"Sample covariance matrix:\n{sample_cov}")
    
    # Original visualization preserved
    # Plot the samples to visualize the distribution.
    plt.figure(figsize=(6, 6))
    plt.scatter(samples[:, 0], samples[:, 1], s=10, alpha=0.5, edgecolors='none')
    plt.title('Metropolis-Hastings samples (2D standard normal)')
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.axis('equal')  # Equal scaling for both axes (so the circular shape isn't distorted).
    plt.grid(True)
    plt.show()
    
    # Added diagnostic plots (textbook incorporated)
    plot_textbook_diagnostics(samples, 
                             true_mean=[0, 0], 
                             true_cov=[[1, 0], [0, 1]])