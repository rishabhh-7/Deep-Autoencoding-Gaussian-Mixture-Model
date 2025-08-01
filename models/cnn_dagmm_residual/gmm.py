import torch, torch.nn as nn, torch.nn.functional as F

class GMM(nn.Module):
    """
    A vectorised, differentiable—or EM—Gaussian Mixture.
    Use `requires_grad=False` if you want pure EM updates.
    """
    def __init__(self, n_components: int, embed_dim: int, device=None):
        # n_components: number of GMM components
        # embed_dim: dimensionality of the latent space (z), usually we take this as something small 2 or 3. 
        # device: device for GMM parameters (default: cpu)
        super().__init__()
        self.K, self.D = n_components, embed_dim
        device = device or "cpu"
        self.register_buffer("eps", torch.tensor(1e-8)) #a constant. defined to avoid division by zero in the EM step.
        # Init
        self.pi      = nn.Parameter(torch.ones(self.K, device=device) / self.K) #we know pi is trainable, hence we define it as nn.Parameter
        #intially we set pi to be uniform distribution, hence we divide by K. Basically a point is equally likely to belong to any of the K gaussians. 
        self.mu      = nn.Parameter(torch.randn(self.K, self.D, device=device))
        #mean vector, we initialize it randomly.
        self.cov_raw = nn.Parameter(
            torch.eye(self.D, device=device).repeat(self.K, 1, 1)
        )  # will be forced PSD
        #if 3 gaussians, then we have 3 covariance matrices, each of size DxD.

    @property #decorator that allows us to define a method as a property of the class.
    # @property decorator lets you call .cov like an attribute (e.g. gmm.cov) rather than a method (gmm.cov()).
    def cov(self):
        # guaranteed PSD (diagonal plus small jitter)
        eye = torch.eye(self.D, device=self.cov_raw.device).unsqueeze(0)
        return self.cov_raw @ self.cov_raw.transpose(-1, -2) + 1e-2 * eye
    #this gives valid covariance for each pass. 

    def forward(self, z):
        """
        z: [B, D], Let B be the minibatch size (number of samples) and D the embedding dimension.
        hence, say dim = 2 and b = 3, then 
        z = torch.tensor([
            [z₁₁, z₁₂],
            [z₂₁, z₂₂],
            [z₃₁, z₃₂],
        ])   

        returns log-likelihood per sample  [B]
        """
        mvn   = torch.distributions.MultivariateNormal(self.mu, self.cov)
           
        log_p = mvn.log_prob(z.unsqueeze(1))           # [B, K]
        log_p += torch.log_softmax(self.pi, 0)          # mixing weights
        return torch.logsumexp(log_p, dim=1)            # [B]
        #here we are returning a vector, [l1,l2,l3..lb] where li is the log likelihood of the ith sample.

    # --- EM update (call inside `torch.no_grad()` if you want manual updates)
    def em_step(self, z, gamma):
        """
        z:     [B, D]  latent codes for each sample
        gamma: [B, K]  responsibility of each of K components for each sample
        we get this gamma from the estimation network 
        """
        # --- 1) Effective counts N_k = sum_i gamma_ik  for k=1..K
        Nk = gamma.sum(dim=0) + self.eps       # shape [K]

        # --- 2) Update mixing weights: pi_k = N_k / sum_j N_j
        self.pi.data = (Nk / Nk.sum()).detach()

        # --- 3) Update means: mu_k = (1 / N_k) * sum_i gamma_ik * z_i
        #    gamma.T @ z  does sum over i:  shape [K,B] @ [B,D] -> [K,D]
        mu_new = (gamma.T @ z) / Nk.unsqueeze(1)  # shape [K, D]
        self.mu.data = mu_new.detach()

        # --- 4) Compute centered data for each sample-component:  z_i - mu_k
        #    z.unsqueeze(1): [B, 1, D]  -> broadcast to [B, K, D]
        #    mu_new.unsqueeze(0): [1, K, D]
        z_centered = z.unsqueeze(1) - mu_new      # shape [B, K, D]

        # --- 5) Covariance:  Sigma_k = (1/N_k) * sum_i gamma_ik · (z_i - mu_k)(z_i - mu_k)^T
        #    Using Einstein summation for clarity:
        #    "bk,bkd,bke->kde" means:
        #      for each k, d, e:  sum over i: gamma[i,k] * z_centered[i,k,d] * z_centered[i,k,e]
        cov_new = torch.einsum("bk,bkd,bke->kde", gamma, z_centered, z_centered)
        cov_new /= Nk.view(-1, 1, 1)             # normalize by N_k

        # --- 6) Store new Cholesky factors:
        #    We want L_k such that L_k L_k^T ≈ cov_new. 
        #    torch.linalg.cholesky returns a lower-triangular L_k.
        #    We add a small jitter to ensure cov_new + jitter*I is PSD.
        L_new = torch.linalg.cholesky(cov_new + 1e-6 * torch.eye(self.D, device=z.device))
        self.cov_raw.data = L_new.detach()

        '''
        We call .detach() so that those manual EM updates to π, μ, and L arent tracked by PyTorchs autograd:

        No gradient history: we dont want to compute or backpropagate gradients through those tensor assignments.

        Pure EM step: treating them as “hard” parameter resets, not part of the differentiable graph.
        '''