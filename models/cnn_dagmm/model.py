import torch
import torch.nn.functional as F
from torch import nn

from compression_network import CompressionNetwork
from estimation_network import EstimationNetwork
from gmm import GMM

class DAGMM(nn.Module):
    """
    Deep Autoencoding Gaussian Mixture Model using separate Compression and Estimation modules.

    Args:
        input_dim (int): Dimensionality of the input data.
        latent_dim (int): Dimensionality of the latent embedding.
        n_gmm_components (int): Number of Gaussian mixture components.
        comp_kwargs (dict, optional): kwargs for CompressionNetwork (__init__).
        est_kwargs (dict, optional): kwargs for EstimationNetwork (__init__).
        device (str | torch.device, optional): Device for GMM parameters.
    """
    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        n_gmm_components: int,
        comp_kwargs: dict | None = None,
        est_kwargs: dict | None = None,
        device: str | torch.device = "cpu",
    ):
        super().__init__()

        # 1) Compression AE
        ck = comp_kwargs or {}
        ck.setdefault('latent_dim', latent_dim)
        self.compression = CompressionNetwork(**ck)

        # 2) Estimation network (now expecting latent_dim + 2 inputs)
        ek = est_kwargs or {}
        ek.setdefault('input_dim',  latent_dim + 2)
        ek.setdefault('output_dim', n_gmm_components)
        self.estimation = EstimationNetwork(**ek)

        

                # 3) GMM on [z, recon_error, cos_sim, gamma]
        gmm_feat_dim = latent_dim + 2 + n_gmm_components
        self.gmm = GMM(n_gmm_components, gmm_feat_dim, device=device)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        # 1) Encode
        z = self.compression.encode(x)

        # 2) Decode / reconstruct
        x_hat = self.compression.decode(z)

        # 3) Reconstruction error: flatten pixels so norm over all dims
        diff = (x - x_hat).view(x.size(0), -1)                    # [B, 3*64*64]
        recon_error = torch.norm(diff, p=2, dim=1, keepdim=True)  # [B, 1]

        # 4) Additional feature: cosine similarity between x and x_hat
        xhat_flat = x_hat.view(x.size(0), -1)                     # [B, 3*64*64]
        cos_sim = F.cosine_similarity(diff, xhat_flat, dim=1, eps=1e-8).unsqueeze(1)  # [B, 1]

        # 5) Estimate mixture responsibilities on [z, recon_error, cos_sim]
        gamma = self.estimation(torch.cat([z, recon_error, cos_sim], dim=1))

        # 6) GMM features: include cos_sim as well
        features = torch.cat([z, recon_error, cos_sim, gamma], dim=1)

        # 7) Sample energy
        energy = -self.gmm(features)

        return {
            'x_hat':      x_hat,
            'z':          z,
            'recon_error':recon_error,
            'cos_sim':    cos_sim,
            'gamma':      gamma,
            'energy':     energy,
        }


    def loss_function(
        self,
        x: torch.Tensor,
        outputs: dict[str, torch.Tensor],
        lambda_energy: float = 0.1,
        lambda_cov: float = 0.05,
    ) -> torch.Tensor:
        """
        Loss = MSE reconstruction + lambda_energy * mean(energy) + lambda_cov * sum(cov_diag).
        """
        # MSE loss

        mse = self.compression.reconstruction_loss(x, outputs['x_hat'])
        energy = outputs['energy'].mean()

        # penalize tiny covariances to avoid singularity
        covs = self.gmm.cov_raw @ self.gmm.cov_raw.transpose(-1,-2)
        cov_penalty = torch.sum(1.0 / (covs.diagonal(dim1=1,dim2=2) + 1e-6))
        return mse + lambda_energy * energy + lambda_cov * cov_penalty
     
       

if __name__ == '__main__':
    # Minimal smoke test
    model = DAGMM(
        input_dim=30,
        latent_dim=10,
        n_gmm_components=5,
        comp_kwargs={'hidden_dims': [128, 64], 'activation': nn.ReLU},
        est_kwargs={'hidden_dims': [64, 32], 'activation': nn.Tanh, 'dropout': 0.5},
        device='cpu',
    )
    x = torch.randn(4, 30)
    out = model(x)
    print('z.shape=', out['z'].shape)
    print('x_hat.shape=', out['x_hat'].shape)
    loss = model.loss_function(x, out)
    print('Loss:', loss.item())
