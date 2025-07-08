class DAGMM(tf.keras.Model):
    def __init__(self, input_dim, latent_dim=4, num_components=3):
        super(DAGMM, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.num_components = num_components
        
        # Encoder
        self.encoder = tf.keras.Sequential([
            layers.Dense(60, activation='tanh'),
            layers.Dense(30, activation='tanh'),
            layers.Dense(10, activation='tanh'),
            layers.Dense(latent_dim, activation='tanh')
        ])
        
        # Decoder
        self.decoder = tf.keras.Sequential([
            layers.Dense(10, activation='tanh'),
            layers.Dense(30, activation='tanh'),
            layers.Dense(60, activation='tanh'),
            layers.Dense(input_dim)
        ])
        
        # Estimation Network
        self.estimation_net = tf.keras.Sequential([
            layers.Dense(10, activation='tanh'),
            layers.Dense(num_components, activation='softmax')
        ])
        
    def call(self, inputs):
        # Compression Network
        z = self.encoder(inputs)
        x_recon = self.decoder(z)
        
        # Reconstruction Error
        rec_error = tf.norm(inputs - x_recon, axis=1, keepdims=True)
        
        # Estimation Network Input
        est_input = tf.concat([z, rec_error], axis=1)
        gamma = self.estimation_net(est_input)
        
        # Compute GMM parameters
        gamma_sum = tf.reduce_sum(gamma, axis=0)
        phi = gamma_sum / tf.cast(tf.shape(gamma)[0], tf.float32)
        
        mu = tf.einsum('nk,ni->ki', gamma, est_input) / gamma_sum[:, tf.newaxis]
        
        # Compute covariance with regularization
        diff = est_input[:, tf.newaxis, :] - mu[tf.newaxis, :, :]
        sigma = tf.einsum('nk,nki,nkj->kij', gamma, diff, diff) / gamma_sum[:, tf.newaxis, tf.newaxis]
        sigma += tf.eye(est_input.shape[1], dtype=tf.float32)[tf.newaxis, :, :] * 1e-6
        
        # Calculate energy
        inv_sigma = tf.linalg.inv(sigma)
        exponent = -0.5 * tf.einsum('nki,kij,nkj->nk', diff, inv_sigma, diff)
        log_det = tf.linalg.logdet(sigma)
        log_prob = exponent - 0.5 * log_det[tf.newaxis, :] + tf.math.log(phi)[tf.newaxis, :]
        energy = -tf.reduce_logsumexp(log_prob, axis=1)
        
        # Loss components
        rec_loss = tf.reduce_mean(tf.square(inputs - x_recon))
        energy_loss = tf.reduce_mean(energy)
        cov_penalty = tf.reduce_sum(tf.linalg.diag_part(sigma)) * 0.005
        
        self.add_loss(rec_loss + 0.1 * energy_loss + cov_penalty)
        
        return x_recon
