import numpy as np
from scipy.stats import norm

np.random.seed(42) 

class ParticleFilter:
    """
    A Particle Filter implementation for tracking state using particles.

    Parameters:
    -----------
    num_particles : int
        The number of particles to use in the filter.
    initial_state : array-like
        The initial state from which to generate particles.
    state_std : float
        The standard deviation for state transitions.
    obs_std : float
        The standard deviation for observations.
    init : str, optional
        The initialization method ('gauss' or 'uniform'), default is 'gauss'.
    init_pos_std : float, optional
        The standard deviation for position initialization, default is 1.
    init_vel_std : float, optional
        The standard deviation for velocity initialization, default is 2.
    state_bounds : list of tuple, optional
        The bounds for uniform initialization, required if init='uniform'.
    """

    def __init__(
        self, num_particles, initial_state, state_std, obs_std, 
        init="gauss", init_pos_std=1, init_vel_std=2, state_bounds=None
    ):
        self.num_particles = num_particles
        self.state_std = state_std
        self.obs_std = obs_std

        self.particles = np.zeros((self.num_particles, 8))

        # Initialize particles
        if init == "gauss":
            self.particles[:, 0] = np.random.normal(
                initial_state[0], init_pos_std, self.num_particles
            )
            self.particles[:, 1] = np.random.normal(
                initial_state[1], init_pos_std, self.num_particles
            )
            self.particles[:, 2] = np.random.normal(
                initial_state[2], init_vel_std, self.num_particles
            )
            self.particles[:, 3] = np.random.normal(
                initial_state[3], init_vel_std, self.num_particles
            )
            self.particles[:, 4] = np.random.normal(
                initial_state[4], init_pos_std, self.num_particles
            )
            self.particles[:, 5] = np.random.normal(
                initial_state[5], init_pos_std, self.num_particles
            )
            self.particles[:, 6] = np.random.normal(
                initial_state[6], init_vel_std, self.num_particles
            )
            self.particles[:, 7] = np.random.normal(
                initial_state[7], init_vel_std, self.num_particles
            )
        elif init == "uniform":
            if state_bounds is None or len(state_bounds) != 8:
                raise ValueError(
                    "state_bounds must be provided and must have 8 bounds when using uniform initialization"
                )
            for i, (low, high) in enumerate(state_bounds):
                self.particles[:, i] = np.random.uniform(low, high, self.num_particles)
        else:
            raise ValueError("Unknown initialization method: {}".format(init))

        self.weights = np.ones(num_particles) / num_particles

    def predict(self, dt, g=9.81):
        """
        Predict the next state of the particles based on the motion model.

        Parameters:
        -----------
        dt : float
            Time step for the prediction.
        g : float, optional
            Gravitational constant, default is 9.81.
        """
        noise = np.random.normal(0, self.state_std, (self.num_particles, 8))
        self.particles[:, 0] += self.particles[:, 2] * dt + noise[:, 0]
        self.particles[:, 1] += self.particles[:, 3] * dt - 0.5 * g * dt ** 2 + noise[:, 1]
        self.particles[:, 2] += noise[:, 2]
        self.particles[:, 3] += -(g * dt) + noise[:, 3]
        self.particles[:, 4] += self.particles[:, 6] * dt + noise[:, 4]
        self.particles[:, 5] += self.particles[:, 7] * dt - 0.5 * g * dt ** 2 + noise[:, 5]
        self.particles[:, 6] += noise[:, 6]
        self.particles[:, 7] += -(g * dt) + noise[:, 7]

    def update(self, observations):
        """
        Update the particle weights based on observations.

        Parameters:
        -----------
        observations : array-like
            Observations to update the weights.
        """
        distances1 = np.linalg.norm(self.particles[:, :2] - observations[0], axis=1)
        distances2 = np.linalg.norm(self.particles[:, 4:6] - observations[1], axis=1)

        self.weights1 = norm.pdf(distances1, loc=0, scale=self.obs_std)
        self.weights2 = norm.pdf(distances2, loc=0, scale=self.obs_std)

        self.weights = self.weights1 * self.weights2
        self.weights += 1.e-300  # avoid round-off to zero
        self.weights /= np.sum(self.weights)

    def resample(self):
        """
        Resample particles based on their weights.
        """
        indices = np.random.choice(
            range(self.num_particles), self.num_particles, p=self.weights
        )
        self.particles = self.particles[indices]
        self.weights = self.weights[indices]

    def neff(self):
        """
        Compute the effective sample size.

        Returns:
        --------
        float
            The effective sample size.
        """
        return 1. / np.sum(np.square(self.weights))

    def systematic_resample(self):
        """
        Perform systematic resampling of particles.

        Returns:
        --------
        array
            Array of indices for resampled particles.
        """
        N = len(self.weights)

        # make N subdivisions, and choose positions with a consistent random offset
        positions = (np.random.random() + np.arange(N)) / N

        indexes = np.zeros(N, 'i')
        cumulative_sum = np.cumsum(self.weights)
        i, j = 0, 0
        while i < N:
            if positions[i] < cumulative_sum[j]:
                indexes[i] = j
                i += 1
            else:
                j += 1
        return indexes

    def resample_from_index(self, indexes):
        """
        Resample particles from given indices.

        Parameters:
        -----------
        indexes : array-like
            Indices to resample particles.
        """
        self.particles[:] = self.particles[indexes]
        self.weights.resize(len(self.particles))
        self.weights.fill(1.0 / len(self.weights))

    def estimate(self):
        """
        Estimate the state based on the particles and their weights.

        Returns:
        --------
        array
            The estimated state.
        """
        return np.average(self.particles, weights=self.weights, axis=0)
