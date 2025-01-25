from copy import deepcopy
import numpy as np
from scipy.linalg import block_diag

np.random.seed(42)


def state_transition_matrix(dt):
    """
    Generates the state transition matrix for a given time step.

    Parameters:
    dt (float): The time step.

    Returns:
    numpy.ndarray: The state transition matrix.
    """
    return np.array([[1, 0, dt, 0], 
                     [0, 1, 0, dt], 
                     [0, 0, 1, 0], 
                     [0, 0, 0, 1]])


def control_input_matrix(dt):
    """
    Generates the control input matrix for a given time step.

    Parameters:
    dt (float): The time step.

    Returns:
    numpy.ndarray: The control input matrix.
    """
    return np.array([[0, 0, 0],
                     [0, 0.5 * dt**2, 0],
                     [0, 0, 0],
                     [0, dt, 0]])


def reshape_z(z, dim_z, ndim):
    """
    Ensures z is a (dim_z, 1) shaped vector.

    Parameters:
    z (array-like): The input array.
    dim_z (int): The desired dimension.
    ndim (int): The desired number of dimensions.

    Returns:
    numpy.ndarray: The reshaped array.
    """
    z = np.atleast_2d(z)
    if z.shape[1] == dim_z:
        z = z.T

    if z.shape != (dim_z, 1):
        raise ValueError(
            "z (shape {}) must be convertible to shape ({}, 1)".format(z.shape, dim_z)
        )

    if ndim == 1:
        z = z[:, 0]

    if ndim == 0:
        z = z[0, 0]

    return z


class KalmanFilter:
    """
    Implements a Kalman filter.

    Attributes:
    dim_x (int): Number of state variables.
    dim_z (int): Number of measurement variables.
    dim_u (int): Number of control inputs.
    x (numpy.ndarray): State estimate.
    P (numpy.ndarray): State covariance matrix.
    Q (numpy.ndarray): Process noise covariance matrix.
    B (numpy.ndarray): Control input matrix.
    F (numpy.ndarray): State transition matrix.
    H (numpy.ndarray): Measurement matrix.
    R (numpy.ndarray): Measurement noise covariance matrix.
    z (numpy.ndarray): Measurement.
    K (numpy.ndarray): Kalman gain.
    y (numpy.ndarray): Measurement residual.
    S (numpy.ndarray): System uncertainty.
    SI (numpy.ndarray): Inverse system uncertainty.
    x_prior (numpy.ndarray): Prior state estimate.
    P_prior (numpy.ndarray): Prior state covariance.
    x_post (numpy.ndarray): Posterior state estimate.
    P_post (numpy.ndarray): Posterior state covariance.
    """

    def __init__(self, dim_x, dim_z, dim_u=0):
        """
        Initializes the Kalman filter.

        Parameters:
        dim_x (int): Number of state variables.
        dim_z (int): Number of measurement variables.
        dim_u (int): Number of control inputs.
        """
        if dim_x < 1:
            raise ValueError('dim_x must be 1 or greater')
        if dim_z < 1:
            raise ValueError('dim_z must be 1 or greater')
        if dim_u < 0:
            raise ValueError('dim_u must be 0 or greater')

        self.dim_x = dim_x
        self.dim_z = dim_z
        self.dim_u = dim_u

        self.x = np.zeros((dim_x, 1))  # state
        self.P = np.eye(dim_x)  # uncertainty covariance
        self.Q = np.eye(dim_x)  # process uncertainty
        self.B = None  # control transition matrix
        self.F = np.eye(dim_x)  # state transition matrix
        self.H = np.zeros((dim_z, dim_x))  # measurement function
        self.R = np.eye(dim_z)  # measurement uncertainty
        self.z = np.array([[None] * self.dim_z]).T

        # gain and residual are computed during the innovation step. We
        # save them so that in case you want to inspect them for various
        # purposes
        self.K = np.zeros((dim_x, dim_z))  # kalman gain
        self.y = np.zeros((dim_z, 1))
        self.S = np.zeros((dim_z, dim_z))  # system uncertainty
        self.SI = np.zeros((dim_z, dim_z))  # inverse system uncertainty

        # identity matrix. Do not alter this.
        self._I = np.eye(dim_x)

        # these will always be a copy of x,P after predict() is called
        self.x_prior = self.x.copy()
        self.P_prior = self.P.copy()

        # these will always be a copy of x,P after update() is called
        self.x_post = self.x.copy()
        self.P_post = self.P.copy()

        self.inv = np.linalg.inv

    def predict(self, u=None, B=None, F=None, Q=None):
        """
        Predicts the next state of the system.

        Parameters:
        u (numpy.ndarray, optional): Control input.
        B (numpy.ndarray, optional): Control input matrix.
        F (numpy.ndarray, optional): State transition matrix.
        Q (numpy.ndarray, optional): Process noise covariance matrix.
        """
        if B is None:
            B = self.B
        if F is None:
            F = self.F
        if Q is None:
            Q = self.Q
        elif np.isscalar(Q):
            Q = np.eye(self.dim_x) * Q

        # x = Fx + Bu
        if B is not None and u is not None:
            self.x = np.dot(F, self.x) + np.dot(B, u)
        else:
            self.x = np.dot(F, self.x)

        # P = FPF' + Q
        self.P = np.dot(np.dot(F, self.P), F.T) + Q

        # save prior
        self.x_prior = self.x.copy()
        self.P_prior = self.P.copy()

    def update(self, z, R=None, H=None):
        """
        Updates the state of the system with a new measurement.

        Parameters:
        z (numpy.ndarray): Measurement.
        R (numpy.ndarray, optional): Measurement noise covariance matrix.
        H (numpy.ndarray, optional): Measurement matrix.
        """
        if z is None:
            self.z = np.array([[None] * self.dim_z]).T
            self.x_post = self.x.copy()
            self.P_post = self.P.copy()
            self.y = np.zeros((self.dim_z, 1))
            return

        if R is None:
            R = self.R
        elif np.isscalar(R):
            R = np.eye(self.dim_z) * R

        if H is None:
            z = reshape_z(z, self.dim_z, self.x.ndim)
            H = self.H

        # y = z - Hx
        # error (residual) between measurement and prediction
        self.y = z - np.dot(H, self.x)

        # common subexpression for speed
        PHT = np.dot(self.P, H.T)

        # S = HPH' + R
        # project system uncertainty into measurement space
        self.S = np.dot(H, PHT) + R
        self.SI = self.inv(self.S)

        # K = PH'inv(S)
        # map system uncertainty into kalman gain
        self.K = np.dot(PHT, self.SI)

        # x = x + Ky
        # predict new x with residual scaled by the kalman gain
        self.x = self.x + np.dot(self.K, self.y)

        # P = (I-KH)P(I-KH)' + KRK'
        # This is more numerically stable
        # and works for non-optimal K vs the equation
        # P = (I-KH)P usually seen in the literature.

        I_KH = self._I - np.dot(self.K, H)
        self.P = np.dot(np.dot(I_KH, self.P), I_KH.T) + np.dot(np.dot(self.K, R), self.K.T)

        # save measurement and posterior state
        self.z = deepcopy(z)
        self.x_post = self.x.copy()
        self.P_post = self.P.copy()


def order_by_derivative(Q, dim, block_size):
    """
    Orders a matrix Q by its derivative blocks.

    Parameters:
    Q (array-like): The input matrix to be ordered.
    dim (int): The dimension of the matrix.
    block_size (int): The size of the blocks.

    Returns:
    numpy.ndarray: The ordered matrix.
    """
    N = dim * block_size
    D = np.zeros((N, N))
    Q = np.array(Q)

    for i, x in enumerate(Q.ravel()):
        f = np.eye(block_size) * x
        ix, iy = (i // dim) * block_size, (i % dim) * block_size
        D[ix:ix + block_size, iy:iy + block_size] = f

    return D


def Q_discrete_white_noise(dim, dt=1.0, var=1.0, block_size=1, order_by_dim=True):
    """
    Generates a Q matrix for the Discrete White Noise process.

    Parameters:
    dim (int): Dimension of the matrix (2, 3, or 4).
    dt (float): Time step.
    var (float): Variance.
    block_size (int): Size of the blocks.
    order_by_dim (bool): Whether to order by dimension or not.

    Returns:
    numpy.ndarray: The generated Q matrix.
    """
    if dim not in [2, 3, 4]:
        raise ValueError("dim must be between 2 and 4")

    if dim == 2:
        Q = [
            [0.25 * dt**4, 0.5 * dt**3],
            [0.5 * dt**3, dt**2]
        ]
    elif dim == 3:
        Q = [
            [0.25 * dt**4, 0.5 * dt**3, 0.5 * dt**2],
            [0.5 * dt**3, dt**2, dt],
            [0.5 * dt**2, dt, 1]
        ]
    else:
        Q = [
            [(dt**6) / 36, (dt**5) / 12, (dt**4) / 6, (dt**3) / 6],
            [(dt**5) / 12, (dt**4) / 4, (dt**3) / 2, (dt**2) / 2],
            [(dt**4) / 6, (dt**3) / 2, dt**2, dt],
            [(dt**3) / 6, (dt**2) / 2, dt, 1.0]
        ]

    if order_by_dim:
        return block_diag(*[Q] * block_size) * var
    return order_by_derivative(np.array(Q), dim, block_size) * var
