import numpy as np

# offsets of each variable in the state vector
iX = 0
iV = 1
NUMVARS = iV + 1 # number of variables in the KF state vector

class KF:
    def __init__(self, initial_x: float, 
                        initial_v: float,
                        accel_variance: float) -> None: # returns nothing
        # mean of state Gaussian RV
        # self._x = np.array([initial_x, initial_v])
        self._x = np.zeros(NUMVARS)

        self._x[iX] = initial_x
        self._x[iV] = initial_v
        # acceleration variance
        self._accel_variance = accel_variance

        # covariance of state Gaussian RV
        # self._P = np.eye(2) # in practice, the uncertainty of initial speed and velocity are different
        self._P = np.eye(NUMVARS)

    def predict(self, dt: float) -> None: # dt: time passed between last and current step
        # x_{k+1} =  F*x_k   : New x
        # P_{k+1} =  F*P_k*F^T  + G*(\sigma_a^2)*G^T : New P
        F = np.eye(NUMVARS)
        F[iX, iV] = dt
        #F = np.array([[1,dt], [0, 1]])
        new_x = F.dot(self._x) 

        #G = np.array([0.5 * dt**2, dt]).reshape((2,1))
        G = np.zeros([2,1])
        G[iX] = 0.5 * dt**2
        G[iV] = dt
        new_P = F.dot(self._P).dot(F.T) + G.dot(self._accel_variance).dot(G.T)

        self._P = new_P
        self._x = new_x

    def update(self, meas_value: float, meas_variance: float): 
        # y = z - H*x_k
        # S_k = H*P_k*H^T + R : innovation covariance (C_{y,y})
        # K = P_k*H^T*S_k^-1  : Kalman gain
        # x_{k+1} = x + K*y : New x
        # P_{k+1} = (I - K*H)*P_k : New P

        H = np.array([1,0]).reshape(1,2)

        z = np.array([meas_value])
        R = np.array([meas_variance])

        y = z - H.dot(self._x)
        S = H.dot(self._P).dot(H.T) + R

        K = self._P.dot(H.T).dot(np.linalg.inv(S))

        new_x = self._x + K.dot(y)
        new_P = (np.eye(2) - K.dot(H)).dot(self._P)

        self._P = new_P
        self._x = new_x

    @property
    def cov(self) -> np.array:
        return self._P


    @property
    def mean(self) -> np.array:
        return self._x

    @property
    def pos(self) -> float:
        #return self._x[0]
        return self._x[iX]

    @property
    def vel(self) -> float:
        # return self._x[1]
        return self._x[iV]