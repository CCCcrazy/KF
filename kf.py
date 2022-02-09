import numpy as np   #create alias np

class KF:
    def __init__(self, initial_x: float, 
                       initial_v: float,
                       accel_variance: float) -> None:
        self._state = np.zeros(2)

        self._state[0] = initial_x
        self._state[1] = initial_v
        
        self._accel_variance = accel_variance   #this is the system noise, we assumes that the acceletion mean is 0

        # covariance of state 
        self._P = np.eye(2)   #[[1,0],[1,0]] initial 

    def predict(self, dt: float) -> None:
        
        F = np.eye(2) 
        F[0, 1] = dt    #F=[[1,dt],[0,1]]
        new_state = F.dot(self._state) # x = F.x

        G = np.zeros((2, 1))  #2 rows and 1 column
        G[0] = 0.5 * dt**2
        G[1] = dt #G=[[0.5 * dt**2],[dt]]
        new_P = F.dot(self._P).dot(F.T) + G.dot(G.T) * self._accel_variance  # P = F P Ft + G Gt a

        self._P = new_P
        self._state = new_state

    def update(self, meas_value: float, meas_variance: float):
        H = np.zeros((1, 2)) #1 row and 2 columns
        H[0, 1] = 1 #H=[[0,1]]

        z = np.array([meas_value])    #this is the measurement value in array form
        R = np.array([meas_variance])   # this is the measurement noise

        y = z - H.dot(self._state)     # y = z - H x
        S = H.dot(self._P).dot(H.T) + R  # S = H P Ht + R

        K = self._P.dot(H.T).dot(np.linalg.inv(S))  # K = P Ht S^-1

        new_state = self._state + K.dot(y) # x = x + K y
        new_P = (np.eye(2) - K.dot(H)).dot(self._P)  # P = (I - K H) * P

        self._P = new_P
        self._state = new_state

    @property
    def cov(self) -> np.array:
        return self._P

    @property
    def mean(self) -> np.array:
        return self._state

    @property
    def pos(self) -> float:
        return self._state[0]

    @property
    def vel(self) -> float:
        return self._state[1]
