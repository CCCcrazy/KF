import numpy as np
import matplotlib.pyplot as plt

from kf import KF

plt.ion()   #interactive mode will be on
plt.figure()  #show figure


real_x = 0.0
meas_variance = 0.1 ** 2
real_v = 0.5     

kf = KF(initial_x=0.0, initial_v=1.0, accel_variance=0.1)

DT = 0.1
NUM_STEPS = 1000
EACH_MEAS_STEPS = 20

mus = []   #mus:array of mean of kf;mus[0] pos array using kf;mus[1] vel array using kf
covs = []  #array of cov
real_xs = [] #array of real position
real_vs = [] #array of real v

for step in range(NUM_STEPS): 
    covs.append(kf.cov)
    mus.append(kf.mean)

    real_x = real_x + DT * real_v   #here he assumes that the acceleration has  mean 0 

    kf.predict(dt=DT)
    if step != 0 and step %EACH_MEAS_STEPS== 0: #only update when takes measurement 
        kf.update(meas_value=real_x + np.random.randn() * np.sqrt(meas_variance),
                  meas_variance=meas_variance)   #Add measurement noise; measurement value not equal to real value
               
    real_xs.append(real_x)
    real_vs.append(real_v)


plt.subplot(2, 1, 1)
plt.title('Position')
plt.plot([mu[0] for mu in mus], 'r')     #position
plt.plot(real_xs, 'b')
plt.plot([mu[0] - 2*np.sqrt(cov[0,0]) for mu, cov in zip(mus,covs)], 'r--')
plt.plot([mu[0] + 2*np.sqrt(cov[0,0]) for mu, cov in zip(mus,covs)], 'r--')  #uncertainty

plt.subplot(2, 1, 2)
plt.title('Velocity')
plt.plot(real_vs, 'b')
plt.plot([mu[1] for mu in mus], 'r')   #velocity
plt.plot([mu[1] - 2*np.sqrt(cov[1,1]) for mu, cov in zip(mus,covs)], 'r--')
plt.plot([mu[1] + 2*np.sqrt(cov[1,1]) for mu, cov in zip(mus,covs)], 'r--')   #uncertainty

plt.show()
plt.ginput(1)

