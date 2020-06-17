import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

k_s = 70050 # N/m
k_t = 70050 # N/m
m_s = 63 # kg
m_us = 11.8 # kg 
b = 1000
g = 9.81 # m/s^2

def quarter_car(q,t):
 
    q1,q2,q3,q4 = q
    if t < 1:
        r = 0
    elif t > 1:
        r = 1 # bump size
    q1dot = q2
    q2dot = -(k_s/m_s)*(q1-q3) - (b/m_s)*(q2-q4) - g/m_s
    q3dot = q4
    q4dot = (k_s/m_us)*(q1-q3) + (b/m_us)*(q2-q4) - k_t*(q3-r) - g/m_us

    states = [q1dot,q2dot,q3dot,q4dot]
    # returns vector of Sprung Disp, Sprung Velocity, Unsprung Disp, Unsprung Velocity
    return states

t = np.linspace(0,3,1000)
q = [0,0,0,0]


sol = odeint(quarter_car,q,t)
def individual_force(q,t):
    # Test r-matrix
    r = np.zeros(1000)
    # sketchy recalculation of the input function
    for i,val in enumerate(t):
        if val > 1:
            r[i] = 1 # bump size
    q1v = q[:,0]
    q2v = q[:,1]
    q3v = q[:,2]
    q4v = q[:,3]

    f_spring = (k_s)*(q1v-q3v)
    f_damper = (b)*(q2v-q4v)
    f_tire = k_t*(q3v - r)
    # print(np.shape(f_tire))
    # print(np.shape(r))
    print(f'Random load in tire is {f_tire[400]}')
    return f_spring, f_damper, f_tire, r

    
f_spring, f_damper, f_tire,r = individual_force(sol, t)
print(sol)
print(np.shape(sol))

label = ['Sprung Disp', 'Sprung Velocity', 'Unsprung Disp', 'Unsprung Velocity']
for i,val in enumerate(label):
    plt.plot(t,sol[:,i], label = label[i])
plt.plot(t,r, label = 'Input')
plt.legend()
plt.show()
plt.close()
plt.plot(t, f_spring, label = 'Force Spring')
plt.plot(t, f_damper, label = 'Force Damper')
plt.plot(t, f_tire, label = 'Force Tire')
plt.legend()
plt.show()