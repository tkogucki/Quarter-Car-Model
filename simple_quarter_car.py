import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint


def quarter_car(q,t):
    k_s = 70050 # N/m
    k_t = 70050 # N/m
    m_s = 63 # kg
    m_us = 11.8 # kg 
    b = 1000
    q1,q2,q3,q4 = q
    if t < 1:
        r = 0
    elif t > 1:
        r = 0.1
    q1dot = q2
    q2dot = -(k_s/m_s)*(q1-q3) - (b/m_s)*(q2-q4)
    q3dot = q4
    q4dot = (k_s/m_us)*(q1-q3) + (b/m_us)*(q2-q4) - k_t*(q3-r)

    states = [q1dot,q2dot,q3dot,q4dot]
    return states

t = np.linspace(0,2,1000)
q = [0,0,0,0]
sol = odeint(quarter_car,q,t)
print(sol)
label = ['Sprung Disp', 'Sprung Velocity', 'Unsprung Disp', 'Unsprung Velocity']
for i,val in enumerate(label):
    plt.plot(t,sol[:,i], label = label[i])
# plt.plot(t,sol)
plt.legend()
plt.show()