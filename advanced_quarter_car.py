import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from backend import vehicle 
from backend import vehicle_generator

file = 'driver.xlsx'
# creating of vehicle class
car = vehicle_generator(file)


g = 32.2 # ft/s^2

# linearly interpolates within damper lookup table
def lin_interp(axle, speed):
    upper_val = 0
    lower_val = 0

    if speed == 0:
        return 0
    # identifying the value that is higher than current speed
    for i,val in enumerate(axle.damp.speed):
        if val > speed:
            upper_val = val
            upper_ind = i
            lower_val = axle.damp.speed.iloc[i-1]
        else:
            print('Speed value was outside curve')
            exit()

    math_1 = (speed - axle.damp.speed.iloc[upper_ind-1])
    math_2 = (axle.damp.force.iloc[upper_ind] - axle.damp.force.iloc[upper_ind - 1])
    math_3 = axle.damp.speed.iloc[upper_ind] - axle.damp.speed.iloc[upper_ind-1]
    force = (math_1 * math_2)/math_3 + axle.damp.force.iloc[upper_ind-1]

    return force


def quarter_car(q, t, axle):
    q1,q2,q3,q4 = q
    if t < 5:
        r = 0
    elif t > 5:
        r = 1/12 # bump size [ft]
    curr_speed = q2-q4
    b = lin_interp(axle, curr_speed)
    q1dot = q2
    q2dot = -(axle.k/axle.sprung)*(q1-q3) - (b/axle.sprung)*(q2-q4) - g/axle.sprung
    q3dot = q4
    q4dot = (axle.k/axle.usprung)*(q1-q3) + (b/axle.usprung)*(q2-q4) - axle.kt*(q3-r) - g/axle.usprung


    # q1dot = q2
    # q2dot = -(k_s/m_s)*(q1-q3) - (b/m_s)*(q2-q4) - g/m_s
    # q3dot = q4
    # q4dot = (k_s/m_us)*(q1-q3) + (b/m_us)*(q2-q4) - k_t*(q3-r) - g/m_us

    states = [q1dot,q2dot,q3dot,q4dot]
    # returns vector of Sprung Disp, Sprung Velocity, Unsprung Disp, Unsprung Velocity
    return states

t = np.linspace(0,10,100000)
q = [0,0,0,0]


sol = odeint(quarter_car,q,t,args = (car.front,))

print(sol)
print(np.shape(sol))

label = ['Sprung Disp', 'Sprung Velocity', 'Unsprung Disp', 'Unsprung Velocity']
colors = ['k', 'b', 'r', 'g']
fig,ax = plt.subplots()
ax2 = ax.twinx()
for i,val in enumerate(label):
    if i == 1 or i == 3:
        ax.plot(t,sol[:,i], label = label[i], color = colors[i])
    else:
        ax2.plot(t,sol[:,i] * 12, label = label[i], color = colors[i])
fig.legend()
ax.set_ylabel('Velocities [ft/s]')
ax2.set_ylabel('Displacement [in]')
plt.show()
plt.close()
