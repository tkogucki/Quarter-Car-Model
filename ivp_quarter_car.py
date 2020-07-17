import numpy as np
import matplotlib.pyplot as plt
import scipy
import pandas as pd
from scipy.integrate import odeint
from backend import vehicle 
from backend import vehicle_generator

file = 'driver.xlsx'
# creating of vehicle class


g = 32.2 # ft/s^2

# creates a ramp for the vehicle tire
def ramp(start_ramp, end_ramp, height_ramp, t):
    if t < start_ramp:
        return 0
    elif t > end_ramp:
        return height_ramp
    else:
        slope = height_ramp/(end_ramp - start_ramp)
        delta = t - start_ramp
        return slope * delta



# linearly interpolates within damper lookup table
def lin_interp(axle, speed):
    upper_val = 0
    lower_val = 0
    speed_val = abs(speed)
    sign = speed/speed_val
    print(f'The current speed is {speed}')
    length = axle.damp.length
    if speed == 0:
        return 0
    elif (speed_val > 1.2 * axle.damp.speed.iloc[length-1]):
        delta_speed = speed_val - axle.damp.speed.iloc[length - 1]
        force = axle.damp.max_damp
        force = sign * force
        print('1.2 Speed Limit Hit')
        return float(force)
    # linearly extrapolates last damping value
    elif (axle.damp.speed.iloc[length-1] <= speed_val):
        delta_speed = speed_val - axle.damp.speed.iloc[length - 1]
        force = delta_speed * axle.damp.slope
        force = sign * force # this is in order to ensure the sign is correct of damping
        print('Extrapolation')

        return float(force)
    # 
    else:
        print('Interpolation')

        # identifying the value that is higher than current speed
        for i,val in enumerate(axle.damp.speed):
            if val > speed_val:
                upper_val = val
                upper_ind = i
                lower_val = axle.damp.speed.iloc[i-1]
                break

        math_1 = (speed_val - axle.damp.speed.iloc[upper_ind-1])
        math_2 = (axle.damp.force.iloc[upper_ind] - axle.damp.force.iloc[upper_ind - 1])
        math_3 = axle.damp.speed.iloc[upper_ind] - axle.damp.speed.iloc[upper_ind-1]
        force = (math_1 * math_2)/math_3 + axle.damp.force.iloc[upper_ind-1]
        force = sign * force # this is in order to ensure the sign is correct of damping

        return float(force)


def quarter_car(t, q, axle): 
    q1,q2,q3,q4,b,r = q

    q1dot = q2
    q2dot = -(axle.k/axle.sprung)*(q1-q3) - (b/axle.sprung)
    q3dot = q4
    q4dot = (axle.k/axle.usprung)*(q1-q3) + (b/axle.usprung) - axle.kt*(q3-r) * 1/axle.usprung

    states = [q1dot,q2dot,q3dot,q4dot,b, r]
    # returns vector of Sprung Disp, Sprung Velocity, Unsprung Disp, Unsprung Velocity
    return states

# wrapper function for the ode to allow for intermediate solutions
def wrapper(t, q, axle):
    q1,q2,q3,q4,b,r = q
    curr_velocity = q2-q4
    b = lin_interp(axle, curr_velocity)
    r = ramp(1, 1.1, 0.0833333, t)
    print(f'Force is {b} lbf')
    q[4] = b
    q[5] = r
    states = quarter_car(t, q, axle)
    return states


if __name__ == "__main__":
    tv = np.linspace(0,5,5000)
    # sprung disp, sprung vel, unsprung disp, unsprung vel, damping force, input
    q = [0,0,0,0,0,0]
    car = vehicle_generator(file)

    # solving ode problem
    sol = scipy.integrate.solve_ivp(wrapper, [0,5], q, args = (car.front,), 
        t_eval = tv, first_step = 0.01, max_step = 0.1, rtol = 1e-3, atol = 1e-5)
    t = sol.t
    q = sol.y
    
    label = ['Sprung Disp', 'Sprung Velocity', 'Unsprung Disp', 'Unsprung Velocity']
    colors = ['k', 'b', 'r', 'g']
    fig,ax = plt.subplots()
    ax2 = ax.twinx()
    for i,val in enumerate(label):
        if i == 1 or i == 3:
            ax.plot(t, q[i] * 12, label = label[i], color = colors[i])
        else:
            ax2.plot(t, q[i] * 12, label = label[i], color = colors[i])
    fig.legend()
    ax.set_ylabel('Velocities [in/s]')
    ax2.set_ylabel('Displacement [in]')
    plt.show()
    plt.close()

    axle = car.front
    q1,q2,q3,q4,b,r = sol.y
    accel_sprung = []
    accel_usprung = []
    for i,val in enumerate(t):
        accel_sprung.append((-(axle.k/axle.sprung)*(q1[i]-q3[i]) - (b[i]/axle.sprung))/g)
        accel_usprung.append(((axle.k/axle.usprung)*(q1[i]-q3[i]) + (b[i]/axle.usprung) - axle.kt*(q3[i]-r[i]) * 1/axle.usprung)/g)

    fig,ax = plt.subplots()
    ax.plot(t, accel_sprung, label = 'Sprung Mass [g]', color = 'k')
    ax.plot(t, accel_usprung, label = 'Unprung Mass [g]', color = 'b')
    # ax2 = ax.twinx()
    # ax2.plot(t, r, label = 'Input', color = 'r', linestyle = '--')
    ax.set_ylabel('Acceleration [g]')
    # ax2.set_ylabel('Road Plot [in]')
    fig.legend()
    plt.show()
    plt.close()

    data_export = np.array([t, r, q1, q2, q3, q4, b, accel_sprung, accel_usprung])
    data_export = np.transpose(data_export)
    df = pd.DataFrame(data = data_export, columns = ['Time', 'Input', 'Sprung Displacement', 'Sprung Velocity', 'Unsprung Disp', "Unsprung Velocity", 'Damping Force', 'Accel Sprung', 'Accel Unsprung'])
    
    df.to_csv('output_ivp.csv', header = True, index = True)