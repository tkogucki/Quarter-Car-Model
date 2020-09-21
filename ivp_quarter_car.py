import numpy as np
import matplotlib.pyplot as plt
import scipy
import pandas as pd
import math
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

# copied from wiki page that brian generated
# speed in mph, everything else in feet
def brian_ramp(vel, h, radius, t):
        vel = vel 
        l = math.sqrt(radius**2 - (radius - h)**2) - vel * t
        if l > 0:
            x_i = h + math.sqrt(radius**2 - l**2) - radius
            return x_i
        else:
            return h

    

# linearly interpolates within damper lookup table
def lin_interp(axle, speed):

    # print(f'The current speed is {speed}')
    length = axle.damp.length
    if speed == 0:
        return 0
    speed_val = abs(speed)
    sign = speed/speed_val
    # if (speed_val > 1.2 * axle.damp.speed.iloc[length-1]):
    #     delta_speed = speed_val - axle.damp.speed.iloc[length - 1]
    #     force = axle.damp.max_damp
    #     force = sign * force
    #     print('1.2 Speed Limit Hit')
    #     return float(force)
    # linearly extrapolates last damping value
    if (axle.damp.speed.iloc[length-1] <= speed_val):
        delta_speed = speed_val - axle.damp.speed.iloc[length - 1]
        force = delta_speed * axle.damp.slope + axle.damp.force.iloc[length - 1]
        force = sign * force # this is in order to ensure the sign is correct of damping
        # print('Extrapolation')
        return float(force)
    # linearly interpolates values
    else:
        # print('Interpolation')

        # identifying the value that is higher than current speed
        for i,val in enumerate(axle.damp.speed):
            if val >= speed_val:
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
    q1,q2,q3,q4,b,r,spring_force,time_force = q
    spring_force = (axle.k) * (q1 - q3)
    tire_force = axle.kt*(q3 - r)

    q1dot = q2
    q2dot = -(axle.k/axle.sprung)*(q1 - q3) - (b/axle.sprung)
    q3dot = q4
    q4dot = (axle.k/axle.usprung)*(q1 - q3) + ((b)/axle.usprung) - axle.kt*(q3 - r) * 1/axle.usprung

    
    states = [q1dot,q2dot,q3dot,q4dot,b, r, spring_force, tire_force]
    # returns vector of Sprung Velocity, Sprung Acceleration, Unsprung Velocity, Unsprung Acceleration
    return states

# wrapper function for the ode to allow for intermediate solutions
def wrapper(t, q, axle, sim):
    q1,q2,q3,q4,b,r,spring_force,tire_force = q

    curr_velocity = (q2 - q4)
    b = lin_interp(axle, curr_velocity)
    # r = ramp(1, 1.1, 0.083, t)
    r = brian_ramp(sim.v_speed, sim.step_height, sim.tire_radius, t)

    q[4] = b
    q[5] = r
    states = quarter_car(t, q, axle)
    return states


# iterates through and provides the outputed values as seen in the diff eq
def output_generator(t, q, axle):
    output_array = []
    for i,val in enumerate(t):
        curr_output = wrapper(val, q[:,i], axle, sim)
        output_array.append(curr_output)
    # converting to numpy array
    output_array = np.array(output_array)

    # converting units for values
    
    for i in output_array:
        # print(i)
        # converting velocities to in/s and accelerations to g's
        i[0] = i[0] * 12
        i[1] = i[1] / g
        i[2] = i[2] * 12
        i[3] = i[3] / g
        # converting road input to inches
        i[5] = i[5] * 12

    # transposing so that structure is (t, returned value)
    output_array = np.transpose(output_array)
    # constructing original q matrix
    q_output = [q[0,:] * 12, q[1,:] * 12, q[2,:] * 12, q[3,:] * 12]


    return output_array, q_output

if __name__ == "__main__":
    tv = np.linspace(0,5,100000)
    # sprung disp, sprung vel, unsprung disp, unsprung vel, damping force, input, spring force, tire force
    q = [0,0,0,0,0,0,0,0]
    print('Simulation is currently running...')

    car, sim = vehicle_generator(file)

    axle = car.front

    # solving ode problem
    sol = scipy.integrate.solve_ivp(wrapper, [0,5], q, args = (axle, sim), 
        t_eval = tv, rtol = 1e-3, atol = 1e-5, max_step = 0.001)
    t = sol.t
    q = sol.y
    output,q = output_generator(t, sol.y, axle)
    q1dot,accel_sprung,q3dot,accel_usprung,b,r,spring_force,tire_force = output
    q1, q2, q3, q4 = q
    
    label = ['Sprung Disp', 'Sprung Velocity', 'Unsprung Disp', 'Unsprung Velocity']
    colors = ['k', 'b', 'r', 'g']
    fig,ax = plt.subplots()
    ax2 = ax.twinx()
    for i,val in enumerate(label):
        if i == 1 or i == 3:
            ax.plot(t, q[i], label = label[i], color = colors[i])
        else:
            ax2.plot(t, q[i], label = label[i], color = colors[i])
    ax.grid(alpha = 0.7)
    ax2.plot(t, r, label = 'Input')
    fig.legend()
    ax.set_ylabel('Velocities [in/s]')
    ax2.set_ylabel('Displacement [in]')


    

    
    fig2,ax = plt.subplots()
    ax.plot(t, accel_sprung, label = 'Sprung Mass [g]', color = 'k')
    ax.plot(t, accel_usprung, label = 'Unprung Mass [g]', color = 'b')
    ax.set_ylabel('Acceleration [g]')
    ax2 = ax.twinx()
    ax2.plot(t, spring_force, label = 'Spring Force [lbf]', color = 'r')
    ax2.plot(t, tire_force, label = 'Tire Force [lbf]', color = 'g')
    ax2.plot(t,b, label = 'Damper Force [lbf]', color = 'magenta')
    ax2.set_ylabel('Force')
    ax.grid(alpha = 0.7)
    fig2.legend()
    plt.show()
    plt.close()



    data_export = np.array([t, r, q1, q2, q3, q4, b, accel_sprung, accel_usprung, spring_force,tire_force])
    data_export = np.transpose(data_export)
    df = pd.DataFrame(data = data_export, columns = ['Time [s]', 'Input [in]', 'Sprung Displacement [in]', 'Sprung Velocity [in/s]', 'Unsprung Disp [in]', 
                                                        "Unsprung Velocity [in/s]", 'Damping Force [lbf]', 'Accel Sprung [g]', 'Accel Unsprung [g]', 'Spring Force [lbf]', 'Tire Force [lbf]'])
    
    # df.to_csv('output_300kt_ramp.csv', header = True, index = True)