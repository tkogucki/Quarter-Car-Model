import numpy as np
import matplotlib.pyplot as plt
import scipy
import pandas as pd
import math
import cmath

# Equations pulled from Blevin's

g_c = 32.17 

kf = 350 * 2 # lbs/in
kr = 400 * 2 # lbs/in
m = 600 # lbs
J = 3.8e5  # lbm in^2
w_dist = .43
wb = 63 # inch

kf = kf * 12
kr = kr * 12
m = m/g_c
wb = wb / 12
J = J / 32 * 1/12**2
lf = (1-w_dist) * wb
lr = w_dist * wb

print(f'Simple frequency: {(1/(2 * math.pi)) * ((kf+kr)/m)**(1/2)}')

x_term = (kf + kr) / m
th_term = (kf * lf**2 + kr * lr**2) / J
pre_term = 1/(2**(3/2) * math.pi)
final_term = (4 * kf * kr * (lf + lr)**2) / (J*m)

# compiling equation left and right
left = x_term + th_term
right = ((x_term + th_term)**2 - final_term)**(0.5)
f1 = pre_term * (left - right)**(0.5)
f2 = pre_term * (left + right)**(0.5)

print(f1, f2)

# mode shapes
m1 = (kf + kr - m*(2 * math.pi * f1)**2)/(lf * kf - lr * kr) * 180/math.pi
m2 = (kf + kr - m*(2 * math.pi * f2)**2)/(lf * kf - lr * kr) * 180/math.pi
print(m1,m2)
