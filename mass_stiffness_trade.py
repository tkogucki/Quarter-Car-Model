import numpy as np
import matplotlib.pyplot as plt
import scipy
import pandas as pd
import math
import cmath

g_c = 32.17 

def make_3d_axes(x,y):
    x_plt = []
    y_plt = []
    for i in x:
        for j in y:
            x_plt.append(i)
            y_plt.append(j)
    return x_plt, y_plt

def calc_2dof(k1, k2, m1, m2):
    k1 = k1 * 12 #lb/ft
    k2 = k2 * 12 #lb/ft
    m1 = m1/g_c # slug
    m2 = m2/g_c # slug

    pre_add = (m2*(k1 + k2) + m1*k2)/(2*m1*m2)
    term_1 = (pre_add)**2
    term_2 = -(k1 * k2)/(m1 * m2)
    paran_1 = (term_2 + term_1)**(1/2)
    lambda_1 = (pre_add + paran_1)**(1/2)
    lambda_2 = (pre_add - paran_1)**(1/2)
    # print(f'Lambda 1 is {lambda_1}')
    # print(f'Lambda 2 is {lambda_2}\n')
    lambda_1 = complex(0,lambda_1)
    lambda_2 = complex(0, -lambda_2)
    w1 = abs(lambda_1.imag) * 1/(math.pi*2)
    w2 = abs(lambda_2.imag) * 1/(math.pi*2)
    # print(f'Omega 1 is {w1} rad/sec')

    # print(f'Omega 2 is {w2} rad/sec \n')

    print(f'Omega 1 is {w1} Hz')

    print(f'Omega 2 is {w2} Hz\n')



    x2_1 = ((m1 * lambda_1**2 + (k1+k2))/k2)
    x2_2 = ((m1 * lambda_2**2 + (k1+k2))/k2)
    print(x2_1, x2_2)

    return(w1, w2, lambda_1, lambda_2, [1, x2_1], [1, x2_2])


if __name__ == '__main__':
    # k1 = 560 #lb/in
    # k2 = 400 #lb/in
    m1 = 20 #lb
    m2 = 130 #lb
    # m1 = np.linspace((0.75 * 20), (1.25 * 20) ,50)
    # m2 = np.linspace((0.75 * 130), (1.25 * 130),50)
    k1 = np.linspace(0.75 * 560 ,1.25 * 560, 75)
    k2 = np.linspace(0.75 * 400, 1.25 * 400, 75)
    w1_l = []
    w2_l = []
    for val,i in enumerate(k1):
        for j in k2:
            w1, w2, lambda_1, lambda_2, m_1, m_2 = calc_2dof(i, j, m1, m2)
            w1_l.append(w1)
            w2_l.append(w2)

    m1_plt,m2_plt = make_3d_axes(k1,k2)
    fig = plt.figure(figsize = (8,6))
    ax = plt.axes(projection = '3d')
    ax.scatter(m1_plt,m2_plt,w2_l, s = 5)
    ax.set_xlabel('Tire Spring Rate [lbs/in]')
    ax.set_ylabel('Car Spring Rate [lbs/in]')
    ax.set_zlabel('1st Natural Frequency [Hz]')
    plt.show()

    fig2 = plt.figure(figsize = (8,6))
    ax2 = plt.axes(projection = '3d')
    ax2.scatter(m1_plt,m2_plt,w1_l, color = 'r', s = 5)
    ax2.set_xlabel('Tire Spring Rate [lbs/in]')
    ax2.set_ylabel('Car Spring Rate [lbs/in]')
    ax2.set_zlabel('2nd Natural Frequency [Hz]')
    plt.show()


