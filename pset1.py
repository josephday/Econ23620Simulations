# Joe Day
# ECON 23620 Inequality: A Macroeconomic Perspective
# Problem Set 1

import numpy as np
import math
from scipy import interpolate
import matplotlib.pyplot as plt

## parameters

gamma = 2 # risk aversion
beta = 0.97 # time value decay
r = 0.03 # interest rate
R = 1 + r
psi = 0.1 # disutility of labor
T = 50 # finite time horizon
b = 0.5 # unemployment benefit
w = np.append(np.linspace(1,2.5,25),  np.linspace(2.5,0.1,25)) # incomes




# set up asset grids
na = 1000
amax = 10
amin = 0 # borrowing constraint

## utility function



def u(c):
  if gamma == 1:
    return math.log(c)
  else:
    return float(c**(1-gamma) - 1)/(1-gamma)
    
u = np.vectorize(u)

## set up grids

# assets
agrid = np.linspace(0,1,na).transpose()
agrid = amin + float(amax-amin)*agrid
# ensures zero in asset grid
agrid[min(range(len(agrid)), key=lambda i: abs(agrid[i]))]=0 

# decisions
V = np.zeros((na,T))
con = np.zeros((na,T))
sav = np.zeros((na,T))
savind = np.zeros((na,T))
works = np.zeros((na,T))

# at t = T
savind[:,T-1] = np.where(agrid==0)
sav[:,T-1] = 0
works[:,T-1] = 0
con[:,T-1] = float(R)*agrid + w[T-1]*works[:,T-1] + b*(1-works[:,T-1]) - sav[:,T-1]
V[:,T-1] = u(con[:,T-1])



## Solve for Value Fn Backwards


solve = True
simulate = True
plot = True

if solve:
  for it in range(T-2,-1,-1):
    print('Solving value function at age {}'.format(it))
    for ia in range(0, na):
      cash_working = float(R) * agrid[ia] + w[it]
      c_working = np.maximum((cash_working-agrid), 1*10**-10)

      cash_not_working = float(R) * agrid[ia] + b
      c_not_working = np.maximum((cash_not_working-agrid), 1*10**-10)
      
      Vchoice_working = u(c_working) + beta * V[:,it+1] - psi 
      Vchoice_not_working = u(c_not_working) + beta * V[:,it+1] 

      working_max = np.max(Vchoice_working)
      not_working_max = np.max(Vchoice_not_working)
      if working_max > not_working_max:
        V[ia,it] = working_max
        savind[ia,it]  = np.argmax(Vchoice_working)
        sav[ia,it] = agrid[np.int_(savind[ia,it])]
        con[ia,it] = cash_working - sav[ia,it]
        works[ia,it] = 1
        print('ya better work bitch')
      else:   
        V[ia,it] = not_working_max
        savind[ia,it]  = np.argmax(Vchoice_not_working)
        sav[ia,it] = agrid[np.int_(savind[ia,it])]
        con[ia,it] = cash_not_working - sav[ia,it]
        works[ia,it] = 0
        print('you dont gotta go to work work work work')

print('sup')

## Simulate
if simulate:
  a_in_sim = np.zeros(T+1)
  a_initial = 0
  inter = interpolate.interp1d(agrid,range(0,na),'nearest')
  a_in_sim[0] = inter(a_initial)


  # create array of income after choice of labor
  true_income = w

  for it in range(0,T):
    print(' Simulating time period {}'.format((it+1)))
    a_in_sim[it+1] = savind[np.int_(a_in_sim[it]), it]
    asim = agrid[np.int_(a_in_sim)] 
    if it==25:
      print(asim)
      
    csim = R*asim[0:T] + true_income - asim[1:(T+1)]

if plot:
  fig = plt.figure(1)
  plt.subplot(1,2,1)
  plt.plot(range(1,51), w, 'k-', lw=1)
  plt.plot(range(1,51), csim, 'r--', lw=1)
  plt.grid
  plt.title('Income and Consumption')
  plt.legend('Income', 'Consumption')

  plt.subplot(1,2,2)
  plt.plot(range(0,51), asim, 'b-', lw=1)
  plt.plot(agrid,np.zeros(na), 'k-', lw=0.5)

