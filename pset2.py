# Joe Day
# ECON 23620 Inequality: A Macroeconomic Perspective
# Problem Set 2

import numpy as np
import math
from scipy import interpolate
from scipy import optimize
import matplotlib.pyplot as plt

## parameters

gamma = 2 # risk aversion
beta = 0.9724 # time value decay
r = 0.02 # interest rate
R = 1 + r
tau = 0.2
T = 0.2

P = [[0.9037, 0.0927, 0.0035, 0.0001, 0.0000],
    [0.0232, 0.9054, 0.0696, 0.0018, 0.0000],
    [0.0006, 0.0464, 0.9060, 0.0464, 0.0006],
    [0.0000, 0.0018, 0.0696, 0.9054, 0.0232],
    [0.0000, 0.0001, 0.0035, 0.0927, 0.9037]]

y = [0.3595, 0.6796, 1.0000, 1.3202, 1.6405]


# set up asset grids
na = 1000
amax = 10
amin = 0 # borrowing constraint
borrow_lim = amin
# computation
max_iter = 1000
tol_iter = 10**(-6)
Nsim = 50000
Tsim = 500

Display = True
DoSimulate = True
MakePlots = True
solve=True

InterpCon = 0
InterpEMUC = 1

## helper functions

def calc_stationary_distribution(P):
  ten_thousand = np.linalg.matrix_power(P, 10000)
  ten_thousand_plus_one = np.linalg.matrix_power(P, 10001)
  assert np.allclose(ten_thousand, ten_thousand_plus_one),  "P may be periodic"
  return ten_thousand[0]

def balance_budget(P, y, tau):
  total= 0
  stationary_dist = calc_stationary_distribution(P)
  for i in range(len(y)):
    total += stationary_dist[i] * y[i]
  T  = total * tau
  return T

## utility function

def u(c):
  if c<= 0 :
    c = 0.000000001
  if gamma == 1:
    return math.log(c)
  else:
    return float(c**(1-gamma) - 1)/(1-gamma)
    
u = np.vectorize(u)

def u1(con):
  con = abs(con)
  if con == 0:
    return 0
  else:  
    return float(con)**(-1*gamma)
u1 = np.vectorize(u1)

def u1_inv(con):

  return np.power(-1*u(con),-1/float(gamma))
#u1_inv = np.vectorize(u1_inv)

# set up grids

# assets
agrid = np.linspace(0,1,na).transpose()
agrid = amin + float(amax-amin)*agrid
# ensures zero in asset grid
agrid[min(range(len(agrid)), key=lambda i: abs(agrid[i]))]=0 


## Initialize consumption function
conguess = np.zeros((na, len(y)))
for i in range(len(y)):
  conguess[:,i] = float(r)*agrid + y[i]
#print(conguess)

income_guess = 1.000


ass1 = np.zeros((na,len(y)))

# decisions
stationary_dist = calc_stationary_distribution(P)
con=conguess
emuc = u1(con)*stationary_dist
income = income_guess 

iter = 0
cdiff = 1000;

def fn_eeqn_c(a, conlast, agrid, cash, last_y):
  y_dist = P[y.index(last_y)]
  c = zeros(1,len(y))
  for iy in range(0,len(ny)):
    c[iy] = interpolate.interp1d(agrid, conlast[:,iy], a)
  return u1(cash-a) - float(beta)*R*(u1(c)*y_dist)


if solve:
  while (iter <= max_iter & cdiff>tol_iter): 
    iter += 1
    conlast = con
    sav = np.zeros((na,len(y)))
    income_last = income
    y_dist = P[y.index(income_last)]
    emuc=u1(conlast)*y_dist
    muc1 = float(beta)*R*emuc
    con1 = u1_inv(muc1)

    #loop over assets
    for iy in range(0,len(y)):
      ass1[:,iy] = (con1[:,iy]+agrid - y[iy])/float(R)
      for ia in range(0,na):
        if agrid[ia]<ass1[0,iy]:
          sav[ia,iy]=borrow_lim
        else:
          sav[ia,iy] = interpolate.interp1d(ass1[:,iy], agrid, 'nearest', bounds_error=False).__call__(agrid[ia])
      con[:,iy]= float(R)*agrid+y[iy] - sav[:,iy]

    cdiff=max(max(abs(con-conlast)))
    print('we are on {} iteration'.format(iter))


"""
      #loop over income
      for iy in range(0,len(y)):
        cash = float(R) * agrid[ia] + y[iy]
        if InterpCon==1:
          if fn_eeqn_c(borrow_lim, conlast, agrid, cash, 0)>=0: #reset y from 0 to last y
            sav[ia,iy] = borrow_lim
          else:
            sav[ia,iy] = optimize.root(fn_eeqn_c, 0.5*cash)

        elif InterpEMUC == 1:
          if u1(cash-borrow_lim) >= float(beta)*R*interpolate.interp1d(agrid, emuc[:,iy], 'nearest').__call__(cash):
            sav[ia,iy] = borrow_lim
          else:
            sav[ia,iy] = optimize.fsolve(u1(cash-x) - float(beta)*R*interpolate.interp1d(agrid, emuc, x) , 0.5*cash)
"""
"""

    print('Solving value function at age {}'.format(it+1))
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
       
      else:   
        V[ia,it] = not_working_max
        savind[ia,it]  = np.argmax(Vchoice_not_working)
        sav[ia,it] = agrid[np.int_(savind[ia,it])]
        con[ia,it] = cash_not_working - sav[ia,it]
        works[ia,it] = 0
      

## Simulate for a given initial wealth
if simulate:
  a_in_sim = np.zeros(T+1)
  a_initial = 0
  inter = interpolate.interp1d(agrid,range(0,na),'nearest')
  a_in_sim[0] = inter(a_initial)

  # create empty array of simulated consumption, employment
  csim=np.zeros(T)
  sim_works=np.zeros(T+1)

  for it in range(0,T):
    print(' Simulating time period {}'.format((it+1)))
    a_in_sim[it+1] = savind[np.int_(a_in_sim[it]), it]
    asim = agrid[np.int_(a_in_sim)] 
    sim_works[it] = works[int(a_in_sim[it]), it]  
    csim[it] = R*asim[it] + w[it] * sim_works[it] + b * (1-sim_works[it]) - asim[it+1]


## Plot results of simulation
if plot:
  fig = plt.figure(1)
  plt.subplot(1,3,1)
  plt.plot(range(1,51), w, 'k-', lw=1)
  plt.plot(range(1,51), csim, 'r--', lw=1)
  plt.grid
  plt.title('Income and Consumption')

  plt.subplot(1,3,2)
  plt.plot(range(0,51), sim_works, 'g-', lw=1)
  plt.title('Work or No Work')
  
  plt.subplot(1,3,3)
  plt.plot(range(0,51), asim, 'b-', lw=1)
  plt.title('Wealth / Assets')

"""


