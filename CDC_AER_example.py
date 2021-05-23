###################################################
# Examples for "Combinatorial Discrete Choice" 
# by Costas Arkolakis and Fabian Eckert
# Code written by Jack Liang
# Last updated 10 July 2018
# Corresponding Author: fabian.eckert@yale.edu
###################################################


###################################################
# Importing packages
###################################################
import pandas as pd
import os
import numpy as np

###################################################
# Importing CDC_AER
###################################################
os.chdir(r'C:\Users\fabri\Documents\GitHub\cdc_solver')   
from CDC_AER import AE
from CDC_AER import AER
from CDC_AER import brute_force

###################################################
# A profit function as in Jia 2008
###################################################
def jia(val, params):
    '''
    val: a {0,1}^n vector
    params: Z (R^(n^2)), X (R^n), delta (R)
    
    Evaluates the function: 
    Profit = sum_i 1(X_i = 1) [ X_i + delta *  sum_j (1{i != j} 1/Z(i,j)) ] 
        where 1(...) is the indicator function
    '''
    
    dim = len(val)
    Z = params[0] 
    X = params[1] 
    delta = params[2]
    
    Imat = np.tile(val.values, (dim, 1))
    to_add = np.sum(np.divide(Imat, Z), axis = 1)
    to_add = np.subtract(to_add, np.diagonal(np.divide(Imat, Z)))
    S = np.add(X , delta * to_add)

    S1 = np.multiply(S, val)
    
    return sum(S1)

###################################################
# Examples
###################################################
k = 4
Z = [[1] * k, [1] * k, [1] * k, [1] * k]
X1 = [5,6,7,8]
#X2 = [1,5,10,20]
delta = -2

init = pd.Series([0.5]*k)


# Example 1, AE is not sufficient
print('Here, AE is not sufficient, but AER is')
print('AE: ')
print(AE(init, jia, True, [Z, X1, delta]))
print('')

# Example 2, AER on the previous
print('AER: ')
print(AER(init, jia, True, [Z, X1, delta]))
print('') 

# Brute force
print('Verifying: ')
print(brute_force(init, jia, [Z, X1, delta]))

print('')




