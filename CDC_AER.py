###################################################
# Code for "Combinatorial Discrete Choice"
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
# Defining the marginal value
###################################################
def marg(val, i, func, params = [] ): 
    '''
    val: a vector of {0, 1}^n that we want to compute the marginal value of
    i: the coordinate with which we want to compute the marginal value
    func: the name of the function
    params: an optional tuple of parameters we feed into the function
    
    output: D_i(val)
    
    A note: due to the way pandas pointers work, the code is 
        slightly more complicated than it should be.
    '''
    
    # a variable to remember the original value of the coordinate
    storage = val[i]
         
    # setting the coordinate to 1
    val_plus = val.copy()
    val_plus.at[i, 1] = 1
    high = func(val_plus, params)
    
    # setting the coordinate to 0
    val_minus = val.copy()
    val_minus.at[i, 1] = 0
    low = func(val_minus, params)
    
    # resetting the value
    val.at[i, 1] = storage
    
    # output
    return high - low    
    
    
###################################################
# AE
###################################################
def AE(fixed, func, submodular, params = [], suppress_output = True):
    '''
    fixed: a vector of {0,0.5, 1}^n with which we will perform AE
    func: the name of the function
    submodular: a boolean indicating if the function is submodular (True)
        or supermodular (False)
    params: an optional tuple of parameters we feed into the function
    suppress_output: an optional boolean, True by default
        if False, the result from each step of AE will be displayed
        useful for evaluating efficiency of AE
        
    output: a vector of {0,0.5, 1}^n
    '''
       
    exit_condition = True
    output = fixed.copy()
    
    while exit_condition:
       
        ## creating some storage vectors
        output_prev = output.copy()
        sup = output_prev.copy()
        inf = output_prev.copy()
        
        ## finding all "uncertain" coordinates, i.e., those = 0.5
        free_indices = list(output_prev[output_prev == .5].index)
        if not suppress_output:
            print(len(free_indices))
        
        ## a vector where all uncertain coordinate are 1
        sup[list(free_indices)] = 1
       
        ## a vector where all uncertain coordinate are 0
        inf[list(free_indices)] = 0
        
        ## submodular case
        if submodular:
            ## checking D_i(f(sup)) and D_i(f(inf)) for each i
            for index in list(free_indices):
                upper = marg(sup, index, func, params)
                lower = marg(inf, index, func, params)
                
                ## fixing coordinates of the output
                if upper < 0 and lower < 0:
                    output[index] = 0
                    free_indices.remove(index)
                elif lower >= 0 and upper >= 0:
                    output[index] = 1
                    free_indices.remove(index)
        
        ## supermodular case
        else:
            ## checking D_i(f(sup)) and D_i(f(inf)) for each i
            for index in free_indices:
                upper = marg(sup, index, func, params)
                lower = marg(inf, index, func, params)
                
                ## fixing coordinates of the output
                if upper > 0 and lower > 0:
                    output[index] = 0
                    free_indices.remove(index)
                elif lower <= 0 and upper <= 0:
                    output[index] = 1
                    free_indices.remove(index)
       
       
        ## if we cannot make any more changes, exit
        if all(output == output_prev):
            exit_condition = False
            
    return output


###################################################
# AER
###################################################
def AER(fixed, func, submodular, params = [], suppress_output = True):
    '''
    fixed: a vector of {0,0.5, 1}^n with which we will perform AER
    func: the name of the function
    submodular: a boolean indicating if the function is submodular (True)
        or supermodular (False)
    params: an optional tuple of parameters we feed into the function
    suppress_output: an optional boolean, True by default
        if False, the result from each step of AER will be displayed
        useful for evaluating efficiency of AER
    
    output: a tuple with the first coordinate
        a vector of {0,1}^n (the optimum x) and 
        the second coordinate a scalar (f(x))
    '''
    
    ## recursive condition, if every coordinate is determined exit
    if len(fixed[fixed == .5]) == 0:
        return fixed, func(fixed, params)
    
    ## trying AE on the input
    out = AE(fixed, func, submodular, params, suppress_output)
    if len(out[out == .5]) == 0:
        return out, func(out, params)
    else:
        to_fix = out[out == .5].index[0]
    
    ## specifying the first coordinate that is not fixed, setting to 0/1
    fixed_0 = pd.Series(list(out.values))
    fixed_0[to_fix] = 0

    fixed_1 = pd.Series(list(out.values))
    fixed_1[to_fix] = 1

    ## recursively running AER on the new input
    out_0, val_0 = AER(fixed_0, func, submodular, params, suppress_output)
    out_1, val_1 = AER(fixed_1, func, submodular, params, suppress_output)
    
    ## returning the max
    if val_0 > val_1:
        return out_0, val_0
    else:
        return out_1, val_1
      

        
###################################################
# brute force (for maxima)
###################################################
def brute_force(fixed, func, params):
    '''
    fixed: a vector of {0,0.5, 1}^n with which we will compute the maximum
    func: the name of the function
    params: an optional tuple of parameters we feed into the function
    '''
    
    output = pd.Series(dtype = pd.StringDtype())
    prof = -1 * np.inf
    dim = len(fixed)
    
    ## iterating over all 2^dim(fixed) possibilities
    for i in range(0, 2 ** dim):
        vec = list(format(i, '#0'+ str(dim+2) + 'b')[2:])
        vec = pd.Series(vec)
        vec = vec.astype(int)
        
        temp = func(vec, params)
        if temp > prof:
            output = vec
            prof = temp
            
    ## returning optimum- note if there are multiple, the last is returned      
    return output, prof  

    
    

