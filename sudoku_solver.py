import numpy as np
grid=np.zeros([9,9])
import sys
def number_checker(row,column,n):
    global grid
    #number to be checked is n, row and column are it's position starting from 0
    if n in grid[row][:]: #checks row
        return False
    for i in range(9): #checks column
        if n == grid[i][column]:
            return False
            break
    for i in range(3):
        for j in range(3):
            if n == grid[3*(row//3) + i][3*(column//3) + j]:
                return False
                break

    return True

def find_empty():
    for i in range(9):
        for j in range(9):
            if grid[i][j]==0:
                return (i,j)
    return None

def solver_actual():
    global grid
    indexs = find_empty()
    if not indexs:
        return True
    
    i,j=indexs
    for k in range(1,10):
        if number_checker(i,j,k):
            grid[i][j]=k
            if solver_actual():
                return True
            grid[i][j]=0

    return False

def solve(grid_to_solve):
    global grid
    grid= grid_to_solve.copy()
    if (solver_actual()):
        return grid
    else:
        return np.zeros([9,9])

#%% 


# %%
