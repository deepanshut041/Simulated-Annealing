from random import random
from sa_helper import *

def anneal():
    sol = (12288, 30, 30, 30, 1)
    old_cost = cost(sol)
    T = 1.0
    T_min = 0.00001
    alpha = 0.9
    lowest_cost = old_cost
    best_sol = sol
    while T > T_min:
        i = 1
        while i <= 10:
            new_sol = neighbor_3(sol)
            new_cost = cost(new_sol)
            ap = acceptance_probability(old_cost, new_cost, T)
            if ap > random():
                sol = new_sol
                old_cost = new_cost
            i += 1
            if lowest_cost > new_cost:
                lowest_cost = new_cost
                best_sol = new_sol
        T = T*alpha
    return best_sol, lowest_cost

best_sol, lowest_cost = anneal()

print("Best Solution for project is :" ,best_sol)
print("Lowest Cost for solution is :" ,lowest_cost)