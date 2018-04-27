from random import random
from sa_helper import *
import matplotlib.pyplot as plt


def anneal():
    sol = (12288, 30, 30, 30, 1)
    old_cost = cost(sol) # Let s = s0
    T = 1.0
    T_min = 0.00001
    alpha = 0.9
    costs = []
    while T > T_min: # T ← temperature(k ∕ kmax)
        i = 1
        while i <= 5: 
            new_sol = neighbor_3(sol) # Pick a random neighbour, snew ← neighbour(s)
            new_cost = cost(new_sol)
            ap = acceptance_probability(old_cost, new_cost, T)
            if ap > random():  # If P(E(s), E(snew), T) ≥ random(0, 1):
                sol = new_sol  # s ← snew
                old_cost = new_cost
            i += 1
            costs.append(new_cost)
        T = T*alpha
    return sol, old_cost, costs

best_sol, accuracy, costs = anneal()

print("Best Solution for project is :" ,best_sol)
print("Highest accuracy for solution is :" ,accuracy)
# plot the cost
plt.plot(costs)
plt.ylabel('cost')
plt.xlabel('Temprature')
plt.title("Simulated Annealing")
plt.show()
