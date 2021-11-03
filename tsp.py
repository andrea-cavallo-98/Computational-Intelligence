# Copyright © 2021 Giovanni Squillero <squillero@polito.it>
# Free for personal or classroom use; see 'LICENCE.md' for details.
# https://github.com/squillero/computational-intelligence

import logging
from math import sqrt
from typing import Any
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

NUM_CITIES = 9
STEADY_STATE = 100
GENOME_LENGTH = NUM_CITIES
POPULATION_SIZE = 20
OFFSPRING_SIZE = 50
TOURNAMEN_SIZE = 2
MUTATION_PROBABILITY = 1 / GENOME_LENGTH

def parent_selection(problem, population):
    tournament = population[np.random.randint(0, len(population), size=(TOURNAMEN_SIZE,))]
    fitness = np.array([problem.evaluate_solution(t) for t in tournament])
    return np.copy(tournament[fitness.argmin()])

def xover(parent1, parent2): 
    offspring = np.zeros(parent1.shape)
    xover_type = np.random.randint(3)
    if xover_type == 0: # cycle crossover
        i = np.random.randint(0, GENOME_LENGTH)
        j = np.random.randint(0, GENOME_LENGTH + 1)
        while j <= i:
            j = np.random.randint(0, GENOME_LENGTH + 1)
        offspring[i:j] = parent2[i:j]
        c = 0
        for n in parent1:
            if c == i:
                c = j
            if n not in offspring:
                offspring[c] = n
                c += 1
    elif xover_type == 1: # partially mapped crossover
        pass
    elif xover_type == 2: # inver over
        i = np.random.randint(0, GENOME_LENGTH)
        offspring = parent1.copy()
        c = np.where(parent2 == parent1[i])[0][0]
        j = np.where(parent1 == parent2[(c+1) % GENOME_LENGTH])[0][0]
        offspring[(i+1) % GENOME_LENGTH] = parent2[(c+1) % GENOME_LENGTH]
        if j > i:
            offspring[i+2:j+1] = parent1[j-1:i:-1]
        else:
            offspring[j+2:i+1] = parent1[i-1:j:-1]
    return offspring


def mutate(parent):
    offspring = np.copy(parent)
    while np.random.random() < MUTATION_PROBABILITY:
        mutation = np.random.randint(4)
        if mutation == 0:
            if np.random.random() < 0.5: #swap mutation
                i = np.random.randint(0, GENOME_LENGTH)
                j = np.random.randint(0, GENOME_LENGTH)
                while j == i:
                    j = np.random.randint(0, GENOME_LENGTH)
                offspring[i], offspring[j] = parent[j], parent[i]
        elif mutation == 1: #inversion mutation
            i = np.random.randint(1, GENOME_LENGTH-1)
            j = np.random.randint(0, GENOME_LENGTH)
            while j <= i:
                j = np.random.randint(0, GENOME_LENGTH)
            offspring[i:j] = parent[j-1:i-1:-1]
        elif mutation == 2: #scramble mutation
            i = np.random.randint(1, GENOME_LENGTH-1)
            j = np.random.randint(0, GENOME_LENGTH)
            while j <= i:
                j = np.random.randint(0, GENOME_LENGTH)
            np.random.shuffle(offspring[i:j])        
        elif mutation == 3: #insert mutation
            i = np.random.randint(1, GENOME_LENGTH-1)
            j = np.random.randint(0, GENOME_LENGTH)
            while j <= i:
                j = np.random.randint(0, GENOME_LENGTH)
            offspring[i+1] = parent[j]
            offspring[i+2:j+1] = parent[i+1:j]
    return offspring



class Tsp:

    def __init__(self, num_cities: int, seed: Any = None) -> None:
        if seed is None:
            seed = num_cities
        self._num_cities = num_cities
        self._graph = nx.DiGraph()
        np.random.seed(seed)
        for c in range(num_cities):
            self._graph.add_node(c, pos=(np.random.random(), np.random.random()))

    def distance(self, n1, n2) -> int:
        pos1 = self._graph.nodes[n1]['pos']
        pos2 = self._graph.nodes[n2]['pos']
        return round(1_000_000 / self._num_cities * sqrt((pos1[0] - pos2[0])**2 +
                                                         (pos1[1] - pos2[1])**2))

    def evaluate_solution(self, solution: np.array) -> float:
        total_cost = 0
        tmp = solution.tolist() + [solution[0]]
        for n1, n2 in (tmp[i:i + 2] for i in range(len(tmp) - 1)):
            total_cost += self.distance(n1, n2)
        return total_cost

    def plot(self, path: np.array = None) -> None:
        if path is not None:
            self._graph.remove_edges_from(list(self._graph.edges))
            tmp = path.tolist() + [path[0]]
            for n1, n2 in (tmp[i:i + 2] for i in range(len(tmp) - 1)):
                self._graph.add_edge(n1, n2)
        plt.figure(figsize=(12, 5))
        nx.draw(self._graph,
                pos=nx.get_node_attributes(self._graph, 'pos'),
                with_labels=True,
                node_color='pink')
        if path is not None:
            print(f"Current path: {self.evaluate_solution(path):,}")
        plt.show()

    @property
    def graph(self) -> nx.digraph:
        return self._graph


def tweak(solution: np.array, *, pm: float = .1) -> np.array:
    new_solution = solution.copy()
    p = None
    while p is None or p < pm:
        i1 = np.random.randint(0, solution.shape[0])
        i2 = np.random.randint(0, solution.shape[0])
        temp = new_solution[i1]
        new_solution[i1] = new_solution[i2]
        new_solution[i2] = temp
        p = np.random.random()
    return new_solution


def main():

    problem = Tsp(NUM_CITIES)
    # initial random solution
    solution = np.array(range(NUM_CITIES))
    np.random.shuffle(solution)
    problem.plot(solution)
    best_fitness = problem.evaluate_solution(solution)

    # initial population
    population = np.array(np.zeros((POPULATION_SIZE, GENOME_LENGTH)))
    population[0, :] = solution
    for i in range(1, POPULATION_SIZE):
        population[i, :] = np.array(range(NUM_CITIES))
        np.random.shuffle(population[i,:])

    generations = 1
    steady_state = 0
    while steady_state < STEADY_STATE:
        generations += 1
        offspring = list()
        for _ in range(OFFSPRING_SIZE):
            p1, p2 = parent_selection(problem, population), parent_selection(problem, population)
            offspring.append(mutate(xover(p1, p2)))
        offspring = np.array(offspring)
        fitness = np.array([problem.evaluate_solution(o) for o in offspring])
        population = np.copy(offspring[fitness.argsort()[:]][:POPULATION_SIZE])
        new_best_fitness = problem.evaluate_solution(population[0])
        if best_fitness == new_best_fitness:
            steady_state += 1
        else:
            steady_state = 0
        best_fitness = new_best_fitness

    problem.plot(offspring[fitness.argmin()])
    print(f"Problem solved in {generations} generations")    

if __name__ == '__main__':
    logging.basicConfig(format='[%(asctime)s] %(levelname)s: %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().setLevel(level=logging.INFO)
    #main()
    p1 = np.array([1,2,3,4,5,6,7,8,9])
    p2 = np.array([4,6,1,8,2,5,3,9,7])
    print(xover(p1, p2))
