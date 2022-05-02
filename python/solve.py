"""Solves an instance.

Modify this file to implement your own solvers.

For usage, run `python3 solve.py --help`.
"""

import argparse
from cmath import inf
from curses.ascii import NUL
import math
from pathlib import Path
from turtle import pen
from typing import Callable, Dict, List
from xml.etree.ElementPath import find
import pulp 
import math
import numpy as np
import re

from instance import Instance
from point import Point
from solution import Solution
from file_wrappers import StdinFileWrapper, StdoutFileWrapper

global temp_towers

def solve_d(instance: Instance) -> Solution:
    all_tower_index = range(((instance.grid_side_length) ** 2))
    corr = []
    for i in all_tower_index:
        tower = num_to_corr(i, instance.grid_side_length)
        covers = [cover(tower, city, instance) for city in instance.cities]
        corr.append(covers)
    # initialize lp problem
    lp = pulp.LpProblem('PPP', pulp.LpMinimize)

    # Create problem Variable
    towers = []
    for i in all_tower_index:
        towers.append(pulp.LpVariable(name="tower" + str(i), lowBound=0, upBound=1, cat="Integer"))

    # Objective Func
    lp += pulp.lpSum([towers[i] for i in all_tower_index])

    # Constraints:
    for j in np.array(corr).T:
        lp += pulp.lpSum([towers[i] * j[i] for i in all_tower_index]) >= 1
    
    #solver:
    status = lp.solve()

    chosen_towers = []
    for i in lp.variables():
        if i.varValue == 1:
            tower_index = int(i.name[5:]) 
            chosen_towers.append(num_to_corr(tower_index, instance.grid_side_length))
   
    print(chosen_towers)
    return Solution(
        instance = instance,
        towers = chosen_towers
    )




# helper functions
"""
def chosen(index):
    chosen_towers = []
    for i in index:
        if towers[i] == 1:
            chosen_towers.append(num_to_corr(i, dimension))
    return chosen_towers
"""

def num_to_corr(num, dimension):
    y = num // (dimension)
    x = num % (dimension)
    return Point(x, y)

def cover(tower, city, instance:Instance):
    if distance(tower, city) <= instance.coverage_radius:
        return 1
    return 0

def penalty(chosen_towers, instance: Instance):
    penalty_towers = [0] * len(chosen_towers)
    for i in range(len(chosen_towers)):
        j = i
        while j < len(chosen_towers):
            if 0 < distance(chosen_towers[i], chosen_towers[j]) <= instance.penalty_radius:
                penalty_towers[i] += 1
                penalty_towers[j] += 1
            j += 1
    return penalty_towers

def distance(i, j):
    return Point.distance_obj(i, j)




def solve_naive(instance: Instance) -> Solution:
    list_of_possible_tower_location = list_of_possible_tower(instance)
    #print( "length", len(list_of_possible_tower_location))
    solution = Solution(instance = instance, towers = [])
    result = find_min_penalty(instance, instance.cities, list(), list_of_possible_tower_location, 0)
    #print(result)

    return Solution(
        instance=instance,
        towers=temp_towers,
    )

def find_min_penalty(instance: Instance, unvisited_cities: List[Point], fixed_towers: List[Point], unexplored_locs: List[Point], penalty):
    global temp_towers 
    temp_towers = fixed_towers
    #print("unvisited", len(unvisited_cities), "num tower", len(fixed_towers))
    if len(unvisited_cities) <= 0:
        return 0
    if len(unexplored_locs) <= 0:
        #print("unvisited", len(unvisited_cities))
        return math.inf
    if len(unexplored_locs) > 0:
        #print("cover", len(unvisited_cities), "tower", len(temp_towers), "return", min(find_min_penalty(instance, covered_cities(instance, unvisited_cities, unexplored_locs[0]), fixed_towers + [(unexplored_locs[0])], unexplored_locs[1:], 0) + penalty_calculator(fixed_towers, penalty, unexplored_locs[0],instance), 
        find_min_penalty(instance, unvisited_cities, fixed_towers, unexplored_locs[1:], 0)
        return min(find_min_penalty(instance, covered_cities(instance, unvisited_cities, unexplored_locs[0]), fixed_towers + [(unexplored_locs[0])], unexplored_locs[1:], 0) + penalty_calculator(fixed_towers, penalty, unexplored_locs[0],instance), 
        find_min_penalty(instance, unvisited_cities, fixed_towers, unexplored_locs[1:], 0))




def covered_cities(instance: Instance, unvisited_cities: List[Point], new_tower):
    new_cities = unvisited_cities
    for i in list(range(len(new_cities)-1, -1, -1)):
        if (Point.distance_obj(new_cities[i], new_tower) <= instance.coverage_radius):
            new_cities.pop(i)
    return new_cities


def penalty_calculator(towers: List[Point], penalty, new_pt:Point, instance: Instance):
    penalty_new = 0
    if len(towers) == 0:
        penalty_new += 170 
    for fidx, first in enumerate(towers):
        num_overlaps = 0
        if Point.distance_obj(first, new_pt) <= instance.penalty_radius:
                num_overlaps += 1
        penalty_new += 170 * math.exp(0.17 * num_overlaps)
    return penalty_new
# return list of possible towers in the grid
def list_of_possible_tower(instance: Instance):
    possible_tower = []
    for i in range(instance.grid_side_length):
        for j in range(instance.grid_side_length):
            p = Point(i, j)
            if (possible_location(instance, p)):
                possible_tower.append(p)
    return possible_tower

# ** check if the current location is a possible location for a new tower, returns a boolean value
def possible_location(instance: Instance, curr: Point) :
    for i in range(len(instance.cities)):
        if (((instance.cities[i].x - curr.x)**2 + (instance.cities[i].y - curr.y)**2) <= instance.coverage_radius ** 2):
            return True
    return False

SOLVERS: Dict[str, Callable[[Instance], Solution]] = {
    "naive": solve_naive, 
    "d": solve_d
}


# You shouldn't need to modify anything below this line.
def infile(args):
    if args.input == "-":
        return StdinFileWrapper()

    return Path(args.input).open("r")


def outfile(args):
    if args.output == "-":
        return StdoutFileWrapper()

    return Path(args.output).open("w")


def main(args):
    with infile(args) as f:
        instance = Instance.parse(f.readlines())
        solver = SOLVERS[args.solver]
        solution = solver(instance)
        assert solution.valid()
        with outfile(args) as g:
            print("# Penalty: ", solution.penalty(), file=g)
            solution.serialize(g)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Solve a problem instance.")
    parser.add_argument("input", type=str, help="The input instance file to "
                        "read an instance from. Use - for stdin.")
    parser.add_argument("--solver", required=True, type=str,
                        help="The solver type.", choices=SOLVERS.keys())
    parser.add_argument("output", type=str,
                        help="The output file. Use - for stdout.",
                        default="-")
    main(parser.parse_args())
