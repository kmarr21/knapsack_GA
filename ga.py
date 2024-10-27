import random
import matplotlib.pyplot as plt
import numpy as np
from typing import List

# MODEL PARAMETERS / CONSTANTS:
# box parameters
BOXES = [
    (20, 6),  # Box 1: (weight, importance)
    (30, 5),  # Box 2
    (60, 8),  # Box 3
    (90, 7),  # Box 4
    (50, 6),   # Box 5
    (70, 9),  # Box 6
    (30, 4),  # Box 7
    (30, 5),  # Box 8
    (70, 4),  # Box 9
    (20, 9),  # Box 10
    (20, 2),  # Box 11
    (60, 1),  # Box 12
]
MAX_WEIGHT = 250 # max weight the pack can get to

# Fitness function calculator; fitness = summed importance of contents as long as weight is under 250, otherwise = 0 
def calculate_fitness(chromosome):
    total_weight = 0
    total_importance = 0
    
    for i, gene in enumerate(chromosome):
        if gene == 1:
            total_weight += BOXES[i][0]
            total_importance += BOXES[i][1]
    
    return total_importance if total_weight <= MAX_WEIGHT else 0

# Random chromosome generator for initial part of algorithm; generates a random binary string the length of the number of boxes
def generate_random_chromosome():
    return [random.randint(0, 1) for _ in range(len(BOXES))]

# Selection mechanism: culls bottom 50% and selects top 50% with BEST fitness scores
def select_population(population, fitness_scores):
    population_fitness = list(zip(population, fitness_scores)) # pairs of chromosome options and their fitness scores
    
    # sort in descending order of fitness
    sorted_population = sorted(population_fitness, key=lambda x: x[1], reverse=True)
    
    # Select top 50%
    population_size = len(population)
    selected = sorted_population[:population_size//2]
    
    # return the best chromosomes (without fitness scores)
    return [chromosome for chromosome, _ in selected]

# 2-point crossover function
def crossover(parent1, parent2):
    point1, point2 = sorted(random.sample(range(len(parent1)), 2)) # selects two points
    
    # swaps sections out of parents at those 2 points to create thet two new children
    child1 = parent1[:point1] + parent2[point1:point2] + parent1[point2:]
    child2 = parent2[:point1] + parent1[point1:point2] + parent2[point2:]
    
    return child1, child2

# Mutation operator: with mutation rate of 2%, performs random bit-flip mutation at a given spot in the chromosome
def mutate(chromosome, mutation_rate=0.02):
    mutated = list(chromosome)
    for i in range(len(mutated)):
        if random.random() < mutation_rate:
            mutated[i] = 1 - mutated[i]
    return mutated

# MAIN FUNCTION to run genetic algorithm
def genetic_algorithm(population_size=100, mutation_rate=0.02, max_generations=1000, generations_no_improve=30):
    if generations_no_improve is None:
        generations_no_improve = 30
    
    # Generate random starting population of designated size (in this model, pop size = 100)
    population = [generate_random_chromosome() for _ in range(population_size)]
    
    best_solution = None
    best_fitness = 0
    no_improve_count = 0
    
    # History tracking for plotting 
    history = {
        'avg_fitness': [],
        'best_fitness': [],
        'generation': []
        }
    # iterate through generations
    for generation in range(max_generations):
        fitness_scores = [calculate_fitness(chromosome) for chromosome in population]
        
        # tracking metrics
        avg_fitness = sum(fitness_scores) / len(fitness_scores)
        current_best_fitness = max(fitness_scores)
        history['avg_fitness'].append(avg_fitness)
        history['best_fitness'].append(current_best_fitness)
        history['generation'].append(generation)
        
        # update best solution as needed
        if current_best_fitness > best_fitness:
            best_fitness = current_best_fitness
            best_solution = population[fitness_scores.index(current_best_fitness)]
            no_improve_count = 0
        else:
            no_improve_count += 1
            
        # Check termination condition: if going for 30+ generations with no improvement, stop the model and return best solution at that point
        if no_improve_count >= generations_no_improve:
            print(f"Terminated after {generation} generations due to no improvement")
            break
            
        # selection and producing of next population
        population = select_population(population, fitness_scores)
        
        # generating new population off of old population
        new_population = []
        while len(new_population) < population_size:
            parent1, parent2 = random.sample(population, 2)
            child1, child2 = crossover(parent1, parent2)
            
            # mutate children with chance = mutation_rate (2% chance)
            if random.random() < mutation_rate: child1 = mutate(child1)
            if random.random() < mutation_rate: child2 = mutate(child2)
                
            new_population.extend([child1, child2])
        
        population = new_population[:population_size]
    
    # returns best solution and history (for plotting)
    return best_solution, history

# Print selected boxes and their weight/importance for final output
def print_solution(chromosome):
    total_weight = 0
    total_importance = 0
    selected_boxes = []
    
    for i, gene in enumerate(chromosome):
        if gene == 1:
            selected_boxes.append(i + 1)
            total_weight += BOXES[i][0]
            total_importance += BOXES[i][1]
    
    print(f"Selected boxes: {selected_boxes}")
    print(f"Total weight: {total_weight}")
    print(f"Total importance: {total_importance}")

# single-run fitness curve plotting function (for final output)
def plot_single_run(history):
    plt.figure(figsize=(10, 6))
    
    plt.plot(history['generation'], history['avg_fitness'], label='Average Fitness', color='blue')
    plt.plot(history['generation'], history['best_fitness'],label='Best Fitness', color='red')
    
    plt.title('Fitness Over Generations (Single Run)')
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.legend()
    plt.grid(True)
    plt.show()

# multi-run fitness curve plotting function (with std. dev. bands)
def plot_multiple_runs(all_histories):
    plt.figure(figsize=(12, 6))
    max_gen = max(max(h['generation']) for h in all_histories)
    all_best_fitness = []
    all_avg_fitness = []
    
    # make sure all histories aligned with same length . . . 
    for history in all_histories:
        best_fitness = history['best_fitness']
        avg_fitness = history['avg_fitness']
        # . . . and pad shorter histories (so plotting indexing works)
        if len(best_fitness) < max_gen + 1:
            best_fitness = best_fitness + [best_fitness[-1]] * (max_gen + 1 - len(best_fitness))
            avg_fitness = avg_fitness + [avg_fitness[-1]] * (max_gen + 1 - len(avg_fitness))
            
        all_best_fitness.append(best_fitness)
        all_avg_fitness.append(avg_fitness)
    
    all_best_fitness = np.array(all_best_fitness) # convert to numpy arrays
    all_avg_fitness = np.array(all_avg_fitness)
    
    # get means and std. dev.s
    best_mean = np.mean(all_best_fitness, axis=0)
    best_std = np.std(all_best_fitness, axis=0)
    avg_mean = np.mean(all_avg_fitness, axis=0)
    avg_std = np.std(all_avg_fitness, axis=0)
    
    generations = range(max_gen + 1)
    
    # plot in the std dev
    plt.plot(generations, best_mean, 'red', label='Best Fitness')
    plt.fill_between(generations, best_mean - best_std, best_mean + best_std, color='red', alpha=0.2)
    
    plt.plot(generations, avg_mean, 'blue', label='Average Fitness')
    plt.fill_between(generations, avg_mean - avg_std, avg_mean + avg_std, color='blue', alpha=0.2)
    
    plt.title(f'Fitness Over Generations ({len(all_histories)} Runs)')
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.legend()
    plt.grid(True)
    plt.show()

# Function to run GA multiple times (if selected by user) and keep track of best solution
def run_multiple_ga(num_runs=10):
    best_overall_solution = None
    best_overall_fitness = 0
    all_histories = []
    
    print(f"\nRunning genetic algorithm {num_runs} times...\n")
    for run in range(num_runs):
        print(f"Run {run + 1}/{num_runs}")
        solution, history = genetic_algorithm(
            population_size=100,
            mutation_rate=0.02,
            generations_no_improve=30
        )
        # calculate fitness of solution returned by this run
        fitness = calculate_fitness(solution)
        all_histories.append(history)
        
        # update best overall fitness if this run was better
        if fitness > best_overall_fitness:
            best_overall_fitness = fitness
            best_overall_solution = solution
            
    return best_overall_solution, all_histories

# function to print introduction for UI, explain algorithm, allow user interaction
def print_intro():
    print("\n" + "="*80)
    print(" "*30 + "KNAPSACK GENETIC ALGORITHM")
    print("="*80 + "\n")
    
    print("This program solves the knapsack problem using a genetic algorithm.")
    print("The goal is to pack a backpack with items to maximize importance while staying under weight limit.\n")
    print("Users will be given a choice of whether to run this algorithm for 1 or multiple (1-100) runs.\n")
    
    print("AVAILABLE BOXES:")
    print("-" * 50)
    print(f"{'Box #':6} {'Weight':8} {'Importance':10}")
    print("-" * 50)
    for i, (weight, importance) in enumerate(BOXES, 1):
        print(f"{i:<6} {weight:8} {importance:10}")
    print("-" * 50)
    print(f"\nBackpack Weight Limit: {MAX_WEIGHT}\n")
    
    print("ALGORITHM PARAMETERS:")
    print("-" * 50)
    print("Population Size: 100")
    print("→ Provides optimal balance between exploration and computational efficiency")
    
    print("\nMutation Rate: 0.02")
    print("→ Balanced exploration vs exploitation")
    
    print("\nGenerations Without Improvement: 30")
    print("→ Allows sufficient time for convergence")
    print("-" * 50 + "\n")

# function to help UI get user preference for number of runs (if multiple runs option selected)
def get_number_of_runs():
    while True:
        print("\nPlease select number of runs:")
        print("1. Standard (10 runs)")
        print("2. Custom number of runs")
        choice = input("Enter choice (1 or 2): ").strip()
        
        if choice == '1':
            return 10
        elif choice == '2':
            while True:
                try:
                    num_runs = int(input("Enter desired number of runs (1-100): "))
                    if 1 <= num_runs <= 100:
                        return num_runs
                    print("Please enter a number between 1 and 100.")
                except ValueError:
                    print("Please enter a valid number.")
        else:
            print("Invalid choice. Please enter 1 or 2.")

# get user prference for single or multiple runs
def get_run_type():
    while True:
        print("\nPlease select run type:")
        print("1. Single run")
        print("2. Multiple runs")
        choice = input("Enter choice (1 or 2): ").strip()
        
        if choice in ['1', '2']:
            return int(choice)
        print("Invalid choice. Please enter 1 or 2.")

# get user confirmation that they are fine to proceed with running algorithm
def get_user_confirmation():
    while True:
        response = input("Would you like to run the algorithm? (y/n): ").lower()
        if response in ['y', 'n']:
            return response == 'y'
        print("Please enter 'y' for yes or 'n' for no.")

# main function to run genetic algorithm, facilitate user interfacing, etc.
if __name__ == "__main__":
    print_intro()
    
    if get_user_confirmation():
        # Get run type preference
        run_type = get_run_type()
        
        if run_type == 1:
            # single run
            print("\nRunning single genetic algorithm...\n")
            best_solution, history = genetic_algorithm(
                population_size=100,
                mutation_rate=0.02,
                generations_no_improve=30
            )
            # print results
            print("\nBest solution found:")
            print_solution(best_solution)
            
            # plot single run fitness curve
            plot_single_run(history)
            
        else:
            # otherwise, run multiple runs
            num_runs = get_number_of_runs()
            best_solution, all_histories = run_multiple_ga(num_runs)
            
            # print overall best results
            print("\nBest solution found across all runs:")
            print_solution(best_solution)
            
            # plot multi-runs results
            plot_multiple_runs(all_histories)
    else:
        print("\nAlgorithm execution cancelled.")