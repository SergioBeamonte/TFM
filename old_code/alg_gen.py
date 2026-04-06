from influence_diagram import *
from diagram_prueba.diagram_prueba import *
import random
from nhlvl.nhlvl import *
from evaluate_rules import *
from txtConverter import *
from genetic_algorithm.population import *

# Tournament selection
def tournament_selection(population, tournament_size, n_selections, replacement=False):
    individuals = copy.copy(population.individuals)
    selected = []
    
    for i in range(n_selections):
        tournament = []
        for j in range(tournament_size):
            tournament.append(individuals[np.random.randint(0, len(individuals))])
            
        tournament.sort(key=lambda x: x.fitness, reverse=True)
        selected.append(tournament[0])
        
        if not replacement:
            individuals.remove(tournament[0])
            
    return selected

# Crossover
def crossover(parent1, parent2):
    children = Individual(parent1.diagram)
    for name in parent1.diagram.nodes.keys():
        if not parent1.diagram.nodes[name].type == "DECISION":
            children.diagram.nodes[name].matrix = crossover_aux(children.diagram, parent1.diagram.nodes[name].matrix, parent2.diagram.nodes[name].matrix)
    return children

# Crossover auxiliar
def crossover_aux(diagram, node1_values, node2_values):
    if len(np.shape(node1_values)) == 1:
        return random.choice([node1_values, node2_values])
    else:
        values = []
        for i in range(len(node1_values)):
            values.append(crossover_aux(diagram, node1_values[i], node2_values[i]))
        return values

# Mutation
def mutation(individual, mutation_rate, mutation_variance):
    for name in individual.diagram.nodes.keys():
        if not individual.diagram.nodes[name].type == "DECISION":
            individual.diagram.nodes[name].matrix = mutation_aux(individual.diagram.nodes[name].matrix, mutation_rate, mutation_variance)

# Mutation auxiliar
def mutation_aux(matrix, mutation_rate, mutation_variance):
    if len(np.shape(matrix)) == 1:
        if np.random.rand() < mutation_rate:
            return mutation_vector(matrix, mutation_variance)
        else:
            return matrix
    else:
        values = []
        for i in range(len(matrix)):
            values.append(mutation_aux(matrix[i], mutation_rate, mutation_variance))
        return values

# Mutation vector
def mutation_vector(matrix, mutation_variance):
    mutation_index = random.randint(0, len(matrix) - 1)
    new_probability = matrix[mutation_index] + random.uniform(-mutation_variance, mutation_variance)
    
    # Ensure the new probability is in the range [0, 1]
    new_probability = max(0, min(new_probability, 1))
    
    # Adjust the other probabilities to keep the total sum in 1
    diference = new_probability - matrix[mutation_index]
    matrix[mutation_index] = new_probability
    
    for i in range(len(matrix)):
        if i != mutation_index:
            matrix[i] = matrix[i] - diference / (len(matrix) - 1)
            matrix[i] = max(0, min(matrix[i], 1))
            
    return matrix

# Genetic algorithm
def genetic_algorithm(diagram, population_size, n_generations, elite_size, selection_porcent, tournament_size, replacement, mutation_rate, mutation_variance, expert_rules, total_rules):
    # Init population
    population = Population(diagram, population_size, expert_rules, count_rules(expert_rules, diagram), total_rules, count_rules(total_rules, diagram))
    results = []
    
    for i in range(n_generations-1):
        print("Generation: ", i+1)
        new_population = []
        
        # Evaluate population
        population.evaluate()
        # Save results
        results.append(population.get_puntuations())
        
        # Select parents
        parents = tournament_selection(population, tournament_size, int(population_size * selection_porcent), replacement)
        
        # delete last parent if odd
        if len(parents) % 2 != 0:
            parents.pop()
            
        for j in range(0, len(parents), 2):
            for k in range(0, 4):
                # Crossover
                children = crossover(parents[j], parents[j+1])
                # Mutation
                mutation(children, mutation_rate, mutation_variance)
                new_population.append(children)
                
        # Selection
        # Order population by fitness
        population.individuals.sort(key=lambda x: x.fitness, reverse=True)
        
        # Search if there is space for the elite
        if len(new_population) + elite_size < population_size:
            new_population.extend(population.individuals[:elite_size])
        else:
            new_population.extend(population.individuals[:population_size - len(new_population)])
            
        # Fill the population with Individuals from the previous generation
        while len(new_population) < population_size:
            new_population.append(population.individuals[np.random.randint(0, population_size - 1)])
            
        population.individuals = new_population
        
    population.evaluate()
    results.append(population.get_puntuations())
    
    return results