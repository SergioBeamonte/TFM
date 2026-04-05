from influence_diagram import *
import copy
import numpy as np

def diagram_analysis(diagram):
    # Add memory arcs
    add_memory_arcs(diagram)
    # Find if there is a path from the utility node to all decision nodes
    if not search_all_decision_node(diagram):
        return None
    # Eliminar sumideros
    eliminate_sinks(diagram)
    
    while len(diagram.node_utility.predecessors) > 0:
        # Find if there is a chance node that can be eliminated
        if not eliminate_chance_node(diagram):
            # Find If there is a decision node that can be eliminated
            if not delete_decision_node(diagram):
                for node1 in diagram.nodes.values():
                    # Find if there is a chance node and it does not have a decision node as a successor
                    if node1.type == "CHANCE" and not any(node2.type == "DECISION" for node2 in node1.sucessors(diagram.nodes)):
                        # Verify that it does not have more successors than the utility node
                        while len(node1.sucessors(diagram.nodes)) > 1:
                            for node2 in node1.sucessors(diagram.nodes):
                                if invertir_nodo(diagram, node1, node2):
                                    break
                        eliminate_chance_node(diagram)
                        break
    return diagram

# Add memory arcs
def add_memory_arcs(diagram):
    update_node = True
    while update_node:
        update_node = False
        for node in diagram.nodes.values():
            if node.type == "DECISION":
                search_decision_node(diagram, node, node)
    return diagram

# Add memory arcs aux
def search_decision_node(diagram, decision_node, node):
    for predecessor in node.predecessors:
        # If it is a decision node, add that node and its predecessors to the search predecessors list and continue searching
        if diagram.nodes[predecessor].type == "DECISION":
            if predecessor not in decision_node.predecessors:
                decision_node.predecessors.append(predecessor)
            for predecessor2 in diagram.nodes[predecessor].predecessors:
                if predecessor2 not in decision_node.predecessors:
                    decision_node.predecessors.append(predecessor2)
                search_decision_node(diagram, decision_node, diagram.nodes[predecessor2])
        else:
            search_decision_node(diagram, decision_node, diagram.nodes[predecessor])
    return None

# Find if there is a path from the utility node to all decision nodes
def search_all_decision_node(diagram):
    decision_total = [node for node in diagram.nodes.values() if node.type == "DECISION"]
    decision_list = []
    search_all_decision_node_aux(diagram, decision_list, [], diagram.node_utility)
    if len(decision_list) == len(decision_total):
        return True
    else:
        return False

# Search all decision nodes aux
def search_all_decision_node_aux(diagram, decision_list, passed_nodes, node):
    if node not in passed_nodes:
        passed_nodes.append(node)
        if node.type == "DECISION":
            decision_list.append(node)
        for predecessor in node.predecessors:
            search_all_decision_node_aux(diagram, decision_list, passed_nodes, diagram.nodes[predecessor])
    return decision_list

# Eliminate sinks
def eliminate_sinks(diagram):
    global_sink = True
    while global_sink:
        global_sink = False
        nodes_copy = diagram.nodes.copy()
        for node in nodes_copy.values():
            if node.type == "UTILITY":
                continue
            sucessors = node.sucessors(nodes_copy)
            if len(sucessors) == 0:
                global_sink = True
                del diagram.nodes[node.name]
                #print("Eliminando nodo sumidero", node.name)
    return diagram

# Eliminate nodes
# Eliminate decision node
def delete_decision_node(diagram):
    for predecessor in diagram.node_utility.predecessors:
        node = diagram.nodes[predecessor]
        # Verify If the conditions to eliminate the node are met
        if node.type == "DECISION" and all(item in node.predecessors for item in diagram.node_utility.predecessors if item != node.name):
            #print("Eliminando nodo de decision", node.name)
            # Adjust order of the matrix
            utility = diagram.node_utility
            decision_index = utility.pointer.index(node.name)
            utility.matrix = np.moveaxis(utility.matrix, decision_index, -1).tolist() # Move the decision node to the last position
            utility.pointer = utility.pointer[:decision_index] + utility.pointer[decision_index+1:] + [node.name] # Move the decision node to the last position
            utility.matrix, matrix = delete_decision_node_aux(diagram, utility.matrix, utility.pointer, node.value)
            utility.pointer = utility.pointer[:-1]
            utility.add_decision(Decision_matrix(node.name, matrix, utility.pointer.copy()))
            # Delete the node
            utility.predecessors.remove(node.name)
            del diagram.nodes[node.name]
            # Eliminate sinks
            eliminate_sinks(diagram)
            return True
    return False

def delete_decision_node_aux(diagram, matrix, pointer, decision_values):
    if np.ndim(matrix) > 1:
        values = []
        decision_matrix = []
        for i in range(len(diagram.nodes[pointer[0]].value)):
            val, del_val = delete_decision_node_aux(diagram, matrix[i], pointer[1:], decision_values)
            values.append(val)
            decision_matrix.append(del_val)
        return values, decision_matrix
    else:
        maximum = max(matrix)
        index = matrix.index(maximum)
        return maximum, decision_values[index]

# Invert node
def invertir_nodo(diagram, node1, node2):
    # Check that there is no other path directed from node1 to node2
    for node in node2.predecessors:
        if node != node1.name:
            if search_path(diagram, node1.name, node):
                return False
                
    # Save Y
    old_node2 = copy.deepcopy(node2)
    # Obtain the new table of Y
    multipy_probability(diagram, node2, node1)
    # Obtain the new table of X
    bayes_theorem(diagram, node1, node2, old_node2)
    return True

# Eliminate chance node
def eliminate_chance_node(diagram):
    for predecessor in diagram.node_utility.predecessors:
        node = diagram.nodes[predecessor]
        # Verify if the conditions to eliminate the node are met
        if node.type == "CHANCE" and len(node.sucessors(diagram.nodes)) == 1:
            # Adjust the utility
            utility = diagram.node_utility
            multipy_probability(diagram, utility, node)
            # Delete the node
            del diagram.nodes[node.name]
            return True
    return False

# Auxiliar functions
def multipy_probability(diagram, node1, node2):
    A = [x for x in node1.predecessors if x not in node2.pointer] # Predecessors of node1 that are not in node2
    B = [x for x in node2.predecessors if x in node1.predecessors] # Predecessors of node2 that are in node1
    C = [x for x in node2.predecessors if x not in node1.predecessors] # Predecessors of node2 that are not in node1
    
    index_node1 = [node1.pointer.index(x) for x in B]
    index_node1.extend([node1.pointer.index(x) for x in A])
    
    if node1.type == "CHANCE":
        index_node1.append(node1.pointer.index(node2.name))
        index_node1.append(node1.pointer.index(node1.name))
        
    node1.matrix = np.transpose(node1.matrix, index_node1).tolist()
    node1.pointer = [node1.pointer[x] for x in index_node1]
    
    index_node2 = [node2.pointer.index(x) for x in B]
    index_node2.append(node2.pointer.index(node2.name))
    others_index2 = [node2.pointer.index(x) for x in C]
    
    if len(others_index2) > 0:
        index_node2 = index_node2[:-1] + others_index2[:-1] + [index_node2[-1]] + [others_index2[-1]]
        
    node2.matrix = np.transpose(node2.matrix, index_node2).tolist()
    node2.pointer = [node2.pointer[x] for x in index_node2]
    
    node1.matrix = multiply_matrix(diagram, B, node1.matrix, node2.matrix)
    
    # Adjust the pointers
    node1.pointer.pop() # Delete the pointer of node2
    add_pointer = [x for x in node2.pointer if x in C] # Add the pointers of node2 that are not in node1
    if len(add_pointer) > 0:
        node1.pointer = node1.pointer[:-1] + add_pointer[:-1] + [node1.pointer[-1]] + [add_pointer[-1]]
        
    # Adjust the predecessors
    node1.predecessors = node1.pointer.copy()
    if node1.type == "CHANCE":
        node1.predecessors.remove(node1.name)
    return 0

def multiply_matrix(diagram, common_predecessors, utility_values, node_values):
    if len(common_predecessors) > 0:
        values = []
        for i in range(len(diagram.nodes[common_predecessors[0]].value)):
            values.append(multiply_matrix(diagram, common_predecessors[1:], utility_values[i], node_values[i]))
        return values
        
    if isinstance(utility_values, list) and len(np.shape(utility_values)) > 2:
        values = []
        for i in range(len(utility_values)):
            values.append(multiply_matrix(diagram, common_predecessors, utility_values[i], node_values))
        return values
    else:
        return np.matmul(utility_values, node_values).tolist()

def search_path(diagram, node1, node2):
    # Search if there is a path from node1 to node2
    for node in diagram.nodes[node2].predecessors:
        if node != node1:
            if search_path(diagram, node1, node) == True:
                return True
    return False

def bayes_theorem(diagram, node1, node2, node2_old):
    # obtain the common predecessors
    B = [x for x in node1.predecessors if x in node2.predecessors and x in node2_old.predecessors]
    A = [x for x in node2.predecessors if x in node1.predecessors and x not in B]
    C = [x for x in node2_old.predecessors if x in node2.predecessors and x not in B]
    
    # node1 -> B, A, value1
    index_node1 = [node1.pointer.index(x) for x in B]
    index_node1.extend([node1.pointer.index(x) for x in A])
    index_node1.append(node1.pointer.index(node1.name))
    node1.matrix = np.transpose(node1.matrix, index_node1)
    node1.pointer = [node1.pointer[x] for x in index_node1]
    
    # node2_old -> B, C, value2, Node1
    index_node2_old = [node2_old.pointer.index(x) for x in B]
    index_node2_old.extend([node2_old.pointer.index(x) for x in C])
    index_node2_old.append(node2_old.pointer.index(node2_old.name))
    index_node2_old.append(node2_old.pointer.index(node1.name))
    node2_old.matrix = np.transpose(node2_old.matrix, index_node2_old)
    node2_old.pointer = [node2_old.pointer[x] for x in index_node2_old]
    
    # node2 B, C, A, value2
    index_node2 = [node2.pointer.index(x) for x in B]
    index_node2.extend([node2.pointer.index(x) for x in C])
    index_node2.extend([node2.pointer.index(x) for x in A])
    index_node2.append(node2.pointer.index(node2.name))
    node2.matrix = np.transpose(node2.matrix, index_node2)
    node2.pointer = [node2.pointer[x] for x in index_node2]
    
    # B, C, A, Value2, Value1
    node1.matrix = bayes_multiply(diagram, A, B, C, node1.matrix, node2.matrix, node2_old.matrix)
    node1.pointer = B + C + A + [node2.name] + [node1.name]
    
    # Adjust the predecessors
    node1.predecessors = node1.pointer.copy()
    if node1.type == "CHANCE":
        node1.predecessors.remove(node1.name)

def bayes_multiply(diagram, A, B, C, node1_values, node2_values, node2_old_values):
    if len(B) > 0:
        values = []
        for i in range(len(diagram.nodes[B[0]].value)):
            values.append(bayes_multiply(diagram, A, B[1:], C, node1_values[i], node2_values[i], node2_old_values[i]))
        return values
        
    if len(C) > 0:
        values = []
        for i in range(len(diagram.nodes[C[0]].value)):
            values.append(bayes_multiply(diagram, A, B, C[1:], node1_values, node2_values[i], node2_old_values[i]))
        return values
        
    if len(A) > 0:
        values = []
        for i in range(len(diagram.nodes[A[0]].value)):
            values.append(bayes_multiply(diagram, A[1:], B, C, node1_values[i], node2_values[i], node2_old_values))
        return values
        
    values = np.zeros(np.shape(node2_old_values))
    for i in range(len(node2_old_values)):
        values[i] = np.divide(node2_old_values[i], node2_values[i]).tolist()
        values[i] = np.multiply(values[i], node1_values).tolist()
    return values

def init_individual(diagram):
    c = copy.deepcopy(diagram)
    c.instance_probabilities()
    return c