import sys
import time
import argparse
from progress import Progress
import random
import networkx as nx

def load_graph(main_args):
    """Load graph from text file

    Parameters:
    args -- arguments named tuple

    Returns:
    A dict mapling a URL (str) to a list of target URLs (str).
    """
    main_graph = {}

    for line in main_args.datafile:
        # Split every line into two URLs
        node, target = line.split()

        # Check if the source node is in the main graph
        if node not in main_graph:
            main_graph[node] = []

        # Append the target to the list of edges the source node has
        main_graph[node].append(target)

        if target not in main_graph:
            main_graph[target] = []

    return main_graph

def print_stats(print_graph):
    """Print number of nodes and edges in the given graph."""
    # Count how many nodes there are
    nodes = len(print_graph)
    # Count how many edges there are
    edges = sum(len(targets) for targets in print_graph.values())
    # Print both total quantities
    print(f"Number of nodes: {nodes}")
    print(f"Number of edges: {edges}")


def represent_adj_matrix(main_graph):
    """Represent the graph by Adjacency Matrix"""
    nodes = list(main_graph.keys())
    node_indices = {node: index_matrix for index_matrix, node in enumerate(nodes)}
    size = len(nodes)

    # Initialize adjacency matrix with zeros
    matrix = [[0] * size for _ in range(size)]

    # Complete the matrix
    for node, targets in main_graph.items():
        for target in targets:
            matrix[node_indices[node]][node_indices[target]] = 1

    return matrix, nodes

def print_adj_matrix(adj_matrix):
    # Print statistics for the adjacency matrix
    num_nodes = len(adj_matrix)
    num_edges = sum(sum(row) for row in adj_matrix)
    print(f"Number of nodes: {num_nodes}")
    print(f"Number of edges: {num_edges}")

def represent_edge_list(main_graph):
    """Represent the graph by Edge List"""
    # Initialize with an empty edge list
    edge_list = []
    # Iterate over each node and their respective of target nodes in the adjacency list
    for node, targets in main_graph.items():
        for target in targets:
            edge_list.append((node, target))

    return edge_list

def print_edge_list(edge_list):
    # Print statistics for the edge list
    num_nodes = len(set(node for edge in edge_list for node in edge))
    num_edges = len(edge_list)
    print(f"Number of nodes: {num_nodes}")
    print(f"Number of edges: {num_edges}")

def represent_networkx(main_graph):
    """Represent the graph by Network X"""
    # Initialize in Network X an empty directed graph
    nx_graph = nx.DiGraph()
    # Iterate over each node and their respective of target nodes in the adjacency list
    for node, targets in main_graph.items():
        for target in targets:
            nx_graph.add_edge(node, target)
    return nx_graph

def print_networkx(nx_graph):
    # Print statistics for network x
    num_nodes = nx_graph.number_of_nodes()
    num_edges = nx_graph.number_of_edges()
    density = nx.density(nx_graph)
    print(f"Number of nodes: {num_nodes}")
    print(f"Number of edges: {num_edges}")
    print(f"Density: {density:.4f}")


def stochastic_page_rank(stoc_graph, n_repetitions):
    """Stochastic PageRank estimation

    Parameters:
    graph -- a graph object as returned by load_graph()
    args -- arguments named tuple

    Returns:
    A dict that assigns each page its hit frequency

    This function estimates the Page Rank by counting how frequently
    a random walk that starts on a random node will after n_steps end
    on each node of the given graph.
    """

    # Initialize hit count with 0 for all nodes
    hit_count = {node: 0 for node in stoc_graph}
    nodes = list(stoc_graph.keys())

    # Start from a randomly selected node
    current_node = random.choice(nodes)
    hit_count[current_node] += 1

    # Implement and set up the progress bar
    progress = Progress(n_repetitions, title="Stochastic PageRank")
    progress.show()

    # Simulate random paths
    for repetition in range(n_repetitions):
        # Select a random node if the current one does not have any outgoing links
        if not stoc_graph[current_node]:
            current_node = random.choice(nodes)
        else:
            # Else, follow a random link
            current_node = random.choice(stoc_graph[current_node])
        hit_count[current_node] += 1

        # Update and display progress bar
        if repetition % 1000 == 0 or repetition == n_repetitions - 1:
            progress += 100
            progress.show()

    progress.finish()

    # Convert the hit counts to percentage format
    total_hits = sum(hit_count.values())
    return {node: count / total_hits for node, count in hit_count.items()}


def distribution_page_rank(dis_graph, n_steps):
    """Probabilistic PageRank estimation

    Parameters:
    graph -- a graph object as returned by load_graph()
    args -- arguments named tuple

    Returns:
    A dict that assigns each page its probability to be reached

    This function estimates the Page Rank by iteratively calculating
    the probability that a random walker is currently on any node.
    """

    # Initialize node probabilities with equal probabilities
    nodes = list(dis_graph.keys())
    num_nodes = len(nodes)
    node_prob = {node: 1 / num_nodes for node in nodes}

    # Implement and set up the progress bar
    progress = Progress(n_steps, title="Distribution PageRank")
    progress.show()

    # Iterate for 'n' number of steps
    for step in range(n_steps):
        # Initialize next_prob with 0 for all nodes
        next_prob = {node: 0 for node in nodes}

        # Update the probabilities depending on the outgoing edges
        for node, targets in dis_graph.items():
            if not targets:
                # Divide its probability by all nodes if the node has no outgoing edges
                for target in nodes:
                    next_prob[target] += node_prob[node] / num_nodes
            else:
                # Else, divide its probability by its outgoing edges
                p = node_prob[node] / len(targets)
                for target in targets:
                    next_prob[target] += p

        # Update the node probabilities
        node_prob = next_prob

        # Update and display the progress bar
        if step % 1000 == 0 or step == step - 1:
            progress += 100  # Update progress based on interval
            progress.show()

    progress.finish()

    return node_prob


def set_parser():
    parser = argparse.ArgumentParser(description="Estimates page ranks from link information")
    parser.add_argument('datafile', nargs='?', type=argparse.FileType('r'), default=sys.stdin,
                        help="Textfile of links among web pages as URL tuples")
    parser.add_argument('-m', '--method', choices=('stochastic', 'distribution'), default='stochastic',
                        help="Selected PageRank algorithm")
    parser.add_argument('-r', '--repeats', type=int, default=1_000_000,
                        help="Number of repetitions (for stochastic method)")
    parser.add_argument('-s', '--steps', type=int, default=100,
                        help="Number of steps a walker takes (for distribution method)")
    parser.add_argument('-n', '--number', type=int, default=20,
                        help="Number of top results to display")
    parser.add_argument('--representation', choices=('adj_list', 'adj_matrix', 'edge_list', 'networkx'),
                        default='adj_list',
                        help="Graph representation type for statistics")
    return parser

def main(args):
    algorithm = distribution_page_rank if args.method == 'distribution' else stochastic_page_rank

    graph = load_graph(args)

    # Argument Parsing for each of the 4 ways to represent the graph:
    if args.representation == 'adj_list':
        # Adjacency List
        print("Using Adjacency List Representation")
        print_stats(graph)
    elif args.representation == 'adj_matrix':
        # Adjacency Matrix
        adjacency_matrix, node_list = represent_adj_matrix(graph)
        print("Using Adjacency Matrix Representation")
        print_adj_matrix(adjacency_matrix)
    elif args.representation == 'edge_list':
        # Edge List
        edge_list = represent_edge_list(graph)
        print("Using Edge List Representation")
        print_edge_list(edge_list)
    elif args.representation == 'networkx':
        # NetworkX
        nx_graph = represent_networkx(graph)
        print("Using Network X Representation")
        print_networkx(nx_graph)

    # Measure runtime for the selected PageRank algorithm
    start_time = time.time()

    # Run the selected PageRank algorithm
    if args.method == 'stochastic':
        ranking = algorithm(graph, args.repeats)
    else:
        ranking = algorithm(graph, args.steps)

    elapsed_time = time.time() - start_time

    # Display top pages (results)
    top = sorted(ranking.items(), key=lambda item: item[1], reverse=True)
    sys.stderr.write(f"Top {args.number} pages:\n")
    print('\n'.join(f'{100 * v:.2f}\t{k}' for k, v in top[:args.number]))
    sys.stderr.write(f"Calculation took {elapsed_time:.2f} seconds.\n")


if __name__ == '__main__':
    parser = set_parser()
    args = parser.parse_args()

    main(args)
