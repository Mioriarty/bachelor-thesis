import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import itertools
import collections

NO_OPTIMISATION                  = 0b000
REMOVE_NODES_WITH_BIG_COMPONENTS = 0b001
REMOVE_LEAVES                    = 0b010
UNIQUIFY_VEC_PATHS               = 0b100

def generate_vectors_with_sum(n, sum):
    for combi in itertools.combinations_with_replacement(range(n), sum):
        counter = collections.Counter(combi)
        yield np.array([counter[i] for i in range(n)][::-1])

def vec_to_str(vec):
    return "(" + ",".join(str(int(v)) for v in vec) + ")"

def str_to_vec(s):
    return np.array([int(x) for x in s[1:-1].split(",")])

def edge_in_optimized_graph(u, v, b):
    return all(vi <= xi for vi, xi in zip(str_to_vec(v), b))

def generate_weight_lookup(A, w):
    return { vec_to_str(A[:,i]): w[i] for i in range(A.shape[1]) }
    

def get_edge_weight(G, u, v, weight_lookup, layer_cutoff = 1):
    if -G.nodes[v]["layer"] > layer_cutoff:
        return ""
    
    v = str_to_vec(v)
    u = str_to_vec(u)

    return weight_lookup[vec_to_str(v - u)]

def gen_edges_with_unique_paths(A, k,crnt_node = None, crnt_layer=0, last_unit_vector = 0):
    if crnt_layer >= k:
        return []

    (m, n) = A.shape

    if crnt_node is None:
        crnt_node = np.zeros(shape=m)

    edges = []
    for unit_vector in list(range(last_unit_vector, n))[::-1]:
        e = np.array([1 if i == unit_vector else 0 for i in range(n)])
        next_node = crnt_node + A @ e
        edges.append((vec_to_str(crnt_node), vec_to_str(next_node)))
        edges += gen_edges_with_unique_paths(A, k, next_node, crnt_layer + 1, unit_vector)
    
    return edges


def generate_graph_from_ilp(A, b, x, w, optimisation=NO_OPTIMISATION):
    pos = dict()
    k = np.sum(x)
    (m, n) = A.shape

    if optimisation & UNIQUIFY_VEC_PATHS:
        edges = gen_edges_with_unique_paths(A, k)
    else:
        edges = []
        for layer in range(k):
            edges += [(vec_to_str(A @ v), vec_to_str(A @ (v+e))) for v, e in itertools.product(generate_vectors_with_sum(n, layer), generate_vectors_with_sum(n, 1))]
    
    if optimisation & REMOVE_NODES_WITH_BIG_COMPONENTS:
        edges = [e for e in edges if edge_in_optimized_graph(e[0], e[1], b)]
    
    G = nx.DiGraph(edges)

    if optimisation & REMOVE_LEAVES:
        for layer in range(k-1, -1, -1):
            for vec in generate_vectors_with_sum(n, layer):
                node = vec_to_str(A @ vec)

                if node in G.nodes and G.out_degree(node) == 0:
                    G.remove_node(node)
    
    weight_lookup = generate_weight_lookup(A, w)

    # Assign layers to nodes
    for layer in range(k+1):
        for v in generate_vectors_with_sum(n, layer):
            node = vec_to_str(A @ v)
            if node in G.nodes:
                G.nodes[node]["layer"] = -layer
    
    print("Graph nodes:", len(G.nodes))
    print("Graph edges:", len(G.edges))
    
    
    node_colors = ["#FF0000" if vec_to_str(b) == node else "#000000" for node in G.nodes()]

    options = {
        "font_size": 6,
        "node_size": 500,
        "node_color": "white",
        "node_shape": "o",
        "edgecolors": node_colors,
        "edge_color": ["red" if edge_in_optimized_graph(u, v, b) and not optimisation & REMOVE_NODES_WITH_BIG_COMPONENTS else "black" for u, v in G.edges()],
        "linewidths": 1,
        "width": 1,
    }

    plt.figure(figsize=(18, 6))
    # plt.rcParams['text.usetex'] = True
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    pos = nx.multipartite_layout(G, subset_key="layer", align="horizontal")
    nx.draw_networkx(G, pos, **options)

    edge_labels = {(u, v): get_edge_weight(G, u, v, weight_lookup, layer_cutoff=400) for u, v in G.edges()}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8, label_pos=0.6, rotate=False)

    # Set margins for the axes so that nodes aren't clipped
    ax = plt.gca()
    ax.margins(0.20)
    plt.axis("off")
    plt.show()

A = np.array([[1, 2, 3, 6],
              [5, 4, 3, 0]])

x = np.array([1, 1, 3, 2])
b = A @ x

w = np.array([1, 2, 3, 4])

generate_graph_from_ilp(A, b, x, w, REMOVE_NODES_WITH_BIG_COMPONENTS | UNIQUIFY_VEC_PATHS | REMOVE_LEAVES)