import pm4py
from pm4py.algo.analysis.woflan import algorithm as woflan
import warnings
import math
from collections import defaultdict
import networkx as nx

warnings.filterwarnings("ignore")

from pm4py.algo.simulation.playout.petri_net import algorithm as playout

#  HELPERS

def jaccard(set_a: set, set_b: set) -> float:
    if not set_a and not set_b:
        return 1.0
    union = set_a | set_b
    return len(set_a & set_b) / len(union) if union else 0.0


def metric_sim(va, vb) -> float:
    if va == 0 and vb == 0:
        return 1.0
    return min(va, vb) / max(va, vb) if max(va, vb) > 0 else 0.0

def is_visible_transition(t):
    return t.label is not None


def is_silent_transition(t):
    return t.label is None


def model_dfg_tau_closure(net, max_depth=50) -> set:
    """
    Extracts approximate directly-follows pairs from a Petri net by traversing
    through places and silent transitions between visible transitions.

    It adds (A, B) if visible transition B can be reached from visible transition A
    through only places and silent transitions.

    # Method: Tau-closure based Directly-Follows Graph (DFG) extraction
    # Computes the set of directly-follows pairs (A -> B) from a Petri net
    # by traversing paths composed of places and silent (tau) transitions.
    # This approximates the behavioural footprint of the model at the level
    # of visible activities, enabling model-to-model comparison without
    # requiring an event log.
    #
    # Citation:
    #   van der Aalst, W.M.P. (2016).
    #   Process Mining: Data Science in Action (2nd ed.), Chapter 6.
    #   Springer, Berlin, Heidelberg.
    """
    pairs = set()
    visible_transitions = [t for t in net.transitions if is_visible_transition(t)]

    for t_start in visible_transitions:
        stack = []
        visited = set()

        # start from output nodes of visible transition
        for arc in t_start.out_arcs:
            stack.append((arc.target, 0))

        while stack:
            node, depth = stack.pop()

            if depth > max_depth:
                continue

            if node in visited:
                continue
            visited.add(node)

            # Case 1: reached a place
            if node in net.places:
                for arc in node.out_arcs:
                    next_node = arc.target

                    if next_node in net.transitions:
                        if is_visible_transition(next_node):
                            pairs.add((t_start.label, next_node.label))
                        else:
                            stack.append((next_node, depth + 1))

            # Case 2: reached a silent transition
            elif node in net.transitions:
                if is_silent_transition(node):
                    for arc in node.out_arcs:
                        stack.append((arc.target, depth + 1))

    return pairs


def get_structural_stats(net) -> dict:
    in_deg  = defaultdict(int)
    out_deg = defaultdict(int)
    for arc in net.arcs:
        out_deg[arc.source] += 1
        in_deg[arc.target]  += 1

    n_places   = len(net.places)
    n_trans    = len(net.transitions)
    n_arcs     = len(net.arcs)
    n_visible  = sum(1 for t in net.transitions if t.label is not None)
    n_silent   = sum(1 for t in net.transitions if t.label is None)
    avg_degree = (2 * n_arcs) / (n_places + n_trans) if (n_places + n_trans) else 0

    and_splits = sum(1 for t in net.transitions if out_deg[t] > 1)
    xor_splits = sum(1 for p in net.places      if out_deg[p] > 1)
    and_joins  = sum(1 for t in net.transitions if in_deg[t]  > 1)
    xor_joins  = sum(1 for p in net.places      if in_deg[p]  > 1)

    connected_places = set()
    connected_trans  = set()
    for arc in net.arcs:
        if arc.source in net.places:      connected_places.add(arc.source)
        if arc.target in net.places:      connected_places.add(arc.target)
        if arc.source in net.transitions: connected_trans.add(arc.source)
        if arc.target in net.transitions: connected_trans.add(arc.target)

    isolated_p = [p.name for p in net.places      if p not in connected_places]
    isolated_t = [t.name for t in net.transitions if t not in connected_trans]

    all_nodes = list(net.places) + list(net.transitions)
    degs = [(in_deg[n] + out_deg[n]) for n in all_nodes]

    return {
        "n_places":   n_places,
        "n_trans":    n_trans,
        "n_arcs":     n_arcs,
        "n_visible":  n_visible,
        "n_silent":   n_silent,
        "avg_degree": avg_degree,
        "and_splits": and_splits,
        "xor_splits": xor_splits,
        "and_joins":  and_joins,
        "xor_joins":  xor_joins,
        "isolated_p": isolated_p,
        "isolated_t": isolated_t,
        "deg_min":    min(degs) if degs else 0,
        "deg_max":    max(degs) if degs else 0,
        "deg_avg":    sum(degs) / len(degs) if degs else 0,
    }


def interpret(score: float) -> str:
    if score > 0.7:  return "high  (>0.7)"
    if score >= 0.4: return "medium (0.4–0.7)"
    return "low  (<0.4)"


def petri_net_to_nx_graph(net):
    G = nx.DiGraph()
    for p in net.places:
        G.add_node(str(p), type="place", label="place")
    for t in net.transitions:
        G.add_node(str(t), type="transition", label=t.label if t.label is not None else "tau")
    for arc in net.arcs:
        G.add_edge(str(arc.source), str(arc.target))
    return G


def node_subst_cost(n1, n2):
    if n1["type"] != n2["type"]:
        return 1.0
    if n1["type"] == "place":
        return 0.0
    if n1["label"] == "tau" and n2["label"] == "tau":
        return 0.0
    return 0.0 if n1["label"] == n2["label"] else 1.0

def node_del_cost(n):
    if n["type"] == "place":  return 0.2
    if n["label"] == "tau":   return 0.2
    return 1.0

def node_ins_cost(n):
    if n["type"] == "place":  return 0.2
    if n["label"] == "tau":   return 0.2
    return 1.0

def edge_del_cost(e):  return 0.2
def edge_ins_cost(e):  return 0.2
def edge_subst_cost(e1, e2): return 0.0


def graph_edit_distance_similarity(net_a, net_b, timeout=10):
    # Method: Graph Edit Distance (GED)
    # Computes the minimum number of edit operations (node/edge insertions,
    # deletions, and substitutions) required to transform graph G_a into G_b.
    # Places and silent transitions carry lower edit costs than visible
    # activity transitions, reflecting their structural rather than semantic role.
    # The raw GED is normalised by the total size of both graphs and then
    # converted to a similarity score: sim = max(0, 1 - GED_norm).
    #
    # Citation:
    #   Bunke, H. (1997).
    #   On a relation between graph edit distance and maximum common subgraph.
    #   Pattern Recognition Letters, 18(8), 689-694. Elsevier.
    G_a = petri_net_to_nx_graph(net_a)
    G_b = petri_net_to_nx_graph(net_b)
    ged = nx.graph_edit_distance(
        G_a, G_b,
        node_subst_cost=node_subst_cost,
        node_del_cost=node_del_cost,
        node_ins_cost=node_ins_cost,
        edge_del_cost=edge_del_cost,
        edge_ins_cost=edge_ins_cost,
        edge_subst_cost=edge_subst_cost,
        timeout=timeout
    )
    if ged is None:
        return None, None, G_a, G_b
    max_size = (
        G_a.number_of_nodes() + G_b.number_of_nodes() +
        G_a.number_of_edges() + G_b.number_of_edges()
    )
    ged_norm = ged / max_size if max_size > 0 else 0.0
    ged_sim  = max(0.0, 1.0 - ged_norm)
    return ged_sim, ged, G_a, G_b


def trace_to_tuple(trace):
    return tuple(event["concept:name"] for event in trace if "concept:name" in event)


def sample_traces(net, im, n=1000):
    return playout.apply(
        net, im,
        variant=playout.Variants.BASIC_PLAYOUT,
        parameters={"noTraces": n}
    )


def variants_from_sampled_log(log):
    return set(trace_to_tuple(trace) for trace in log)


def trace_sampling_similarity(net_a, im_a, net_b, im_b, n=1000):
    # Method: Stochastic Trace Sampling + Jaccard Similarity
    # Generates synthetic event logs by simulating random traces through
    # each Petri net using basic playout (uniform random token firing).
    # The resulting sets of unique trace variants are compared using
    # Jaccard similarity to estimate behavioural overlap between the two models.
    # This is a log-free, sampling-based approximation of language inclusion.
    #
    # Citation:
    #   Leemans, S.J.J., Fahland, D., & van der Aalst, W.M.P. (2014).
    #   Comparing process models using behavioural profiles.
    #   In: Business Process Management Workshops,
    #   Lecture Notes in Business Information Processing,
    #   vol. 202, pp. 129-140. Springer.
    log_a = sample_traces(net_a, im_a, n)
    log_b = sample_traces(net_b, im_b, n)
    variants_a = variants_from_sampled_log(log_a)
    variants_b = variants_from_sampled_log(log_b)
    return jaccard(variants_a, variants_b), variants_a, variants_b


#  1. LOAD MODELS

bpmn_a = pm4py.read_bpmn("../models/sepsis/ground/s_ground_truth.bpmn")
net_a, im_a, fm_a = pm4py.convert_to_petri_net(bpmn_a)
print("Ground Truth / Reference model")

bpmn_b = pm4py.read_bpmn("../models/sepsis/model/s_test_i3.bpmn")
net_b, im_b, fm_b = pm4py.convert_to_petri_net(bpmn_b)
print("Generated / LLM model")


#  2. SEMANTIC SIMILARITY  –  Activity Labels
#
# Method: Jaccard Similarity on Activity Label Sets
# Computes the ratio of shared activity labels to the total union of labels
# across both models. A score of 1.0 means both models use exactly the same
# set of visible activity names; 0.0 means they share no activities at all.
# This is a purely lexical (name-based) measure and does not capture
# ordering or structural relationships between activities.
#
# Citation:
#   Jaccard, P. (1912).
#   The distribution of the flora in the alpine zone.
#   New Phytologist, 11(2), 37-50. Wiley-Blackwell.

print("=" * 55)
print("SEMANTIC SIMILARITY  –  Activity Labels")
print("=" * 55)

acts_a = set(t.label for t in net_a.transitions if t.label is not None)
acts_b = set(t.label for t in net_b.transitions if t.label is not None)

only_in_a = acts_a - acts_b
only_in_b = acts_b - acts_a
common    = acts_a & acts_b

print("Activity Coverage")
print(f"  Activities in Model A          : {len(acts_a)}")
print(f"  Activities in Model B          : {len(acts_b)}")
print(f"  Common activities              : {len(common)}")
print(f"  Only in Model A                : {only_in_a  if only_in_a  else '(none)'}")
print(f"  Only in Model B                : {only_in_b  if only_in_b  else '(none)'}")

print("\nJaccard Similarity")
j_sem = jaccard(acts_a, acts_b)
print(f"  Jaccard(acts_A, acts_B)        : {j_sem:.4f}  [{interpret(j_sem)}]")


#  3. STRUCTURAL SIMILARITY  –  Topology
#
# Method: Element-wise Structural Metric Similarity
# Compares key topological properties of two Petri nets: number of places,
# transitions, arcs, visible and silent transitions, and the counts of
# AND/XOR split and join gateways. For each metric, a pairwise similarity
# score is computed as min(a, b) / max(a, b), yielding 1.0 for identical
# values and approaching 0 for large differences. The final structural
# similarity score is the unweighted average across all metrics.
#
# Citation:
#   Mendling, J., Reijers, H.A., & van der Aalst, W.M.P. (2010).
#   Seven process modeling guidelines (7PMG).
#   Information and Software Technology, 52(2), 127-136. Elsevier.

print("=" * 55)
print("STRUCTURAL SIMILARITY  –  Topology")
print("=" * 55)

st_a = get_structural_stats(net_a)
st_b = get_structural_stats(net_b)

display_metrics = [
    ("n_places",   "Places"),
    ("n_trans",    "Transitions"),
    ("n_arcs",     "Arcs"),
    ("n_visible",  "Visible transitions"),
    ("n_silent",   "Silent transitions"),
    ("and_splits", "AND-splits"),
    ("xor_splits", "XOR-splits"),
    ("and_joins",  "AND-joins"),
    ("xor_joins",  "XOR-joins"),
]

print("Topology Overview")
print(f"  {'Metric':<28} {'Model A':>9} {'Model B':>9}")
print(f"  {'─'*48}")
for key, label in display_metrics:
    print(f"  {label:<28} {st_a[key]:>9} {st_b[key]:>9}")
print(f"  {'Avg arc-degree':<28} {st_a['avg_degree']:>9.3f} {st_b['avg_degree']:>9.3f}")
print(f"  {'Degree min':<28} {st_a['deg_min']:>9} {st_b['deg_min']:>9}")
print(f"  {'Degree max':<28} {st_a['deg_max']:>9} {st_b['deg_max']:>9}")

print("\nIsolated Nodes")
print(f"  Model A – isolated places      : {st_a['isolated_p'] if st_a['isolated_p'] else '(none)'}")
print(f"  Model A – isolated transitions : {st_a['isolated_t'] if st_a['isolated_t'] else '(none)'}")
print(f"  Model B – isolated places      : {st_b['isolated_p'] if st_b['isolated_p'] else '(none)'}")
print(f"  Model B – isolated transitions : {st_b['isolated_t'] if st_b['isolated_t'] else '(none)'}")

print("\nStructural Similarity Score")
sim_keys = ["n_places", "n_trans", "n_arcs", "n_visible", "n_silent",
            "and_splits", "xor_splits", "and_joins", "xor_joins"]

sim_scores = []
print(f"  {'Metric':<28} {'Sim':>8}")
print(f"  {'─'*38}")
for key in sim_keys:
    s = metric_sim(st_a[key], st_b[key])
    sim_scores.append(s)
    label = next((lbl for k, lbl in display_metrics if k == key), key)
    print(f"  {label:<28} {s:>8.4f}")

struct_sim = sum(sim_scores) / len(sim_scores)
print(f"\n  Structural similarity (avg)    : {struct_sim:.4f}  [{interpret(struct_sim)}]")

# Graph Edit Distance (GED)
# Method: Graph Edit Distance (GED) — see graph_edit_distance_similarity() above.
# The Petri net is converted to a directed bipartite graph (places and transitions
# as nodes, arcs as edges) and GED is computed with custom edit costs that assign
# higher penalties to visible activity transitions than to places or silent transitions.
# The result is normalised to [0, 1] and inverted to obtain a similarity score.
#
# Citation:
#   Bunke, H. (1997).
#   On a relation between graph edit distance and maximum common subgraph.
#   Pattern Recognition Letters, 18(8), 689-694. Elsevier.

print("\nGraph Edit Distance (GED)")
try:
    ged_sim, ged_value, G_a, G_b = graph_edit_distance_similarity(net_a, net_b, timeout=10)
    if ged_sim is None:
        print("  GED computation timed out.")
    else:
        print(f"  GED distance                  : {ged_value:.4f}")
        print(f"  GED similarity                : {ged_sim:.4f}  [{interpret(ged_sim)}]")
        print(f"  Graph nodes Model A           : {G_a.number_of_nodes()}")
        print(f"  Graph nodes Model B           : {G_b.number_of_nodes()}")
        print(f"  Graph edges Model A           : {G_a.number_of_edges()}")
        print(f"  Graph edges Model B           : {G_b.number_of_edges()}")
except Exception as e:
    print(f"  GED computation failed: {e}")
    ged_sim = None
    ged_value = None


#  4. MODEL-TO-MODEL BEHAVIOUR  –  DFG
#
# Method: Directly-Follows Graph (DFG) Jaccard Similarity via Tau-Closure
# For each model, the set of directly-follows pairs (A -> B) is extracted
# by traversing the Petri net through places and silent transitions
# (tau-closure). The two resulting sets of activity-level pairs are then
# compared using Jaccard similarity. This is an approximation of the
# behavioural footprint of each model at the level of visible activities.
#
# Citation:
#   van der Aalst, W.M.P. (2016).
#   Process Mining: Data Science in Action (2nd ed.), Chapter 6.
#   Springer, Berlin, Heidelberg.

print("=" * 55)
print("MODEL-TO-MODEL BEHAVIOUR  –  DFG")
print("=" * 55)

dfg_a = model_dfg_tau_closure(net_a)
dfg_b = model_dfg_tau_closure(net_b)

only_dfg_a = dfg_a - dfg_b
only_dfg_b = dfg_b - dfg_a
common_dfg = dfg_a & dfg_b

print("Directly-Follows Pairs")
print(f"  DFG pairs  Model A             : {len(dfg_a)}")
print(f"  DFG pairs  Model B             : {len(dfg_b)}")
print(f"  Common DFG pairs               : {len(common_dfg)}")
print(f"  Only in Model A                : {len(only_dfg_a)}")
print(f"  Only in Model B                : {len(only_dfg_b)}")

if only_dfg_a:
    print("\n  Pairs only in A (max 10):")
    for p in sorted(only_dfg_a)[:10]:
        print(f"    {p[0]}  →  {p[1]}")

if only_dfg_b:
    print("\n  Pairs only in B (max 10):")
    for p in sorted(only_dfg_b)[:10]:
        print(f"    {p[0]}  →  {p[1]}")

print("\nJaccard Similarity  (DFG pairs)")
j_dfg = jaccard(dfg_a, dfg_b)
print(f"  Jaccard(DFG_A, DFG_B)          : {j_dfg:.4f}  [{interpret(j_dfg)}]")


#  5. MODEL-TO-MODEL BEHAVIOUR  –  Trace Sampling
#
# Method: Stochastic Trace Sampling + Jaccard Similarity on Variants
# Both Petri nets are simulated using basic playout to generate synthetic
# event logs of n traces each. The unique trace variants (sequences of
# visible activity labels) from each log are extracted and compared using
# Jaccard similarity. This approximates the overlap in the language
# (set of accepted traces) of the two models without requiring a real log.
# A higher score indicates that the two models accept more of the same
# behavioural sequences.
#
# Citation:
#   Leemans, S.J.J., Fahland, D., & van der Aalst, W.M.P. (2014).
#   Comparing process models using behavioural profiles.
#   In: Business Process Management Workshops,
#   Lecture Notes in Business Information Processing,
#   vol. 202, pp. 129-140. Springer.

print("=" * 55)
print("MODEL-TO-MODEL BEHAVIOUR  –  Trace Sampling")
print("=" * 55)

sample_sim, variants_a, variants_b = trace_sampling_similarity(
    net_a, im_a, net_b, im_b, n=5000
)

only_var_a = variants_a - variants_b
only_var_b = variants_b - variants_a
common_var = variants_a & variants_b

print("Sampled Trace Variants")
print(f"  Sampled variants Model A       : {len(variants_a)}")
print(f"  Sampled variants Model B       : {len(variants_b)}")
print(f"  Common sampled variants        : {len(common_var)}")
print(f"  Only in Model A                : {len(only_var_a)}")
print(f"  Only in Model B                : {len(only_var_b)}")

if only_var_a:
    print("\n  Variants only in A (max 5):")
    for v in list(only_var_a)[:5]:
        print(f"    {v}")

if only_var_b:
    print("\n  Variants only in B (max 5):")
    for v in list(only_var_b)[:5]:
        print(f"    {v}")

print("\nJaccard Similarity  (Sampled Trace Variants)")
print(f"  Jaccard(sampled variants A, B) : {sample_sim:.4f}  [{interpret(sample_sim)}]")


#  6. COMBINED SCORE
#
# Method: Weighted Composite Similarity Score
# Combines the three main similarity dimensions — semantic (activity labels),
# structural (Petri net topology), and behavioural (DFG footprint) — into
# a single score using fixed weights. Semantic similarity is weighted
# slightly higher (0.34) to reflect the importance of shared vocabulary,
# while structural and behavioural dimensions each contribute equally (0.33).
#
# Citation:
#   Dijkman, R., Dumas, M., van Dongen, B., Käärik, R., & Mendling, J. (2011).
#   Similarity of business process models: Metrics and evaluation.
#   Information Systems, 36(2), 498-516. Elsevier.

print("=" * 55)
print("COMBINED SIMILARITY SCORE")
print("=" * 55)

combined = 0.34 * j_sem + 0.33 * struct_sim + 0.33 * j_dfg

print("Weighted Composite")
print(f"  Semantic similarity    (w=0.34) : {j_sem:.4f}")
print(f"  Structural similarity  (w=0.33) : {struct_sim:.4f}")
print(f"  Behaviour similarity   (w=0.33) : {j_dfg:.4f}")
print(f"  {'─'*45}")
print(f"  Combined score                  : {combined:.4f}  [{interpret(combined)}]")


#  7. FINAL SUMMARY

print("=" * 55)
print("FINAL SUMMARY")
print("=" * 55)

print(f"\n  {'Check':<40} {'Score':>8}  Interpretation")
print(f"  {'─'*65}")
rows = [
    ("Semantic similarity  (Jaccard acts)",   j_sem),
    ("Structural similarity (avg metric)",    struct_sim),
    ("Structural similarity (GED)", ged_sim if ged_sim is not None else 0.0),
    ("Structural similarity (Jaccard DFG)",    j_dfg),
    ("Behaviour similarity (Trace Sampling)", sample_sim),
    ("Combined score",                        combined)
]
for label, score in rows:
    print(f"  {label:<40} {score:>8.4f}  {interpret(score)}")