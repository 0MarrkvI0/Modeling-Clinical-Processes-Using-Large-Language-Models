import pm4py
from pm4py.algo.conformance.tokenreplay import algorithm as tbr
from pm4py.algo.analysis.woflan import algorithm as woodland
from pm4py.algo.evaluation.generalization import algorithm as generalization_evaluator
from pm4py.algo.evaluation.simplicity import algorithm as simplicity_evaluator
from pm4py.algo.evaluation.precision import algorithm as precision_evaluator
from pm4py.statistics.variants.log import get as variants_get
from pm4py.objects.log.obj import EventLog
import warnings
warnings.filterwarnings("ignore")


from pm4py.statistics.variants.log import get as variants_get
from pm4py.objects.log.obj import EventLog

# 1. LOAD FILES
log  = pm4py.read_xes("../logs/sepsis_cases_log.xes")
log = pm4py.convert_to_event_log(log)

bpmn = pm4py.read_bpmn("../models/sepsis/model/s_test_i3.bpmn")
net, im, fm = pm4py.convert_to_petri_net(bpmn)

print(f"Loaded traces: {len(log)}")
print(f"Loaded events : {sum(len(trace) for trace in log)}")

print("=" * 55)
print("  PROCESS MODEL EVALUATION REPORT")
print("=" * 55)


# 2. SOUNDNESS  (Woflan)
# Method: Woflan (Workflow net analyser)
# Verifies whether the Petri net is "sound" — i.e. whether it is free of
# deadlocks, livelocks, and unbounded places.
# A sound workflow net guarantees that every process instance can always
# reach the final marking and that no tokens are left behind after completion.
#
# Citation:
#   Verbeek, H.M.W., Basten, T., & van der Aalst, W.M.P. (2001).
#   Diagnosing workflow processes using Woflan.
#   The Computer Journal, 44(4), 246-279. Oxford University Press.

print("\nSoundness (Woflan)")
try:
    is_sound = woodland.apply(net, im, fm, parameters={
        woodland.Parameters.RETURN_ASAP_WHEN_NOT_SOUND: True,
        woodland.Parameters.PRINT_DIAGNOSTICS: True,
        woodland.Parameters.RETURN_DIAGNOSTICS: False
    })
    print(f"  Sound : {is_sound}")
except Exception as e:
    print(f"  Soundness check failed: {e}")

# 3. FITNESS  (alignment-based)
# Method: Optimal alignment using the A* algorithm
# on the synchronisation product of the log and the model.
# Finds the optimal sequence of edit operations (moves) between
# a trace and the model with minimum cost.
# A fitness value of 1.0 means every trace in the log can be
# replayed perfectly on the model; lower values indicate deviations.
#
# Citation:
#   Adriansyah, A., van Dongen, B.F., & van der Aalst, W.M.P. (2011).
#   Conformance checking using cost-based fitness analysis.
#   In: IEEE 15th International Enterprise Distributed Object
#   Computing Conference (EDOC), pp. 55-64. IEEE.

print("\nETC / Alignment-Based Fitness")
try:
    fitness_align = pm4py.fitness_alignments(log, net, im, fm)
    print(f"  Alignment fitness (avg)    : {fitness_align['average_trace_fitness']:.4f}")
    print(f"  % Fitting traces (align)   : {fitness_align['percentage_of_fitting_traces']:.4f}")
    print(f"  Log fitness (align)        : {fitness_align['log_fitness']:.4f}")
except Exception as e:
    print(f"  Alignment-based fitness failed: {e}")
    fitness_align = None

# 4. PRECISION  (alignment-based / Align-ETConformance)
# Method: Align-ETConformance
# An extension of ETConformance using an alignment-based approach.
# More accurate than token-based precision but computationally more expensive.
# Measures how much behaviour allowed by the model is actually observed in the log.
# A precision of 1.0 means the model allows no behaviour beyond what is in the log;
# lower values indicate the model is overly permissive.
#
# Citation:
#   Adriansyah, A., van Dongen, B.F., Munoz-Gama, J., & Carmona, J. (2013).
#   Alignment-based precision checking.
#   In: Business Process Management Workshops,
#   Lecture Notes in Business Information Processing,
#   vol. 132, pp. 137-149. Springer.

print("\nPrecision (Alignment-Based / ETC)")
try:
    precision_align = pm4py.precision_alignments(log, net, im, fm)
    print(f"  Precision (alignment) : {precision_align:.4f}")
except Exception as e:
    print(f"  Alignment-based precision failed: {e}")
    precision_align = None

# 4.5 F1 SCORE (fitness + precision)
# Method: Harmonic mean of fitness and precision
# Combines fitness (recall) and precision into a single balanced score.
# Useful when comparing models where one may trade off fitness for precision
# or vice versa. A score of 1.0 indicates both perfect fitness and precision.
print("\nF1 Score (Fitness + Precision)")
try:
    if fitness_align and precision_align is not None:
        fitness_val = fitness_align['log_fitness']
        precision_val = precision_align

        if (fitness_val + precision_val) > 0:
            f1_score = 2 * (fitness_val * precision_val) / (fitness_val + precision_val)
        else:
            f1_score = 0.0

        print(f"  F1 Score : {f1_score:.4f}")
    else:
        f1_score = None
        print("  F1 Score : N/A")
except Exception as e:
    print(f"  F1 computation failed: {e}")
    f1_score = None


# 5. GENERALIZATION
# Method: Token-replay based generalization
# Measures whether the model is "overfitted" to only the traces present in the log.
# Computed based on the frequency of use of each transition in the Petri net
# during token replay. Transitions that are rarely fired lower the generalization score.
# A score close to 1.0 indicates that the model generalises well beyond the observed log.
#
# Citation:
#   Buijs, J.C.A.M., van Dongen, B.F., & van der Aalst, W.M.P. (2012).
#   On the role of fitness, precision, generalization and simplicity
#   in process discovery.
#   In: On the Move to Meaningful Internet Systems (OTM 2012),
#   Lecture Notes in Computer Science, vol. 7565, pp. 305-322. Springer.

print("\nGeneralization")
try:
    gen = generalization_evaluator.apply(log, net, im, fm)
    print(f"  Generalization : {gen:.4f}")
except Exception as e:
    print(f"  Generalization computation failed: {e}")
    gen = None

# 6. SIMPLICITY  (arc degree)
# Method: Arc degree simplicity
# Measures the complexity of the model via the average number of arcs
# per node in the Petri net. Formula: 1 / (1 + avg_arc_degree).
# A higher score indicates a simpler model with fewer connections per node.
# Simpler models are generally easier to understand and maintain.
#
# Citation:
#   Mendling, J., Reijers, H.A., & van der Aalst, W.M.P. (2010).
#   Seven process modeling guidelines (7PMG).
#   Information and Software Technology, 52(2), 127-136. Elsevier.

print("\nSimplicity")
try:
    simplicity = simplicity_evaluator.apply(net)
    print(f"  Simplicity : {simplicity:.4f}")
except Exception as e:
    print(f"  Simplicity computation failed: {e}")
    simplicity = None


# 7. NON-FITTING TRACES  (detailed replay)
print("\n── Non-Fitting Traces (first 5, token replay) ─")
try:
    replayed = tbr.apply(log, net, im, fm)

    # prepare variants (trace -> list of traces)
    variants = variants_get.get_variants(log)

    # mapping: trace (tuple) -> frequency
    variant_freq = {variant: len(traces) for variant, traces in variants.items()}

    count = 0
    for trace, tr in zip(log, replayed):

        if tr["trace_fitness"] < 1.0:

            acts = tuple(e["concept:name"] for e in trace if "concept:name" in e)

            print(f"\n  Trace fitness: {tr['trace_fitness']:.3f}"
                  f" | missing: {tr['missing_tokens']}"
                  f" | remaining: {tr['remaining_tokens']}")

            # ORIGINAL TRACE (variant)
            print(f"  Variant (original trace): {acts}")

            # FREQUENCY in log
            freq = variant_freq.get(acts, 1)
            print(f"  Frequency in log       : {freq}")

            count += 1
            if count >= 5:
                break

    if count == 0:
        print("  All traces are fitting!")

except Exception as e:
    print(f"  Detailed replay failed: {e}")


# SUMMARY TABLE

print("\n" + "=" * 55)
print("  SUMMARY")
print("=" * 55)
rows = [
    ("Soundness",                          locals().get("is_sound",        "N/A")),
    ("Fitness (alignment, log)",           f"{fitness_align['log_fitness']:.4f}"   if fitness_align  else "N/A"),
    ("Precision (alignment / ETC)",        f"{precision_align:.4f}"                 if precision_align else "N/A"),
    ("F1 Score",                         f"{f1_score:.4f}"                       if f1_score is not None else "N/A"),
    ("Generalization",                     f"{gen:.4f}"                             if gen            else "N/A"),
    ("Simplicity",                         f"{simplicity:.4f}"                      if simplicity     else "N/A"),
]
for label, value in rows:
    print(f"  {label:<36} {value}")
print("=" * 55)