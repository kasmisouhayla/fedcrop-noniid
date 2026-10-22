"""Merge the raw outputs of experiments/run_main.py (parts A and B) and run_extra.py (A and B)."""
import json, os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = lambda f: json.load(open(os.path.join(ROOT, "results", f)))
A, B = R("raw_main_A.json"), R("raw_main_B.json")
main = {"config": A["config"], "config_dp": B["config_dp"], "E1": A["E1"], "E2": B["E2"], "E3": B["E3"], "E4": B["E4"], "E5": B["E5"]}
json.dump(main, open(os.path.join(ROOT, "results", "results_main.json"), "w"))
XA, XB = R("raw_extra_A.json"), R("raw_extra_B.json")
json.dump(XA, open(os.path.join(ROOT, "results", "results_yield.json"), "w"))
json.dump(XB, open(os.path.join(ROOT, "results", "results_extra.json"), "w"))
print("merged into results/results_main.json, results_yield.json, results_extra.json")
