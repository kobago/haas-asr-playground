"""Compare the ONNX conversions with the PyTorch model on JVNV, utterance by utterance.

python compare_jvnv.py ../results/hyp/

Prints how many hypotheses are identical and the utterances where the ONNX
version loses the most characters (cer_strict x reference length).
"""
import json, sys, jacer
H = sys.argv[1]
a = {r["id"]: r for r in map(json.loads, open(H + "haas_fp16_jvnv.jsonl"))}
for name in ["onnx-fp32_t6_jvnv", "onnx-q_t6_jvnv"]:
    b = {r["id"]: r for r in map(json.loads, open(H + name + ".jsonl"))}
    same = sum(a[i]["hyp"] == b[i]["hyp"] for i in a)
    diffs = []
    for i in a:
        ca = jacer.cer([a[i]["ref"]], [a[i]["hyp"]], ["cer_strict"])["cer_strict"]
        cb = jacer.cer([a[i]["ref"]], [b[i]["hyp"]], ["cer_strict"])["cer_strict"]
        diffs.append(((cb - ca) * len(a[i]["ref"]), i))
    diffs.sort(reverse=True)
    longs = sum(len(b[i]["hyp"]) > 2 * len(b[i]["ref"]) + 10 for i in b)
    print("==", name, "identical:", same, "/", len(a), "long hyps:", longs)
    for d, i in diffs[:4]:
        print(f" +{d:.0f} {i}\n   REF {a[i]['ref']}\n   PT  {a[i]['hyp']}\n   ONX {b[i]['hyp'][:120]}")
