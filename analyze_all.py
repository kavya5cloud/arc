"""Every number reported in the paper's pilot sections, recomputed from raw results.

Writes numbers.json. With --check, compares against the committed numbers.json and
exits non-zero on any difference, so a reviewer can verify the paper in one command.

Statistics: per-game mean over seeds, then mean over games with a 95% percentile
bootstrap over games (2,000 resamples, numpy seed 0). Paired savings use the same
per-game means. Compute = median model simulations per game over all seeds.
"""
import json, glob, sys, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def load(pattern):
    runs = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(HERE, pattern)))]
    merged = {}
    for run in runs:
        for k, v in run.items():
            if k != "H":
                merged.setdefault(k, []).append(v)
    return merged, (runs[0].get("H") if runs else None)


def summarise(merged, rng, pairs):
    acts = {p: np.stack([np.array([r[0] for r in v], float) for v in vs]).mean(0) for p, vs in merged.items()}
    out = {}
    for p, vs in merged.items():
        idx = rng.integers(len(acts[p]), size=(2000, len(acts[p])))
        lo, hi = np.percentile(acts[p][idx].mean(1), [2.5, 97.5])
        flat = [r for v in vs for r in v]
        out[p] = {"mean": round(float(acts[p].mean()), 2), "ci": [round(float(lo), 2), round(float(hi), 2)],
                  "win_pct": round(100 * float(np.mean([r[1] for r in flat])), 1),
                  "median_sims": int(np.median([r[2] for r in flat])),
                  "seeds": len(vs), "games": len(acts[p])}
    for a, b in pairs:  # saving of a relative to b (positive = a needs fewer actions)
        if a in acts and b in acts:
            d = acts[b] - acts[a]
            idx = rng.integers(len(d), size=(2000, len(d)))
            lo, hi = np.percentile(d[idx].mean(1), [2.5, 97.5])
            out[f"{a}_saves_vs_{b}"] = {"mean": round(float(d.mean()), 2), "ci": [round(float(lo), 2), round(float(hi), 2)],
                                        "a_better_pct": round(100 * float((d > 0).mean()), 1)}
    return out


def main():
    rng = np.random.default_rng(0)
    numbers = {}
    m, _ = load("res_*.json")
    numbers["pilot1_H288"] = summarise(m, rng, [("ours", "single"), ("ours", "curiosity")])
    for size in (288, 768, 1920):
        m, _ = load(f"res2/h{size}_s*.json")
        m3, _ = load(f"res3/h{size}_s*.json")
        m.update(m3)
        pairs = [(p, b) for p in ("ours-full", "ours-s8", "ours-s16", "ours-s32", "rate-s16")
                 for b in ("single", "curiosity", "worldcoder", "empa")]
        numbers[f"pilot2_H{size}"] = summarise(m, rng, pairs)
    for size in (648, 2592):
        m, _ = load(f"res3/z{size}_s*.json")
        pairs = [("rate-s16", b) for b in ("single", "curiosity", "worldcoder", "empa")]
        numbers[f"pilot3_hazard_H{size}"] = summarise(m, rng, pairs)
    for cost in (0, 5, 10, 20, 40):
        m, _ = load(f"res4/c{cost}_s*.json")
        pairs = [("rate-s16", "worldcoder"), ("ours-s16", "worldcoder"), ("rate-s16", "ours-s16")]
        numbers[f"pilot4_hazardcost{cost}_H648"] = summarise(m, rng, pairs)
    for size in (648, 2592):
        m3, _ = load(f"res3/z{size}_s*.json")
        m = {k: v for k, v in m3.items() if k in ("worldcoder", "rate-s16")}
        for part in ("sub", "tau"):
            m5, _ = load(f"res5/{part}_z{size}_s*.json")
            m.update(m5)
        pairs = [(p, "rate-full") for p in ("rate-s8", "rate-s16", "rate-s32")]
        pairs += [(p, "worldcoder") for p in ("rate-full", "rate-s8", "rate-s16", "rate-s32", "tau50-s16", "tau80-s16")]
        pairs += [(p, "rate-s16") for p in ("tau50-s16", "tau80-s16")]
        numbers[f"pilot5_checks_H{size}"] = summarise(m, rng, pairs)
    path = os.path.join(HERE, "numbers.json")
    if "--check" in sys.argv:
        ref = json.load(open(path))
        if ref != numbers:
            print("MISMATCH between raw results and numbers.json")
            sys.exit(1)
        print("OK: every reported number matches the raw results")
    else:
        json.dump(numbers, open(path, "w"), indent=1)
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
