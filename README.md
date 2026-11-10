# When Mistakes Cost: Posterior Plan Selection for Program World-Model Agents

Kavya Shree, Pandit Deendayal Energy University. ARC Prize 2026, Paper Track.

- **Paper:** [paper/paper.pdf](paper/paper.pdf)
- **One-line summary:** keep rival world-model programs and act on the plan most likely to win per action. In grid games with hidden rules, this beats an optimistic single-program baseline by 1.65 to 2.33 actions when wrong moves are costly, with about 20 times fewer model simulations. When mistakes are free, optimism is equal or slightly better.
- **Reproduce every number:** `./reproduce.sh` (about 20 minutes on 4 CPUs, no GPU, no network).

## Pilot experiments: plan-relevant hypothesis testing in unknown grid games

Every number in the paper's pilot sections comes from this folder.

## One command

```bash
./reproduce.sh
```

It reruns every experiment into `fresh/`, checks that each raw result file is
byte-identical to the committed one, and checks that `numbers.json` (the source of
every number in the paper) matches the raw results. Tested with Python 3.13.18 and
numpy 2.5.3 on 4 CPUs; a full run takes about 20 minutes, mostly full posterior scoring in pilot 5. No GPU and no network are needed.

## Files

| File | What it is |
| --- | --- |
| `pilot.py` | Pilot 1: 288-hypothesis games, four policies |
| `pilot2.py` | Pilots 2 to 4: scaling, sampling, EMPA-style and WorldCoder-style baselines, hazards, hazard-cost sweep (`HAZARD_COST` env var, default 10) |
| `analyze_all.py` | Recomputes every reported number into `numbers.json`; `--check` verifies |
| `res_*.json`, `res2/`, `res3/`, `res4/` | Raw per-game results: `[actions, won, model simulations, oracle actions]` |
| `numbers.json` | Every reported mean, 95% CI, win rate, compute and paired saving |

## The games

An 8×8 grid with hidden rules: which key moves which way (24 options), what each
obstacle color does (blocks or passable; with hazards, also "resets to start, +10
actions"), and which target color wins. The agent sees only its position and a win
flag. Every policy keeps the exact set of hypotheses consistent with its history.

## Policies

| Name in code | Policy |
| --- | --- |
| `random` | Uniform random actions |
| `single` | One random consistent hypothesis; plan to its goal; replan on surprise |
| `curiosity` | Full identification first: max predictive-entropy actions until nothing reachable is informative |
| `empa` | EMPA-style (our simplified reimplementation): learn own controls, then seek contact with object types of unknown effect, then exploit |
| `worldcoder` | WorldCoder-style (our simplified reimplementation): one consistent program, the most optimistic one (shortest route to a win) |
| `ours-full`, `ours-sN` | Plan with the highest posterior P(win given the plan), scored on all or N sampled hypotheses |
| `rate-sN` | Ours, revised: highest P(win given the plan) per action, N sampled hypotheses |

The EMPA-style and WorldCoder-style policies capture one mechanism of each paper
(theory-directed exploration of unknown object types; optimism as a constraint on a
single program). They are not reimplementations of the full systems.

## Protocol

Development seeds 101 to 103 were used to choose between variants. Seeds 1 to 3 are
held out and are the only ones reported. The hazard test (pilot 3) was written down
before the held-out run, including policies, sizes, sample size and hazard cost.

## Determinism

Each game and each policy run has its own numpy seed, derived from the experiment seed
and the game index. Outputs depend only on the code and those seeds.
