"""Pilot 2: scaling the hypothesis space, and cutting compute by sampling.

Same games as pilot.py, generalised: K obstacle colours (each passable or not)
and T target colours (one wins). |H| = 24 * 2^K * T.
  K=2,T=3 -> 288   K=3,T=4 -> 768   K=4,T=5 -> 1920

Policies: random, single, curiosity, ours-full (score plans against every
consistent hypothesis; only run at |H|=288), ours-s32 (plans and scoring use a
random sample of at most 32 consistent hypotheses).
Usage: python3 pilot2.py <seed> <n_games> <K> <T> <policies,comma,separated> [hazard]
Policies also include: worldcoder (one consistent program, the most optimistic: shortest
route to a win), empa (learn own controls, then contact object types of unknown effect,
then exploit), rate-sN (ours: plan with the highest P(win | plan) per action, N samples).
"""
import itertools, sys, json
from collections import deque
import numpy as np

N = 8
DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
PERMS = list(itertools.permutations(range(4)))
CAP = 300
EVALS = 0
START = (0, 0)      # set per game; hazards send the agent back here
HAZARD_COST = int(__import__('os').environ.get('HAZARD_COST', '10'))  # extra actions per reset; 10 pre-registered


def hyps(K, T, hazard=False):
    vals = (0, 1, 2) if hazard else (0, 1)   # 0 blocks, 1 passable, 2 hazard (reset to start)
    return [(p, pas, g) for p in PERMS for pas in itertools.product(vals, repeat=K)
            for g in range(10, 10 + T)]


def is_hazard(grid, pos, act, h):
    perm, pas, goal = h
    dr, dc = DIRS[perm[act]]
    r, c = pos[0] + dr, pos[1] + dc
    return 0 <= r < N and 0 <= c < N and 1 <= grid[r][c] <= len(pas) and pas[grid[r][c] - 1] == 2


def step(grid, pos, act, h):
    global EVALS
    EVALS += 1
    perm, pas, goal = h
    dr, dc = DIRS[perm[act]]
    r, c = pos[0] + dr, pos[1] + dc
    if not (0 <= r < N and 0 <= c < N):
        return pos, False
    cell = grid[r][c]
    if 1 <= cell <= len(pas) and pas[cell - 1] == 0:
        return pos, False
    if 1 <= cell <= len(pas) and pas[cell - 1] == 2:
        return START, False
    return (r, c), cell == goal


def plan(grid, pos, h):
    q, seen = deque([(pos, [])]), {pos}
    while q:
        p, path = q.popleft()
        for a in range(4):
            np_, win = step(grid, p, a, h)
            if win:
                return path + [a]
            if np_ not in seen:
                seen.add(np_)
                q.append((np_, path + [a]))
    return None


def make_game(rng, K, T, H):
    global START
    while True:
        u = rng.random((N, N))
        grid = np.zeros((N, N), dtype=int)
        for k in range(K):
            grid[(u >= 0.30 * k / K) & (u < 0.30 * (k + 1) / K)] = k + 1
        cells = [(r, c) for r in range(N) for c in range(N)]
        rng.shuffle(cells)
        start = cells[0]
        START = start
        grid[start] = 0
        for t in range(T):
            grid[cells[1 + t]] = 10 + t
        g = grid.tolist()
        true = H[rng.integers(len(H))]
        if plan(g, start, true) is not None:
            return g, start, true


def entropy_at(H, grid, pos, act):
    outs = {}
    for h in H:
        o = step(grid, pos, act, h)
        outs[o] = outs.get(o, 0) + 1
    p = np.array(list(outs.values()), float) / len(H)
    return float(-(p * np.log2(p)).sum())


def robust_plan(grid, pos, H, rng, sample, rate=False):
    S = H if sample is None or len(H) <= sample else [H[i] for i in rng.choice(len(H), sample, replace=False)]
    plans = {}
    for h in S:
        pl = plan(grid, pos, h)
        if pl is not None:
            plans.setdefault(tuple(pl), h)
    best, best_score = None, None
    for pl, h in plans.items():
        wins = 0
        for g in S:
            p, w = pos, False
            for a in pl:
                p, w = step(grid, p, a, g)
                if w:
                    break
            wins += w
        score = (wins / len(S) / len(pl), 0) if rate else (wins / len(S), -len(pl))
        if best_score is None or score > best_score:
            best, best_score = (list(pl), h), score
    return best


def unknown_colors(H, K, T):
    """Object types whose effect the consistent hypotheses still disagree on (EMPA-style)."""
    unk = set()
    for k in range(K):
        if len({h[1][k] for h in H}) > 1:
            unk.add(k + 1)
    goals = {h[2] for h in H}
    if len(goals) > 1:
        unk |= goals
    return unk


def contact_plan(grid, pos, h, unk):
    """Shortest path (under h) whose last action moves into a cell of an unknown type."""
    q, seen = deque([(pos, [])]), {pos}
    while q:
        p, path = q.popleft()
        for a in range(4):
            dr, dc = DIRS[h[0][a]]
            r, c = p[0] + dr, p[1] + dc
            if 0 <= r < N and 0 <= c < N and grid[r][c] in unk:
                return path + [a]
            np_, win = step(grid, p, a, h)
            if not win and np_ not in seen:
                seen.add(np_)
                q.append((np_, path + [a]))
    return None


def run(policy, grid, start, true, H0, rng):
    global START
    START = start
    pos, H, n, queue, chosen = start, list(H0), 0, [], None
    while n < CAP:
        pred = None
        if policy == "random":
            a = int(rng.integers(4))
        elif policy == "curiosity":
            ents = [entropy_at(H, grid, pos, x) for x in range(4)]
            if max(ents) > 0:
                a = int(np.argmax(ents))
            else:
                h = H[0]
                target, q, seen = None, deque([(pos, [])]), {pos}
                while q and target is None:
                    p, path = q.popleft()
                    for b in range(4):
                        np_, w = step(grid, p, b, h)
                        if np_ in seen or w:
                            continue
                        seen.add(np_)
                        if max(entropy_at(H, grid, np_, x) for x in range(4)) > 0:
                            target = path + [b]
                            break
                        q.append((np_, path + [b]))
                pl = target or plan(grid, pos, h) or [int(rng.integers(4))]
                a = pl[0]
        else:
            if not queue:
                if policy == "single":
                    chosen = H[rng.integers(len(H))]
                    queue = plan(grid, pos, chosen) or [int(rng.integers(4))]
                elif policy == "worldcoder":
                    # one program, chosen to be consistent AND optimistic: the shortest route to a win
                    best = None
                    for h in H:
                        pl = plan(grid, pos, h)
                        if pl is not None and (best is None or len(pl) < len(best[0])):
                            best = (pl, h)
                    queue, chosen = best if best else ([int(rng.integers(4))], H[0])
                elif policy == "empa":
                    # explore toward contact with object types of unknown effect, then exploit
                    chosen = H[rng.integers(len(H))]
                    ents = [entropy_at(H, grid, pos, x) for x in range(4)] if len({h[0] for h in H}) > 1 else [0]
                    if max(ents) > 0:
                        # first learn how its own avatar moves: most informative single action
                        queue = [int(np.argmax(ents))]
                    else:
                        K = max(len(h[1]) for h in H)
                        unk = unknown_colors(H, K, 0)
                        pl = contact_plan(grid, pos, chosen, unk) if unk else None
                        queue = pl or plan(grid, pos, chosen) or [int(rng.integers(4))]
                else:
                    sample = None if policy.endswith("-full") else int(policy.split("-s")[1])
                    best = robust_plan(grid, pos, H, rng, sample,
                                       rate=policy.startswith("rate") or policy.startswith("tau"))
                    queue, chosen = best if best else ([int(rng.integers(4))], H[0])
            a = queue[0]
            experiment = False
            if policy.startswith("tau"):
                # tau-based plan-agreement stopping (Formal model): if less than tau of the sampled
                # posterior agrees with the plan's predicted trajectory AND the rivals disagree on
                # the very next step, take the most informative action there instead, then replan.
                tau = int(policy[3:].split("-s")[0]) / 100
                ns = int(policy.split("-s")[1])
                S = H if len(H) <= ns else [H[i] for i in rng.choice(len(H), ns, replace=False)]

                def traj(h):
                    p, out = pos, []
                    for b in queue:
                        o = step(grid, p, b, h)
                        out.append(o)
                        p = o[0]
                        if o[1]:
                            break
                    return out
                ref = traj(chosen)
                agree = sum(traj(g) == ref for g in S) / len(S)
                if agree < tau and entropy_at(S, grid, pos, a) > 0:
                    ents = [entropy_at(S, grid, pos, x) for x in range(4)]
                    a = int(np.argmax(ents))
                    experiment = True
            pred = step(grid, pos, a, chosen)
        obs = step(grid, pos, a, true)
        n += 1 + (HAZARD_COST if is_hazard(grid, pos, a, true) else 0)
        H = [h for h in H if step(grid, pos, a, h) == obs] or H
        if pred is not None:
            if obs != pred or experiment:
                queue = []
            else:
                queue.pop(0)
        pos, win = obs
        if win:
            return n, True
    return n, False


def main(seed, n_games, K, T, policies, hazard=False):
    global EVALS, START
    H0 = hyps(K, T, hazard)
    rng = np.random.default_rng(seed)
    games = [make_game(rng, K, T, H0) for _ in range(n_games)]
    res = {"H": len(H0)}
    for pol in policies:
        out = []
        for i, (g, s, t) in enumerate(games):
            EVALS = 0
            n, ok = run(pol, g, s, t, H0, np.random.default_rng(seed * 1000 + i))
            ev = EVALS
            START = s
            out.append((n, ok, ev, len(plan(g, s, t))))
        res[pol] = out
    json.dump(res, sys.stdout)


if __name__ == "__main__":
    main(int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), sys.argv[5].split(","),
         hazard=len(sys.argv) > 6 and sys.argv[6] == "hazard")
