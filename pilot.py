"""Pilot: does goal-directed hypothesis testing save actions in unknown grid games?

Each game: 8x8 grid. Hidden rules = (key->direction permutation, whether obstacle
colours A and B are passable, which of three target colours wins). 24*4*3 = 288
hypotheses. The agent sees only its position and a win flag after each action.

Policies
  random      : uniform random actions.
  single      : sample ONE consistent hypothesis, plan to its goal, replan on surprise.
  curiosity   : max predictive-entropy exploration until no reachable state is
                informative, then plan (full identification first).
  ours        : posterior-robust planning: among plans of consistent hypotheses,
                execute the one with the highest posterior P(win | plan); replan on surprise.
"""
import itertools, sys, json, random
from collections import deque
import numpy as np

N = 8
DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
PERMS = list(itertools.permutations(range(4)))
HYPS = [(p, a, b, g) for p in PERMS for a in (0, 1) for b in (0, 1) for g in (3, 4, 5)]
CAP = 300
EVALS = 0  # hypothesis-step simulations (compute proxy)


def step(grid, pos, act, h):
    global EVALS
    EVALS += 1
    perm, pa, pb, goal = h
    dr, dc = DIRS[perm[act]]
    r, c = pos[0] + dr, pos[1] + dc
    if not (0 <= r < N and 0 <= c < N):
        return pos, False
    cell = grid[r, c]
    if (cell == 1 and not pa) or (cell == 2 and not pb):
        return pos, False
    return (r, c), cell == goal


def plan(grid, pos, h):
    """BFS shortest action sequence to win under h; None if impossible."""
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


def make_game(rng):
    while True:
        grid = np.zeros((N, N), dtype=int)
        u = rng.random((N, N))
        grid[u < 0.15] = 1
        grid[(u >= 0.15) & (u < 0.30)] = 2
        cells = [(r, c) for r in range(N) for c in range(N)]
        rng.shuffle(cells)
        start = cells[0]
        grid[start] = 0
        for k, col in enumerate((3, 4, 5)):
            grid[cells[1 + k]] = col
        true = HYPS[rng.integers(len(HYPS))]
        if plan(grid, start, true) is not None:
            return grid, start, true


def consistent(H, grid, pos, act, obs):
    return [h for h in H if step(grid, pos, act, h) == obs]


def entropy_at(H, grid, pos, act):
    outs = {}
    for h in H:
        o = step(grid, pos, act, h)
        outs[o] = outs.get(o, 0) + 1
    p = np.array(list(outs.values()), float) / len(H)
    return float(-(p * np.log2(p)).sum())


def run(policy, grid, start, true, rng):
    pos, H, n = start, list(HYPS), 0
    queue = []  # (action, predicted obs)
    while n < CAP:
        if policy == "random":
            a = int(rng.integers(4))
        elif policy in ("single", "ours"):
            if not queue:
                if policy == "single":
                    h = H[rng.integers(len(H))]
                    pl = plan(grid, pos, h) or [int(rng.integers(4))]
                    queue = [(a, None) for a in pl]
                    chosen = h
                else:
                    best, best_score = None, -1
                    plans = {}
                    for h in H:
                        pl = plan(grid, pos, h)
                        if pl is not None:
                            plans.setdefault(tuple(pl), h)
                    for pl, h in plans.items():
                        wins = 0
                        for g in H:
                            p = pos
                            for a in pl:
                                p, w = step(grid, p, a, g)
                                if w:
                                    break
                            wins += w
                        score = (wins / len(H), -len(pl))
                        if score > (best_score if best else (-1, 0)):
                            best, best_score = (pl, h), score
                    if best is None:
                        best = ([int(rng.integers(4))], H[0])
                    queue = [(a, None) for a in best[0]]
                    chosen = best[1]
                # attach predictions lazily
            a = queue[0][0]
            pred = step(grid, pos, a, chosen)
        elif policy == "curiosity":
            ents = [entropy_at(H, grid, pos, a) for a in range(4)]
            if max(ents) > 0:
                a = int(np.argmax(ents))
            else:
                # walk (under a consistent hypothesis) to nearest informative state, else plan to goal
                h = H[0]
                target_path = None
                q, seen = deque([(pos, [])]), {pos}
                while q and target_path is None:
                    p, path = q.popleft()
                    for b in range(4):
                        np_, w = step(grid, p, b, h)
                        if np_ in seen or w:
                            continue
                        seen.add(np_)
                        if max(entropy_at(H, grid, np_, x) for x in range(4)) > 0:
                            target_path = path + [b]
                            break
                        q.append((np_, path + [b]))
                pl = target_path or plan(grid, pos, h) or [int(rng.integers(4))]
                a = pl[0]
        obs = step(grid, pos, a, true)
        n += 1
        H = consistent(H, grid, pos, a, obs) or H
        if policy in ("single", "ours"):
            if obs != pred:
                queue = []
            else:
                queue.pop(0)
        pos, win = obs
        if win:
            return n, True
    return n, False


def main(seed, n_games):
    global EVALS
    rng = np.random.default_rng(seed)
    games = [make_game(rng) for _ in range(n_games)]
    res = {}
    for pol in ("random", "single", "curiosity", "ours"):
        out = []
        for i, (g, s, t) in enumerate(games):
            EVALS = 0
            n, ok = run(pol, g, s, t, np.random.default_rng(seed * 1000 + i))
            ev = EVALS
            opt = len(plan(g, s, t))
            out.append((n, ok, ev, opt))
        res[pol] = out
    json.dump(res, sys.stdout)


if __name__ == "__main__":
    main(int(sys.argv[1]), int(sys.argv[2]))
