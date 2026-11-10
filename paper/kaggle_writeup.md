TITLE
When Mistakes Cost: Posterior Plan Selection for Program World-Model Agents

SUBTITLE
Keep rival world models and act on the plan most likely to win per action: it beats optimism when wrong moves are costly, at a fraction of the compute.

WRITEUP BODY (paste below this line)

Abstract

Interactive agents with executable world models face a trade-off: identifying an environment fully can waste actions, while acting on an optimistic model risks costly mistakes. We study an uncertainty-aware decision rule that maintains a posterior over competing world-model programs and executes the plan with the highest posterior probability of success per action, replanning when a prediction fails.

In controlled grid games with hidden rules (648 to 2,592 hypotheses in the costly-mistake experiments), and with claims specified before held-out evaluation, this rule requires 1.65 to 2.33 fewer actions than an optimistic single-program baseline when incorrect actions trigger costly resets. It also uses approximately 20 times fewer model simulations in the largest tested setting. A mechanism experiment shows the advantage grows with mistake cost, from no significant advantage at zero cost to 7.32 fewer actions at the highest cost tested. When mistakes are free, optimistic planning is equal or slightly better. Scoring plans against 16 sampled hypotheses matched full posterior scoring with 49 to 190 times fewer simulations. These findings motivate a decision rule that existing program-based agents could adopt, rather than a new world-model architecture.

Kavya Shree
Pandit Deendayal Energy University, Gujarat, India

Read the full paper here: https://github.com/kavya5cloud/arc/blob/main/paper/paper.pdf
Code and raw results (MIT license, one-command reproduction): https://github.com/kavya5cloud/arc
Linked code submission: [KAGGLE SUBMISSION LINK]
