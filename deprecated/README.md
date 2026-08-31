# Deprecated

`eval_loop.py`, `loop_diag.py` — a hand-rolled control loop that drove the policy
directly with `robot.send_action`, bypassing lerobot's rollout pipeline. It issued
commands at 27 fps and the arm never moved, while the identical checkpoint driven
through `lerobot-rollout` did move.

Rather than debug a reimplementation of a path that already works, use the official
one: `lerobot-rollout --strategy.type=episodic` runs N attempts with reset phases
and a return to the start pose between them. `robot.ps1 eval -Trials N` wraps it.

The lesson is the general one: when a proven path exists, reuse it. Debugging your
own copy of it is work that teaches nothing about the actual problem.
