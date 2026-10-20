from inspect_robots import eval
from inspect_robots.mock import CubePickEmbodiment, ScriptedPolicy, RandomPolicy
from inspect_robots.scene import Scene
from inspect_robots.scorer import success_at_end
from inspect_robots.task import Task

task = Task(
    name="cubepick-random",
    scenes=[Scene(id=f"layout-{i}", instruction="reach the cube", init_seed=i) for i in range(5)],
    scorer=success_at_end(),
    max_steps=80,
)

# The two inputs that can be swapped are the policy (VLA) and an embodiment (robot or sim)
(log,) = eval(task, RandomPolicy(), CubePickEmbodiment())
print(log.status, log.results.metrics)  # Result metric is what the scorer deems success, or the {'success_at_end': 1.0} that we added
