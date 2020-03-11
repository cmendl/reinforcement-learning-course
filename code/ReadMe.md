RL coding exercises
-------------------

### Requirements
Python 3 and NumPy

### Exercise (Markov decision processes, iterative algorithms)
1. Complete `mdp_template.py` (reference solution in `mdp.py`, but at least try to solve `mdp_template.py` first). You may test the implementation via `python test_mdp.py`.
2. Open and run the Jupyter notebook `run_iterative_plain_maze.ipynb`; re-run after changing the maze geometry in `maze_geometry.txt`.
3. Read `env.py` and learn how the maze geometry gets translated to a state space transition probability table (see `_get_maze_transition_probabilities`).
4. Open, read and run the Jupyter notebook `run_iterative_ghost_maze.ipynb`. Modify the rewards by editing the class `MazeGhostEnv` in `env.py`, or the ghost movement rules by editing `_get_ghost_maze_transition_probabilities` (e.g., allowing diagonal movement). Then re-run the Jupyter notebook `run_iterative_ghost_maze.ipynb`.

### Exercise (Classical Q-learning algorithm)
1. Complete the function definition `q_learn` within `run_qlearn_ghost_maze.ipynb`, and run the Jupyter notebook.
2. How sensitive are the results to changes of the learning rate `eta` (e.g., setting `eta = 0.1` or `eta = 0.001` instead of `eta = 0.01`)?

### Exercise (Policy gradients)
1. Complete the function `discounted_rewards` and familiarize yourself with `policy_gradient_iteration` within `pg_template.py` (reference solution in `pg.py`). How is the logarithm of the policy (appearing in the policy gradient theorem) implemented?
2. Open and run the notebooks `run_policy_gradients_plain_maze.ipynb` and `run_policy_gradients_ghost_maze.ipynb`.
3. Re-run the notebooks using the larger maze in `maze_geometry3.txt`, or invent your own geometry.
