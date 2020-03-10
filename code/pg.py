"""
Reinforcement learning via policy gradients.

Code inspired by
    A. Karpathy
    Deep reinforcement learning: Pong from pixels
    https://karpathy.github.io/2016/05/31/rl
    https://gist.github.com/karpathy/a4166c7fe253700972fcbc77e4ea32c5
"""

from __future__ import print_function
import numpy as np
import optim


def discounted_rewards(gamma, r):
    """
    Compute cumulative discounted rewards
    `G_t = sum_{t'=t}^\infty \gamma^{t'-t} r_{t'}`
    for all time steps `t`.

    Args:
        gamma:  discount factor
        r:      array of rewards: r[t] is reward at time step t

    Returns:
        cumulative discounted rewards for all time steps
    """
    g = np.zeros_like(r)
    x = 0
    for t in reversed(range(len(r))):
        x = gamma*x + r[t]
        g[t] = x
    return g


def policy_gradient_iteration(model, env, gamma, eta=1e-3, nepisodes=100000):
    """
    Policy gradient iteration.

    Args:
        model:     to-be optimized policy model (typically a neural network)
        env:       environment
        gamma:     discount factor
        eta:       learning rate
        nepisodes: number of episodes (to-be played games)

    Returns:
        trained policy model
    """
    # every how many episodes to perform a parameter update
    batch_size = 10

    # configurations for Adam update rule
    optim_configs = { k: { 'learning_rate': eta } for k in model.params }

    running_mean = None

    for i in range(nepisodes // batch_size):

        # record inputs, taken actions and normalized discounted rewards
        # for all time steps (required for training)
        xs = []
        ys = []
        Gs = []

        for j in range(batch_size):

            # episode counter, starting from 1
            episode = i*batch_size + j + 1

            # whether to show current episode (render all time steps)
            show_episode = episode % (nepisodes//10) == 0
            if show_episode:
                print('frames of episode {}:'.format(episode))

            # play a game
            # initial (uniformly random) state (except "game over" state)
            s = np.random.randint(env.num_states - 1)
            r = env.rewards[s]
            game_over = False
            rs = []
            while not game_over:
                # evaluate the policy model to get action probability distribution
                # (formally minibatch of size N = 1)
                x = env.observation(s).reshape(-1)
                aprob = model.evaluate(x[None, :])[0]

                # sample an action from the returned probability
                a = np.random.choice(env.num_actions, p=aprob)

                if show_episode:
                    print('step', len(rs))
                    print('reward:', r)
                    print(env.render(s))
                    print('action:', env.action_strings[a])
                    print(13 * '_')

                # record current time step
                xs.append(x)    # state image
                ys.append(a)    # taken action
                rs.append(r)    # reward

                # transition to next state
                (snext, rnext, game_over) = env.step(s, a)
                s = snext
                r = rnext

            # reward for final "game over" state (should be zero)
            rs.append(r)

            if show_episode:
                # show final "game over" frame
                print(env.render(s))

            # compute the cumulative discounted rewards ("return") backwards through time
            G = discounted_rewards(gamma, np.array(rs))
            # cumulative discounted reward of initial state
            G0 = G[0]
            # shift discounted rewards by one time step such that feedback to policy
            # model works correctly (want rewards _after_ taking action 'a' as feedback)
            G = G[1:]
            # normalize the cumulative discounted rewards to improve gradient estimation
            G -= np.mean(G)
            if len(G) > 1:
                G /= np.std(G)
            # concatenate to form "advantage" vector
            Gs += G.tolist()

            running_mean = G0 if running_mean is None else 0.999*running_mean + 0.001*G0

            if episode % (nepisodes//100) == 0:
                print('episode {} completed, nsteps: {}, total discounted reward: {:g}, running mean: {:g}'.format(episode, len(rs)-1, G0, running_mean))

        # compute gradients modulated by "advantage" (normalized cumulative discounted rewards)
        _, grads = model.loss(np.vstack(xs), np.array(ys, dtype=int), np.array(Gs))

        # parameter update using Adam optimizer
        for k in model.params:
            model.params[k], optim_configs[k] = optim.adam(model.params[k], grads[k], optim_configs[k])

    return model
