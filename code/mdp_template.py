"""
Iterative algorithms and utility functions for Markov Decision Processes (MDPs).
"""

from __future__ import print_function
import numpy as np


def value_iter_step(tprob, rewards, gamma, u):
    """
    Value iteration step.

    Args:
        tprob:   tensor of rank 3 containing transition probability table:
                 tprob[snext, s, a] = P(snext | s, a) is probability of transition
                 to next state 'snext' given current state 's' and action 'a'
        rewards: vector of rewards for each state
        gamma:   discount factor
        u:       current vector of utilities (value function)

    Returns:
        numpy.ndarray: next vector of utilities (value function)
    """
    #
    # TODO: Implement a value iteration step.
    # (Hint: the function np.amax could be useful here.)
    #


def value_iteration(tprob, rewards, gamma, epsilon=1e-14, maxsteps=5000):
    """Value iteration algorithm."""
    u = np.zeros(tprob.shape[1])
    for i in range(maxsteps):
        unext = value_iter_step(tprob, rewards, gamma, u)
        diff = np.linalg.norm(unext - u)
        u = unext
        if diff <= epsilon*(1 - gamma)/gamma:
            print('value iteration with epsilon={} completed after {} iterations'.format(epsilon, i))
            break
    return u


def policy_from_utility(tprob, u):
    """
    Obtain policy based on utility function.

    Args:
        tprob: tensor of rank 3 containing transition probability table:
               tprob[snext, s, a] = P(snext | s, a) is probability of transition
               to next state 'snext' given current state 's' and action 'a'
        u:     current vector of utilities (value function)

    Returns:
        numpy.ndarray: policy vector
    """
    #
    # TODO: Compute policy given utility function 'u'.
    # (Hint: the function np.argmax could be useful here.)
    #


def utility_from_policy(tprob, rewards, gamma, pol):
    """
    Compute utility function based on given policy (requires gamma < 1).

    Args:
        tprob:   tensor of rank 3 containing transition probability table:
                 tprob[snext, s, a] = P(snext | s, a) is probability of transition
                 to next state 'snext' given current state 's' and action 'a'
        rewards: vector of rewards for each state
        gamma:   discount factor
        pol:     current policy vector, pol[s] contains action for state 's'

    Returns:
        numpy.ndarray: vector of utilities
    """
    nstates = tprob.shape[0]
    # transition probabilities for given policy 'pol' (transposed matrix)
    tprob_pol = np.array([tprob[:, s, pol[s]] for s in range(nstates)])
    return np.linalg.solve(np.identity(nstates) - gamma * tprob_pol, rewards)


def policy_iter_step(tprob, rewards, gamma, pol):
    """
    Policy iteration step.

    Args:
        tprob:   tensor of rank 3 containing transition probability table:
                 tprob[snext, s, a] = P(snext | s, a) is probability of transition
                 to next state 'snext' given current state 's' and action 'a'
        rewards: vector of rewards for each state
        gamma:   discount factor
        pol:     current policy vector, pol[s] contains action for state 's'

    Returns:
        numpy.ndarray: next policy vector
    """
    #
    # TODO: Implement a policy iteration step.
    #


def policy_iteration(tprob, rewards, gamma, maxsteps=100):
    """Policy iteration algorithm."""
    pol = np.zeros(tprob.shape[1], dtype=int)
    for i in range(maxsteps):
        pnext = policy_iter_step(tprob, rewards, gamma, pol)
        diff = np.linalg.norm(pnext - pol)
        if diff == 0:
            print('policy iteration completed after {} iterations'.format(i))
            break
        pol = pnext
    return pol


def utility_from_qvalue(Q):
    """
    Extract utility function from Q-value function.

    Args:
        Q: matrix storing Q-value function:
           Q[s, a] is action-value for state 's' and action 'a'

    Returns:
        numpy.ndarray: vector of utilities
    """
    return np.amax(Q, axis=1)


def policy_from_qvalue(Q):
    """
    Extract policy from Q-value function.

    Args:
        Q: matrix storing Q-value function:
           Q[s, a] is action-value for state 's' and action 'a'

    Returns:
        numpy.ndarray: policy vector
    """
    return np.argmax(Q, axis=1)


def q_iter_step(tprob, rewards, gamma, Q):
    """
    Q-value iteration step.

    Args:
        tprob:   tensor of rank 3 containing transition probability table:
                 tprob[snext, s, a] = P(snext | s, a) is probability of transition
                 to next state 'snext' given current state 's' and action 'a'
        rewards: vector of rewards for each state
        gamma:   discount factor
        Q:       matrix storing current Q-value function:
                 Q[s, a] is action-value for state 's' and action 'a'

    Returns:
        numpy.ndarray: next Q-value function
    """
    u = utility_from_qvalue(Q)
    return rewards[:, None] + gamma * np.tensordot(u, tprob, axes=1)


def q_iteration(tprob, rewards, gamma, epsilon=1e-14, maxsteps=5000):
    """Q-value iteration algorithm."""
    Q = np.zeros((tprob.shape[1], tprob.shape[2]))
    for i in range(maxsteps):
        Qnext = q_iter_step(tprob, rewards, gamma, Q)
        diff = np.linalg.norm(Qnext - Q)
        Q = Qnext
        if diff <= epsilon:
            print('Q-value iteration with epsilon={} completed after {} iterations'.format(epsilon, i))
            break
    return Q
