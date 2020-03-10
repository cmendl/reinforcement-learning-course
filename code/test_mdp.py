"""
Unit tests for mdp module.
"""

import unittest
import numpy as np
import mdp


class TestMDP(unittest.TestCase):


    def test_value_iter_step(self):

        num_states = 7
        num_actions = 5

        # fictitious MDP model
        tprob_shape = (num_states, num_states, num_actions)
        tprob = np.sin(np.linspace(0, 10, num=np.prod(tprob_shape)))**2
        tprob = tprob.reshape(tprob_shape)
        # normalization
        for s in range(num_states):
            for a in range(num_actions):
                tprob[:, s, a] /= np.sum(tprob[:, s, a])
        rewards = np.linspace(-0.1, 0.4, num=num_states)
        gamma = 0.85
        u = np.linspace(0, 1, num=num_states)

        unext = mdp.value_iter_step(tprob, rewards, gamma, u)

        unext_ref = np.array([0.36953495, 0.43307525, 0.49157556, 0.55009479,
                              0.61368587, 0.68587015, 0.77361628])
        self.assertAlmostEqual(rel_error(unext, unext_ref), 0., delta=1e-7,
                               msg='"value_iter_step" check failed')


    def test_policy_from_utility(self):

        num_states = 11
        num_actions = 4

        # fictitious MDP model
        tprob_shape = (num_states, num_states, num_actions)
        tprob = np.mod(np.sin(np.linspace(10, 40, num=np.prod(tprob_shape)))**2, 0.0453)
        tprob = tprob.reshape(tprob_shape)
        # normalization
        for s in range(num_states):
            for a in range(num_actions):
                tprob[:, s, a] /= np.sum(tprob[:, s, a])
        u = np.cos(np.linspace(10, 50, num=num_states))

        pol = mdp.policy_from_utility(tprob, u)

        pol_ref = np.array([0, 0, 2, 0, 3, 3, 0, 2, 3, 0, 1])
        self.assertEqual(np.linalg.norm(pol - pol_ref), 0,
                         msg='"policy_from_utility" check failed')


    def test_policy_iter_step(self):

        num_states = 12
        num_actions = 3

        # fictitious MDP model
        tprob_shape = (num_states, num_states, num_actions)
        tprob = np.mod(np.sin(np.linspace(20, 50, num=np.prod(tprob_shape)))**2, 0.0312)
        tprob = tprob.reshape(tprob_shape)
        # normalization
        for s in range(num_states):
            for a in range(num_actions):
                tprob[:, s, a] /= np.sum(tprob[:, s, a])
        rewards = np.linspace(-0.1, 0.4, num=num_states)
        gamma = 0.9
        pol = np.array([0, 2, 0, 1, 0, 1, 2, 0, 1, 0, 1, 1])

        pnext = mdp.policy_iter_step(tprob, rewards, gamma, pol)

        pnext_ref = np.array([0, 0, 2, 2, 1, 0, 2, 0, 0, 1, 0, 1])
        self.assertEqual(np.linalg.norm(pnext - pnext_ref), 0,
                         msg='"policy_iter_step" check failed')


def rel_error(x, y):
    """Compute relative errors."""
    return np.max(np.abs(x - y) / (np.maximum(np.abs(x) + np.abs(y), 1e-8)))


if __name__ == '__main__':
    unittest.main()
