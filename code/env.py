# -*- coding: utf-8 -*-

from __future__ import print_function
import numpy as np
from util import parse_maze_file


class Env(object):
    """
    Abstract "environment" class, inspired by OpenAI Gym.
    """

    def step(self, s, a):
        """Run one timestep of the environment's dynamics."""
        raise NotImplementedError

    def reset(self):
        """Sample an initial state."""
        raise NotImplementedError

    def render(self, s):
        """Render state 's'."""
        raise NotImplementedError


class DiscreteEnv(Env):
    """
    Discrete environment based on transition probability table
    tprob of shape (num_states, num_states, num_actions):
    tprob[snext, s, a] = P(snext | s, a) is probability of transition
    to next state 'snext' given current state 's' and action 'a'.
    """

    def __init__(self, tprob, isd, rewards):
        """
        Args:
            tprob:   transition probability table
            isd:     initial state distribution, vector of length num_states
            rewards: reward of each state, vector of length num_states
        """
        # dimension consistency checks
        assert tprob.shape[0] == tprob.shape[1]
        assert len(isd) == tprob.shape[0]
        self.tprob = tprob
        self.isd = isd
        self.rewards = rewards

    @property
    def num_states(self):
        return self.tprob.shape[0]

    @property
    def num_actions(self):
        return self.tprob.shape[2]

    def step(self, s, a):
        # transition to next state
        snext = np.random.choice(self.num_states, p=self.tprob[:, s, a])
        rnext = self.rewards[snext]
        # check whether game is over (assuming last state is "game over" state)
        game_over = (snext == self.num_states - 1)
        return (snext, rnext, game_over)

    def reset(self):
        # sample initial state
        return np.random.choice(self.num_states, p=self.isd)


def _get_maze_transition_probabilities(locs, movedir, exits):
    """
    Construct transition probability table for a 2D maze.

    Args:
        locs:    maze field locations (list of coordinates)
        movedir: movement directions
        exits:   exit coordinates

    Returns:
        numpy.ndarray: tprob[snext, s, a] is probability of transition to next
                       state 'snext' given current state 's' and action 'a'
    """
    num_states  = len(locs) + 1     # including final "game over" state
    num_actions = len(movedir)

    p = np.zeros((num_states, num_states, num_actions))
    for s, c in enumerate(locs):
        if c in exits:
            # transition to "game over" state (assumed to be last state)
            p[-1, s, :] = 1
        else:
            for a in range(num_actions):
                for j in [-1, 0, 1]:
                    # next location
                    cnext = [c[0] + movedir[(a + j) % num_actions][0],
                             c[1] + movedir[(a + j) % num_actions][1]]
                    if cnext in locs:
                        snext = locs.index(cnext)
                    else:
                        # remain at current field
                        snext = s
                    # update probability table
                    p[snext, s, a] += (0.8 if j == 0 else 0.1)
    # "game over" state transitions to itself
    p[-1, -1, :] = 1

    return p


class MazeEnv(DiscreteEnv):
    """
    2D maze environment.
    """
    def __init__(self, filename, r):
        """
        Construct maze environment.
        """
        width, height, locs, start, exitsP, exitsN = parse_maze_file(filename)
        # movement directions; must be circularly ordered
        movedir = [
            [ 1,  0],
            [ 0,  1],
            [-1,  0],
            [ 0, -1]]
        # concatenate exit coordinates
        exits = exitsP + exitsN
        tprob = _get_maze_transition_probabilities(locs, movedir, exits)
        # consistency check
        err = 0
        for s in range(tprob.shape[1]):
            for a in range(4):
                err += abs(np.sum(tprob[:, s, a]) - 1)
        assert err < 1e-14*tprob.shape[1], 'transition probability table check failed, error: {}'.format(err)
        # rewards
        rewards = np.zeros(tprob.shape[1])
        for i, c in enumerate(locs):
            if c in exitsP:
                rewards[i] = 1
            elif c in exitsN:
                rewards[i] = -1
            else:
                rewards[i] = r
        # initial state distribution is delta function corresponding to 'start' field
        isd = np.zeros(tprob.shape[1])
        isd[locs.index(start)] = 1
        # init parent class
        super(MazeEnv, self).__init__(tprob, isd, rewards)
        self.width = width
        self.height = height
        self.locs = locs
        self.exitsP = exitsP
        self.exitsN = exitsN
        # mask for policy comparisons (value at exit fields and game over state not relevant)
        self.pmask = np.ones(tprob.shape[1], dtype=int)
        self.pmask[-1] = 0
        for c in exits:
            self.pmask[locs.index(c)] = 0

    @property
    def action_strings(self):
        return [u'→', u'↑', u'←', u'↓']

    def maze_array(self, x):
        """
        Arrange the values in 'x' into an array with the geometric layout of the maze.
        """
        # inaccessible locations marked by 'NaN'
        a = np.nan * np.ones((self.height, self.width))
        for i, c in enumerate(self.locs):
            a[c[1], c[0]] = x[i]
        # field (0, 0) at lower left corner
        return np.flipud(a)

    def draw_policy(self, pol):
        """
        Draw policy (as string).
        """
        a = np.zeros((self.height, self.width), dtype=np.unicode_)
        # mark all locations as inaccessible first
        a[:, :] = u'█'
        # policy arrows
        for i, c in enumerate(self.locs):
            a[c[1], c[0]] = self.action_strings[pol[i]]
        # mark exits (policy not relevant there)
        for c in self.exitsP:
            a[c[1], c[0]] = 'E'
        for c in self.exitsN:
            a[c[1], c[0]] = 'F'
        a = [' '.join(row) for row in a]
        a = '\n'.join(reversed(a))
        return a

    def observation(self, s):
        """
        Compute an "observation" matrix corresponding to state 's'.
        """
        obs = np.zeros((self.height, self.width))
        # mark all locations as inaccessible first
        obs[:, :] = -0.1
        # regular fields
        for c in self.locs:
            obs[c[1], c[0]] = 0
        # mark exits
        for c in self.exitsP:
            obs[c[1], c[0]] = 0.5
        for c in self.exitsN:
            obs[c[1], c[0]] = -0.5
        if s < self.num_states - 1: # if not game over yet
            # player character
            c = self.locs[s]
            obs[c[1], c[0]] = 1
        return obs

    def render(self, s):
        """
        Render state 's' (as string).
        """
        if s < self.num_states - 1:
            a = np.zeros((self.height, self.width), dtype=np.unicode_)
            # mark all locations as inaccessible first
            a[:, :] = u'█'
            # regular fields
            for c in self.locs:
                a[c[1], c[0]] = u'░'
            # mark exits
            for c in self.exitsP:
                a[c[1], c[0]] = 'E'
            for c in self.exitsN:
                a[c[1], c[0]] = 'F'
            # player character
            c = self.locs[s]
            a[c[1], c[0]] = u'☺'
            a = [' '.join(row) for row in a]
            a = '\n'.join(reversed(a))
            return a
        else:
            return 'Game over!'

    def play(self, pol, gamma, maxsteps=50):
        """
        Play a game using policy 'pol'.
        """
        s = self.reset()
        r = self.rewards[s]
        u = 0
        for n in range(maxsteps):
            print('step', n)
            print('reward:', r)
            u += gamma**n * r
            # draw maze and player
            print(self.render(s))
            # choose action according to policy
            a = pol[s]
            print('action:', self.action_strings[a])
            print((2*self.width+5) * '_')
            # transition to next state
            (s, r, game_over) = self.step(s, a)
            # "game over" state reached?
            if game_over:
                print(self.render(s))
                break
        if not game_over:
            print('maximum number of steps reached...')
        print('cumulative discounted reward (gamma = {}): {}'.format(gamma, u))


def _get_ghost_maze_transition_probabilities(locs, movedir, exits):
    """
    Construct transition probability table for a 2D maze with a ghost.

    States are enumerated as `ig*len(locs) + ip` with ip the index of the player
    field and ig the index of the ghost field.

    The game is over if the player reaches an exit field or is captured by the
    ghost (on the same field as ghost).

    The ghost moves randomly, with probability 0.4 towards the player and
    with probability 0.2 in one of the other three directions.

    Args:
        locs:    maze field locations (array of coordinates)
        movedir: movement directions
        exits:   exit coordinates

    Returns:
        numpy.ndarray: tprob[snext, s, a] is probability of transition to next
                       state 'snext' given current state 's' and action 'a'
    """
    num_states  = len(locs)**2 + 1  # including final "game over" state
    num_actions = len(movedir)

    p = np.zeros((num_states, num_states, num_actions))
    for ig, cg in enumerate(locs):  # ghost
        for ip, cp in enumerate(locs):  # player
            # state index (all combinations of player and ghost positions)
            s = ig*len(locs) + ip
            if cp in exits or ip == ig:
                # transition to "game over" state (assumed to be last state)
                # if player on exit field or at same location as ghost
                p[-1, s, :] = 1
            else:
                # ghost movement
                # direction from ghost towards player
                pdir = np.array(cp, dtype=float) - np.array(cg, dtype=float)
                pdir /= np.linalg.norm(pdir)
                # index of preferred movement direction towards player
                kpref = np.argmin(np.linalg.norm(np.array(movedir) - pdir, axis=1))
                for k in range(len(movedir)):
                    cgnext = [cg[0] + movedir[k][0],
                              cg[1] + movedir[k][1]]
                    if cgnext in locs:
                        ignext = locs.index(cgnext)
                    else:
                        # remain at current field
                        ignext = ig
                    # player actions
                    for a in range(num_actions):
                        for j in [-1, 0, 1]:
                            cpnext = [cp[0] + movedir[(a + j) % num_actions][0],
                                      cp[1] + movedir[(a + j) % num_actions][1]]
                            if cpnext in locs:
                                ipnext = locs.index(cpnext)
                            else:
                                # remain at current field
                                ipnext = ip
                            # next state index
                            snext = ignext*len(locs) + ipnext
                            # update probability table;
                            # ghost has higher probability to move towards player
                            p[snext, s, a] += (0.4 if k == kpref else 0.2) * (0.8 if j == 0 else 0.1)
    # "game over" state transitions to itself
    p[-1, -1, :] = 1

    return p


class MazeGhostEnv(DiscreteEnv):
    """
    2D maze environment with a ghost (simplified version of Pac-Man).
    """
    def __init__(self, filename, r):
        """
        Construct maze environment.
        """
        width, height, locs, start, exitsP, exitsN = parse_maze_file(filename)
        # movement directions; must be circularly ordered
        movedir = [
            [ 1,  0],
            [ 0,  1],
            [-1,  0],
            [ 0, -1]]
        # concatenate exit coordinates
        exits = exitsP + exitsN
        tprob = _get_ghost_maze_transition_probabilities(locs, movedir, exits)
        # consistency check
        err = 0
        for s in range(tprob.shape[1]):
            for a in range(4):
                err += abs(np.sum(tprob[:, s, a]) - 1)
        assert err < 1e-14*tprob.shape[1], 'transition probability table check failed, error: {}'.format(err)
        # rewards
        rewards = np.zeros(tprob.shape[1])
        for ig, cg in enumerate(locs):
            for ip, cp in enumerate(locs):
                # state index
                s = ig*len(locs) + ip
                if cp in exitsP:
                    rewards[s] = 1
                elif cp in exitsN:
                    rewards[s] = -1
                elif ip == ig:
                    # player and ghost at same location
                    rewards[s] = -2
                else:
                    rewards[s] = r
        # initial state distribution
        isd = np.zeros(tprob.shape[1])
        ghost_start = []
        for c in locs:
            # ghost must initially have a minimum distance from player
            if max(abs(c[0] - start[0]), abs(c[1] - start[1])) > 1:
                ghost_start.append(c)
        for cg in ghost_start:
            ig = locs.index(cg)
            # state index
            s = ig*len(locs) + locs.index(start)
            isd[s] = 1.0 / len(ghost_start)
        # consistency check
        assert abs(np.sum(isd) - 1) < 1e-15*tprob.shape[1]
        # init parent class
        super(MazeGhostEnv, self).__init__(tprob, isd, rewards)
        self.width = width
        self.height = height
        self.locs = locs
        self.exitsP = exitsP
        self.exitsN = exitsN
        # mask for policy comparisons (value directly preceding "game over" state not relevant)
        self.pmask = np.ones(tprob.shape[1], dtype=int)
        self.pmask[-1] = 0
        for ig, cg in enumerate(locs):
            for ip, cp in enumerate(locs):
                # state index
                s = ig*len(locs) + ip
                if cp in exits or ip == ig:
                    # player at exit field or player and ghost at same location
                    self.pmask[s] = 0

    @property
    def action_strings(self):
        return [u'→', u'↑', u'←', u'↓']

    def maze_array(self, x):
        """
        Arrange the values in 'x' into an array with the geometric layout of the maze.
        """
        # inaccessible locations marked by 'NaN'
        a = np.nan * np.ones((self.height, self.width))
        for i, c in enumerate(self.locs):
            a[c[1], c[0]] = x[i]
        # field (0, 0) at lower left corner
        return np.flipud(a)

    def draw_policy(self, pol):
        """
        Draw policy (as string).
        """
        ps = ''
        for ig, cg in enumerate(self.locs):     # iterate over ghost positions
            # policy for current ghost position
            a = np.zeros((self.height, self.width), dtype=np.unicode_)
            # mark all locations as inaccessible first
            a[:, :] = u'█'
            # policy arrows
            for ip, cp in enumerate(self.locs):
                # state index
                s = ig*len(self.locs) + ip
                a[cp[1], cp[0]] = self.action_strings[pol[s]]
            # mark exits (policy not relevant there)
            for c in self.exitsP:
                a[c[1], c[0]] = 'E'
            for c in self.exitsN:
                a[c[1], c[0]] = 'F'
            # ghost character
            a[cg[1], cg[0]] = 'G'
            a = [' '.join(row) for row in a]
            a = '\n'.join(reversed(a))
            ps = '\n\n'.join([ps, a])
        return ps[2:]

    def observation(self, s):
        """
        Compute an "observation" matrix corresponding to state 's'.
        """
        obs = np.zeros((self.height, self.width))
        # mark all locations as inaccessible first
        obs[:, :] = -0.1
        # regular fields
        for c in self.locs:
            obs[c[1], c[0]] = 0
        # mark exits
        for c in self.exitsP:
            obs[c[1], c[0]] = 0.5
        for c in self.exitsN:
            obs[c[1], c[0]] = -0.5
        if s < self.num_states - 1: # if not game over yet
            # player and ghost location index
            ip = s %  len(self.locs)
            ig = s // len(self.locs)
            # player character
            cp = self.locs[ip]
            obs[cp[1], cp[0]] = 1
            # ghost character
            cg = self.locs[ig]
            obs[cg[1], cg[0]] = -1
        return obs

    def render(self, s):
        """
        Render state 's' (as string).
        """
        if s < self.num_states - 1:
            a = np.zeros((self.height, self.width), dtype=np.unicode_)
            # mark all locations as inaccessible first
            a[:, :] = u'█'
            # regular fields
            for c in self.locs:
                a[c[1], c[0]] = u'░'
            # mark exits
            for c in self.exitsP:
                a[c[1], c[0]] = 'E'
            for c in self.exitsN:
                a[c[1], c[0]] = 'F'
            # player and ghost location index
            ip = s %  len(self.locs)
            ig = s // len(self.locs)
            # player character
            cp = self.locs[ip]
            a[cp[1], cp[0]] = u'☺'
            # ghost character
            cg = self.locs[ig]
            a[cg[1], cg[0]] = 'G'
            a = [' '.join(row) for row in a]
            a = '\n'.join(reversed(a))
            return a
        else:
            return 'Game over!'

    def play(self, pol, gamma, maxsteps=50):
        """
        Play a game using policy 'pol'.
        """
        s = self.reset()
        r = self.rewards[s]
        u = 0
        for n in range(maxsteps):
            print('step', n)
            print('reward:', r)
            u += gamma**n * r
            # draw maze, player and ghost
            print(self.render(s))
            # choose action according to policy
            a = pol[s]
            print('action:', self.action_strings[a])
            print((2*self.width+5) * '_')
            # transition to next state
            (s, r, game_over) = self.step(s, a)
            # "game over" state reached?
            if game_over:
                print(self.render(s))
                break
        if not game_over:
            print('maximum number of steps reached...')
        print('cumulative discounted reward (gamma = {}): {}'.format(gamma, u))
