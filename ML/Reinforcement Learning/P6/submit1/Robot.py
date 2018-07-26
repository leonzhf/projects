import random
import math
class Robot(object):

    def __init__(self, maze, alpha=0.5, gamma=0.9, epsilon0=0.5):

        self.maze = maze
        self.valid_actions = self.maze.valid_actions
        self.state = None
        self.action = None

        # Set Parameters of the Learning Robot
        self.alpha = alpha
        self.gamma = gamma

        self.epsilon0 = epsilon0
        self.epsilon = epsilon0
        self.t = 0

        self.Qtable = {}
        self.epsilon_linear_decay = False
        self.reset()

    def reset(self):
        """
        Reset the robot
        """
        self.state = self.sense_state()
        self.create_Qtable_line(self.state)

    def set_status(self, learning=False, testing=False):
        """
        Determine whether the robot is learning its q table, or
        exceuting the testing procedure.
        """
        self.learning = learning
        self.testing = testing

    def set_epsilon_linear_decay(self, linear_decay):
        self.epsilon_linear_decay = linear_decay

    def update_parameter(self):
        """
        Some of the paramters of the q learning robot can be altered,
        update these parameters when necessary.
        """
        if self.testing:
            # TODO 1. No random choice when testing
            self.epsilon = 0
            pass
        else:
            # TODO 2. Update parameters when learning
            self.t += 1
            self.epsilon = self.epsilon0 / ( 1 + self.t) if self.epsilon_linear_decay else self.epsilon0 / ( 1 + math.sqrt(self.t))

        return self.epsilon

    def sense_state(self):
        """
        Get the current state of the robot. In this
        """

        # TODO 3. Return robot's current state
        return self.maze.sense_robot()

    def create_Qtable_line(self, state):
        """
        Create the qtable with the current state
        """
        # TODO 4. Create qtable with current state
        # Our qtable should be a two level dict,
        # Qtable[state] ={'u':xx, 'd':xx, ...}
        # If Qtable[state] already exits, then do
        # not change it.
        if(state not in self.Qtable):
            self.Qtable[state] = {'u':0, 'r':0, 'd':0, 'l':0}

    # leon
    # def epsilon_greedy_probs(self):
        
    #     q_s = self.Qtable[self.state]
    #     nA = len(self.valid_actions)

    #     action_max = max(q_s, key=q_s.get)
    #     max_index = None
    #     if 'u' == action_max:
    #         max_index = 0 
    #     elif 'r' == action_max:
    #         max_index = 1
    #     elif 'd' == action_max:
    #         max_index = 2
    #     else:
    #         max_index = 3

    #     prob = np.ones(nA) * ( self.epsilon / nA)
    #     prob[max_index] = 1 - epsilon + epsilon / nA
    #     return prob

    # # get epsilon-greedy action probabilities
    # policy_s = epsilon_greedy_probs(env, Q[next_state], i_episode, 0.1)
    # # pick next action A'
    # next_action = np.random.choice(np.arange(env.nA), p=policy_s)
 

    def choose_random_action(self):
        return random.choice(self.valid_actions)


    def choose_action_with_highest_q_value(self):
        s_a = self.Qtable[self.state]
        return max(s_a, key = s_a.get)

    def choose_action(self):
        """
        Return an action according to given rules
        """
        def is_random_exploration():

            # TODO 5. Return whether do random choice
            # hint: generate a random number, and compare
            # it with epsilon
            return random.random() <= self.epsilon

        if self.learning:
            if is_random_exploration():
                # TODO 6. Return random choose aciton
                return self.choose_random_action()
            else:
                # TODO 7. Return action with highest q value
                return self.choose_action_with_highest_q_value()
        elif self.testing:
            # TODO 7. choose action with highest q value
            return self.choose_action_with_highest_q_value()
    
        else:
            # TODO 6. Return random choose aciton
            return self.choose_random_action()
    
    def update_Qtable(self, r, action, next_state):
        """
        Update the qtable according to the given rule.
        """
        if self.learning:
            # TODO 8. When learning, update the q table according
            # to the given rules
            q_s_a = self.Qtable[self.state][action]
            q_snext = self.Qtable[next_state]
            self.Qtable[self.state][action] = (1 - self.alpha) * q_s_a + self.alpha * (r + self.gamma * max(q_snext.values()))

    def update(self):
        """
        Describle the procedure what to do when update the robot.
        Called every time in every epoch in training or testing.
        Return current action and reward.
        """
        self.state = self.sense_state() # Get the current state
        self.create_Qtable_line(self.state) # For the state, create q table line

        action = self.choose_action() # choose action for this state
        reward = self.maze.move_robot(action) # move robot for given action

        next_state = self.sense_state() # get next state
        self.create_Qtable_line(next_state) # create q table line for next state

        if self.learning and not self.testing:
            self.update_Qtable(reward, action, next_state) # update q table
            self.update_parameter() # update parameters

        return action, reward
