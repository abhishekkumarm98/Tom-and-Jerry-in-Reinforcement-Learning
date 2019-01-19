# Tom-and-Jerry-in-Reinforcement-Learning
In a grid world environment, the agent (Tom) will find the shortest path to reach goal (Jerry).

Our objective is to teach the agent to navigate goal in an environment by applying reinforcement learning and deep learning(Deep Q network). 

States- 36 possible states  (0,0),(1,1),.......(5,5).
Actions- left, right, up and down.

reward= +1 ; If an agent,Tom moves closer to the goal,Jerry.
         0 ; If an agent,Tom does not move or remain in same position, due to restriction by grid boundary.
        -1 ; If an agent,Tom goes away from the goal,Jerry.
        
In the project, we have defined different classes and methods inside it. Each class has its own
functionality that will help to serve our motive. Classes are :

Environment:
First, we initialize our environment. The environment is almost similar to OpenAI's Gym
Environments, it has six methods: _update_state(), _get_reward(), _is_over(), step(), render()
and step().
        
Memory:
In the memory class, we are augmenting our dataset by storing an array consisting of current
State(s), action (a), immediate reward(r) and next state(s_). The sampled dataset is nothing but
the experiences what will be faced by the agent.

Agent:
When we initialize the agent, we must pass both a state_dim and action_dim into it's
constructor. These values tell the agent the dimensions of the input and the output of the
neural network. The agent has three main methods: act(), observe(), and replay().
The act() method takes the current state as input and returns the agent's selected action.
Within the method, we first check if we should choose a random action (in order to explore) by
comparing a randomly generated number to epsilon. If the agent decides to return a random
action, then it's simply a randomly selected action from the action space (in this case, an integer
in [0,3]). If the agent doesn't choose a random action, then the state is passed to the agent's
neural network (i.e., it's Brain object). This results in an array of expected discounted rewards
corresponding to each action.

The observe() method receives an observation tuple (s, a, r, s_) as input and it is saved to the
agent's Memory.

The replay() method is where any actual learning occurs. Up to this point, we haven't trained the
agent's neural network, only applied it to determine actions given states. In order to train it, we
implement Experience Replay, which, in short, allows the agent to not only learn from recent
observations but also previous observations. During experience replay, we randomly sample a set
of observations from the agent's Memory. These observations are then constructed in a way that
allows us to pass them through the neural network to train it.

First the agent will randomly select its action by a certain percentage, called ‘exploration
rate’ or ‘epsilon’. This is because, it is better for the agent to try all kinds of things before it
starts to see the patterns. When it is not deciding the action randomly, the agent will predict
the reward value based on the current state and pick the action that will give the highest
reward. We want our agent to decrease the number of random action, as it goes, so we
introduce an exponential-decay epsilon, that eventually will allow our agent to explore the
environment.

The Q- learning considers each action (a) in its current state and chooses action that maximizes
Q (s, a). Basically, finding Q function corresponds to learning the optimal policy (deterministic).

Conclusion:

During implementing this project, I realized deep concepts of reinforcement learning.
Like discounting factor(gamma) which tells about higher the discounting factor smaller the
discount which infers that an agent cares more about future cumulative reward. Exponential
decay for epsilon, which takes random action to explore the environment. I must say from this
project, I experienced one new aspect of Machine Learning that fascinates me to learn more
about it.
