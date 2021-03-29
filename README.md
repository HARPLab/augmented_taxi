# Augmented Taxi Domain

This repository primarily consists of 

* An implementation of the original Taxi domain [Dietterich, JAIR2000] with additional environmental complexities. Built on David Abel's [simple_rl framework](https://github.com/david-abel/simple_rl). 
* Methods of a) selecting demonstrations that effectively summarize the agent's policy (i.e. behavior) to a human, and b) requesting demonstrations of what the human believes an agent would do in specific environments. 

Required packages include [numpy](http://www.numpy.org/), [matplotlib](http://matplotlib.org/), [pypoman](https://github.com/stephane-caron/pypoman) to perform computational geometry with polytopes (i.e. BEC regions, see below), and [pygame](http://www.pygame.org/news) if you want to visualize some MDPs.

The main file is augmented_taxi.py, which currently has functions for a) generating an agent in the Augmented Taxi MDP, b) obtaining a [Bayesian IRL summary](https://link.springer.com/article/10.1007/s10514-018-9771-0) of the agent's policy, c) obtaining a [BEC summary](https://arxiv.org/pdf/1805.07687.pdf) of the agent's policy, and d) obtaining test environments to query the human's understanding of the agent's policy. 

