# DQN_maze_finding

### 1. Abstract

1. Generate the random Maze
2. Make the state from image using ‘CNN’
3. Make the action from state using ‘DQN’
4. Predict the next action from the now_state and action
5. Give the reward depending on result and train the model

### 2. Member

장준형, 양승운

### 3. Code implementation

python test.py

### 4. Model explanation

In this program, we use the DQN model to train. 
Therefore, we use replay buffer and target network to train well.

![image](https://user-images.githubusercontent.com/89409079/229114221-a7363a50-99d0-44fb-9a00-a43d4b4b9fbe.png)

### 4. Training

<img width="526" alt="image" src="https://user-images.githubusercontent.com/89409079/229114357-879ede1a-97e7-405b-8b72-c37ee5b31081.png">

In the terminal, use 'python test.py'
Above picture is the process of training. 

### 5. Result

<img width="306" alt="image" src="https://user-images.githubusercontent.com/89409079/229113675-25ce4113-06e8-47c7-a79d-a5924b3575eb.png">

The graph shows that number of steps to reach the destination decreases when the episode goes on 

### 6. Conclusion

We train the model using 10 random maze
At first it follow the random policy, so it doesn’t finish the episode
As the episode begins to grow, it follow the optimal policy as the epsilon changes 
Finally the red point reaches to destination in any maze.
