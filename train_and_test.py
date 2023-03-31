import time
import numpy as np
import matplotlib.pyplot as plt
import numpy
import torch
import random
from random_environment import Environment
from agent import Agent
import cv2
# Main entry point

# import argparse

# parse = argparse.Argument()

# parse.add_argument('--load', type = int, defalut=0)
# parse.add_argument('--batch_size', type = int , defalut=16)

# args = parser.parse_args()


if __name__ == "__main__":

    # This determines whether the environment will be displayed on each each step.
    # When we train your code for the 10 minute period, we will not display the environment.
    display_on = True

    # Create a random seed, which will define the environment
    random_seed = int(time.time())
    np.random.seed(242)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    pixel = 1
    n = 10
    # Create a random environment
    environment1 = Environment(magnification=500/pixel, scale=pixel)
    environment2 = Environment(magnification=500/pixel, scale=pixel)
    environment3 = Environment(magnification=500/pixel, scale=pixel)
    environment4 = Environment(magnification=500/pixel, scale=pixel)
    environment5 = Environment(magnification=500/pixel, scale=pixel)
    environment6 = Environment(magnification=500/pixel, scale=pixel)
    environment7 = Environment(magnification=500/pixel, scale=pixel)
    environment8 = Environment(magnification=500/pixel, scale=pixel)
    environment9 = Environment(magnification=500/pixel, scale=pixel)
    environment10 = Environment(magnification=500/pixel, scale=pixel)

    environment_list = [environment1, environment2, environment3, environment4, environment5,environment6, environment7, environment8, environment9, environment10]



    # Create an agent
    agent = Agent()
    # if args.load = True:
    #agent.network.q_network.load_state_dict(torch.load('q_network.pth'))

    # Get the initial state
    #state = environment.init_state

    # Determine the time at which training will stop, i.e. in 10 minutes (600 seconds) time
    
    
   
    for i in range(n):
        count = 0

        #environment = Environment(magnification=500/pixel, scale=pixel)
        #state = environment.init_state
        #image = environment.image
        # state = environment.image
    # Train the agent, until the time is up
        start_time = time.time()
        end_time = start_time + 30
        image_size=64
        environment = environment_list[random.randrange(0, 10)]
        state = environment.init_state


        #while time.time() < end_time:
        num_while = 0
        while count < 150:
            print(count)
            print('##############################:' ,num_while)
            num_while +=1
            # If the action is to start a new episode, then reset the state
            if agent.has_finished_episode():
                # choose environment
                environment = environment_list[random.randrange(0, 10)]

                state = environment.init_state
            
            image = environment._image(state)
            image = cv2.resize(image,(image_size,image_size))/255
            # state = environment.image
            # Get the state and action from the agent

            #image = torch.Tensor(np.expand_dims(np.transpose(image, (2, 0, 1)), 0)).float().to(device)
            image = torch.Tensor(np.expand_dims(np.transpose(image, (2, 0, 1)), 0)).float().to(device)
            action = agent.get_next_action(image)

            # Get the next state and the distance to the goal
            next_state, distance_to_goal = environment.step(state, action)
            # Return this to the agent
            next_image = environment._image(next_state)
            next_image = cv2.resize(next_image,(image_size,image_size))/255

            #next_image = torch.Tensor(np.expand_dims(np.transpose(next_image, (2, 0, 1)), 0)).float().to(device)
            next_image = torch.Tensor(np.expand_dims(np.transpose(next_image, (2, 0, 1)), 0)).float().to(device)
            #print("dsddddddddddddddddddddddddddddddd",next_image.shape,image.shape)

            agent.set_next_state_and_distance(image, next_image, distance_to_goal)
            # Set what the new state is
            state = next_state
            # Optionally, show the environment
            if display_on:
                environment.show(state)
            if distance_to_goal < 0.07:
                count += 1


    #if agent.reached_goal:
    #    print("Agent reached goal during training")
    #else:
    #    print("Agent did not reach goal during training")
    
    # Test the agent for 100 steps, using its greedy policy
        torch.save(agent.network.q_network.state_dict(), 'q_network.pth')
        state = environment.init_state
        has_reached_goal = False
        for step_num in range(400):
            action = agent.get_greedy_action(image)
            image = cv2.resize(image,(image_size,image_size))
            image = torch.Tensor(np.expand_dims(np.transpose(image, (2, 0, 1)), 0)).float().to(device)
            next_state, distance_to_goal = environment.step(image, action)
            # The agent must achieve a maximum distance of 0.03 for use to consider it "reaching the goal"
            if distance_to_goal < 0.03:
                has_reached_goal = True
                break
            state = next_state

        # Print out the result
    if has_reached_goal:
        print('Reached goal in ' + str(step_num) + ' steps.')
    else:
        print('Did not reach goal. Final distance = ' + str(distance_to_goal))


