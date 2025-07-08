import torch
import random
import numpy as np
from collections import deque # store memories - its a stack LIFO
from game import SnakeGameAI, Direction, Point
from model import Linear_QNet, QTrainer
from helper import plot

MAX_MEMORY = 100_000_00 # 1000000 items in memory
BATCH_SIZE = 100000
LR = 0.001

class Agent:
    def __init__(self):
        self.num_of_games = 0
        self.epsilon = 0 # control randomness
        self.gamma = 0.9 # discount rate - confused about this
        self.memory = deque(maxlen=MAX_MEMORY) # if we exceed memory it will auto call popleft()
        self.model = Linear_QNet(11,256,3)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)

    def get_state(self, game):
        head = game.snake[0]    # list of the snake
        point_l = Point(head.x -20,head.y)  # points next to the head in all directions
        point_r = Point(head.x +20,head.y)  # checks if its next to the boundry in all directions
        point_u = Point(head.x,head.y-20)  # if this hits the boundry and if thats a danger
        point_d = Point(head.x,head.y+20)  # - 20 hard coded value for block size  (tuple)

        dir_l = game.direction == Direction.LEFT   # boolean if the current game direction equals to
        dir_r = game.direction == Direction.RIGHT # left , right , up or down
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        state = [
            #danger straight - if danger is ahead
            (dir_r and game.is_collision(point_r)) or # current direction right and the point right of
            (dir_l and game.is_collision(point_l)) or # us gives us a collision then we have a danger.
            (dir_u and game.is_collision(point_u)) or
            (dir_d and game.is_collision(point_d)),

            #danger right - if danger is to the right of the snake
            (dir_u and game.is_collision(point_r)) or # if we go up and the point to the right of us
            (dir_d and game.is_collision(point_l)) or # gives us a collision then we have a danger to the right
            (dir_l and game.is_collision(point_u)) or
            (dir_r and game.is_collision(point_d)),

            #danger left - if danger is to the left of the snake
            (dir_d and game.is_collision(point_r)) or # ^^
            (dir_u and game.is_collision(point_l)) or
            (dir_r and game.is_collision(point_u)) or
            (dir_l and game.is_collision(point_d)),

            #Move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,

            # Food Location
            game.food.x < game.head.x, # food left
            game.food.x > game.head.x, # food right
            game.food.y < game.head.y, # food up
            game.food.y > game.head.y  # food down 
            ]
        
        return np.array(state,dtype=int)  # trick to turn true or false booleans to 
        # to 0 or 1s.

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done)) #only append as one tuple extra () 
        # if this exceeds max memory then popleft off stack (if MEM_MEMORY is reached)

    def train_long_memory(self): # grab 1000 samples from memory  - whole batch of paramters can call both and of different sizes
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory,BATCH_SIZE) #returns list of tuples
        else:
            mini_sample = self.memory
            
        #extract each of the states actions etc and 
        states, actions, rewards, next_states, dones = zip(*mini_sample)
        # multiple of each
        self.trainer.train_step(states, actions, rewards, next_states, dones)
    
    def train_short_memory(self, state, action, reward, next_state, done):  # for one of parameters
        self.trainer.train_step(state, action, reward, next_state, done)
        
    
    def get_action(self,state):
        # at the start do random moves: which is the tradeoff before exploration of the enviroment
        # compared to explotation of the agent/model more explotation as the model becomes more 
        # advanced
        self.epsilon = 400 - self.num_of_games  # more games less epsilon which will be the randomness
        final_move = [0,0,0]     
        if random.randint(0,200) < self.epsilon: # this can become negative and then there will no longer
            move = random.randint(0,2)  # any random moves
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()  # look on ipad for this
            # e.g [5.0,2.7,0.1] - [1,0,0] (item as it only gets the largest value)
            final_move[move] = 1
        
        return final_move


def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = Agent()
    game = SnakeGameAI()
    while True:
        #get old state
        state_old = agent.get_state(game)

        #get move
        final_move = agent.get_action(state_old)

        #perform move and get new state
        reward, done, score = game.play_step(final_move)
        state_new = agent.get_state(game)

        #Train short memory
        agent.train_short_memory(state_old, final_move, reward, state_new, done)

        #remember
        agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            # trains again on all previous moves - helps improve itself
            game.reset()
            agent.num_of_games += 1 
            agent.train_long_memory()

            if score > record:
                record = score
                agent.model.save()

            print('Game', agent.num_of_games, 'Score', score, 'Record:', record)

            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.num_of_games
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores)

if __name__ == '__main__':
    train()
