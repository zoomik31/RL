import numpy as np
import torch as t
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim
import pygame
from Enviroment import *
# from random_train import *
import random
import matplotlib.pyplot as plt

EPS = 0.25

class DQL(nn.Module):
    def __init__(self, env, LearningRate=0.01, num_layers=9):

        super().__init__()
        # Слои
        self.inp = nn.Linear(num_layers, 32)
        self.hidden1 = nn.Linear(32, 32)
        self.dropout = nn.Dropout(0.3)
        self.hidden2 = nn.Linear(32, 32)
        self.out = nn.Linear(32, 5)
  
        # Память
        self.memory_states = []
        self.memory_actions = []
        self.memory_rewards = []
        self.memory_next_states = []
        self.memory_dones = []
        self.memory_len = 0
        self.epochs_num = 1
        self.loses = []
        self.epoch = []
        self.rewards_per_epoch = []
        self.min_lose = -1

        #Параметры
        self.optimizer = optim.Adam(self.parameters(), lr=LearningRate)
        self.criterion = nn.SmoothL1Loss()
        self.activation = nn.LeakyReLU()

        self.env = env
    
    #Вывод
    def forward(self, x):
        x = self.activation(self.inp(x))
        x = self.dropout(x)
        x = self.hidden1(x)
        x = self.dropout(x)
        x = self.activation(self.hidden2(x))
        x = self.activation(self.out(x))
        return x
    
    #Запоминание всех объектов для обучения
    def remember(self, state, action, reward, next_state, done):
        self.memory_states.append(state)
        self.memory_actions.append(action)
        self.memory_rewards.append(reward)
        self.memory_next_states.append(next_state)
        self.memory_dones.append(done)
        self.memory_len += 1

    # метод для получения тензоров
    def samplebatch(self):
        
        states = t.FloatTensor(self.memory_states).cpu()
        self.memory_states.clear()
        
        actions = t.IntTensor(self.memory_actions).cpu()
        self.memory_actions.clear()

        rewards = t.FloatTensor(self.memory_rewards).cpu()
        self.memory_rewards.clear()
        print(rewards)

        next_states = t.FloatTensor(self.memory_next_states).cpu()
        self.memory_next_states.clear()

        dones = t.FloatTensor(self.memory_dones).cpu()
        self.memory_dones.clear()

        self.memory_len = 0
        return (states, actions, rewards, next_states, dones)
    

    def train(self):
        if (self.memory_len == 0):
            return
        memory_len = self.memory_len
        states, actions, rewards, next_states, dones = self.samplebatch()
       
        NeuroNowAnswer = self.forward(states)
        NeuroNextAnswer = self.forward(next_states)
        predicted_now_value = NeuroNowAnswer[range(memory_len), actions]
        predicted_future_value = t.max(NeuroNextAnswer, dim=1)[0]
        predict_target = rewards + 0.8 * predicted_future_value * (1-dones)
        loss = self.criterion(predict_target, predicted_now_value)
        
        self.loses.append(loss.cpu().item())
        self.rewards_per_epoch.append(t.sum(rewards.cpu()).item())
        self.optimizer.zero_grad()
        loss.backward()
        if (self.inp.weight.grad.norm() < 0.0001):
            self.inp.weight.grad.data += t.FloatTensor([0.001]).cpu()
        self.optimizer.step()

        print(f"Ошибка: {loss}")
    
    def draw_plot(self):
        self.epoch.append(self.epochs_num)
        plt.plot(self.epoch, self.loses)
        plt.ion()
        plt.show()
        self.epochs_num +=1
        plt.pause(0.001)
    
    def rollback(self):
        env.del_state()

        env.train_step = 1
        self.epochs_num = 1
        self.loses = []
        self.epoch = []

        EPS = 0.4

        plt.close('all')
    
    def checkpoint(self):
        if self.loses[-1] < self.min_lose or self.min_lose == -1:
            self.min_lose = self.loses[-1]
            torch.save(self, r'E:\VS_project\souless\model_checkpoint.pt') 

    def game_main(self, model=None):
        if model != None:
            state = self.env.get_state_main()
            answer = model.forward((torch.FloatTensor(state)))
            action = torch.argmax(answer).item()
            
            self.env.step_main(action)
        else:
            state = self.env.get_state_main()
            if (random.random() < EPS):
                action = random.choice(range(4))
            else:
                answer = self.forward((torch.FloatTensor(state)))
                action = torch.argmax(answer).item()
            next_state, reward, done = self.env.step_main(action)
            self.remember(state, action, reward, next_state, int(done))

    def game_side(self):
        state = self.env.get_state_side()
        if (random.random() < EPS):
            action = random.choice(range(4))
        else:
            answer = self.forward((torch.FloatTensor(state)))
            action = torch.argmax(answer).item()
        next_state, reward, done = self.env.step_side(action)
        self.remember(state, action, reward, next_state, int(done))

if __name__ == "__main__":
    pygame.init()
    maps = "E:\VS_project\souless\maps.xlsx"
    screen = pygame.display.set_mode((WIDTH_SCREEN, HEIGHT_SCREEN))
    pygame.display.set_caption("game")
    clock = pygame.time.Clock()
    env = Game(screen)
    agent = DQL(num_layers=34)#34 13
    env.generate_button()
    agent.rollback()

    while True:
        screen.fill((155, 255, 155))
        clock.tick(FPS) 
        
        env.button_tracking(agent)
        
        if env.on_mission:
            env.save_button.draw_button(screen)
            agent.game()
            
            if env.train_step % 20 == 0:
                print(env.train_step)
                agent.train()
                agent.checkpoint()
                # env.car.restart()
                # agent.draw_plot()
            if env.train_step % 500 == 0:
                print(env.train_step)
                EPS = 0
                # agent.train()
                # env.car.restart()
                # agent.draw_plot()
            
            env.train_step += 1
        else:
            if env.train_step > 1:
                agent.rollback()
            env.map_button_2.draw_button(screen)
            pygame.display.flip()