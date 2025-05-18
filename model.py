import numpy as np
import torch as t
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim
import pygame
from Environment import *
import random
import matplotlib.pyplot as plt

EPS = 0.4

class DQL(nn.Module):
    def __init__(self, LearningRate=0.01, num_layers=9):

        super().__init__()
        # Слои
        self.inp = nn.Linear(num_layers, 256)
        self.hidden1 = nn.Linear(256, 256)
        self.hidden2 = nn.Linear(256, 256)
        self.out = nn.Linear(256, 5)

        # Память
        self.memory_states = []
        self.memory_actions = []
        self.memory_rewards = []
        self.memory_next_states = []
        self.memory_len = 0
        self.epochs_num = 1
        self.loses = []
        self.epoch = []
        self.rewards_per_epoch = []

        #Параметры
        self.optimizer = optim.Adam(self.parameters(), lr=LearningRate)
        self.criterion = nn.MSELoss()
        self.activation = nn.LeakyReLU()
    
    #Вывод
    def forward(self, x):
        x = self.activation(self.inp(x))
        x = self.activation(self.hidden1(x))
        x = self.activation(self.hidden2(x))
        x = self.activation(self.out(x))
        return x
    
    #Запоминание всех объектов для обучения
    def remember(self, state, action, reward, next_state):
        self.memory_states.append(state)
        self.memory_actions.append(action)
        self.memory_rewards.append(reward)
        self.memory_next_states.append(next_state)
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

        self.memory_len = 0
        return (states, actions, rewards, next_states)
    

    def train(self):
        if (self.memory_len == 0):
            return
        memory_len = self.memory_len
        states, actions, rewards, next_states = self.samplebatch()
       
        NeuroNowAnswer = self.forward(states)
        NeuroNextAnswer = self.forward(next_states)
        predicted_now_value = NeuroNowAnswer[range(memory_len), actions]
        predicted_future_value = t.max(NeuroNextAnswer, dim=1)[0]
        predict_target = rewards + 0.8 * predicted_future_value
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
        env.forest.empty()
        env.empty_space.empty()
        env.border.empty()

        env.train_step = 1
        self.epochs_num = 1
        self.loses = []
        self.epoch = []

        plt.close('all')

    def save_button_tracking(self):
        for event in pygame.event.get():
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if env.save_button.button_rect.collidepoint(event.pos):
                    env.save_button.save_model(self)
    
    def game(self):
        state = env.get_state()
        if (random.random() < EPS):
            action = random.choice(range(4))
        else:
            answer = self.forward((torch.FloatTensor(state)))
            action = torch.argmax(answer).item()
        next_state, reward = env.step(action)
        self.remember(state, action, reward, next_state)

if __name__ == "__main__":
    pygame.init()
    maps = {"map 1": ['E:\VS_project\souless\map_1.xlsx', (420, 395)], 
            "map 2": ['E:\VS_project\souless\map_2.xlsx', (420, 455)],
            "map 3": ['E:\VS_project\souless\map_4.xlsx', (420, 515)]}
    screen = pygame.display.set_mode((WIDTH_SCREEN, HEIGHT_SCREEN))
    pygame.display.set_caption("game")
    clock = pygame.time.Clock()
    env = Game(screen, maps)
    agent = DQL(num_layers=30)
    env.generate_button()

    while True:
        screen.fill((155, 255, 155))
        clock.tick(FPS) 
        
        env.button_tracking()
        
        if env.on_mission:
            env.back_button.draw_button(screen)
            env.save_button.draw_button(screen)
            agent.game()
            agent.save_button_tracking()

            if env.train_step % 300 == 0:
                print(env.train_step)
                agent.train()
                env.car.restart()
                # agent.draw_plot()
                env.zeroing_grad()

            if env.train_step == 20000:
                EPS = 0
            
            env.train_step += 1
        else:
            if env.train_step > 1:
                agent.rollback()
            env.map_button_1.draw_button(screen)
            env.map_button_2.draw_button(screen)
            env.map_button_3.draw_button(screen)
            pygame.display.flip() 