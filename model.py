import numpy as np
import torch as t
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim
import pygame
from Environment import *
import random

EPS = 0.4

class DQL(nn.Module):
    def __init__(self, LearningRate=0.01, num_layers=9):

        super().__init__()
        # Слои
        self.inp = nn.Linear(num_layers, 128)
        self.hidden1 = nn.Linear(128, 128)
        self.hidden2 = nn.Linear(128, 128)
        self.out = nn.Linear(128, 5)

        # Память
        self.memory_states = []
        self.memory_actions = []
        self.memory_rewards = []
        self.memory_next_states = []
        self.memory_len = 0
        self.loses = []
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
    map = 'E:\VS_project\souless\map_4.xlsx'
    screen = pygame.display.set_mode((WIDTH_SCREEN, HEIGHT_SCREEN))
    pygame.display.set_caption("game")
    clock = pygame.time.Clock()
    env = Game(screen, map)
    agent = DQL(num_layers=12)
    env.generate_map()
    train_step = 1

    while True:
        screen.fill((155, 255, 155))
        clock.tick(FPS) 
        

        for event in pygame.event.get():
            #Закрытие игры
            if event.type == pygame.QUIT:
                pygame.quit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_DOWN:
                    env.car.go_down()
                elif event.key == pygame.K_RIGHT:
                    env.car.go_right()
                elif event.key == pygame.K_UP:
                    env.car.go_up()
                elif event.key == pygame.K_LEFT:
                    env.car.go_left()
        
        agent.game()

        if train_step % 200 == 0:
            print(train_step)
            agent.train()
            env.car.restart()

        
        train_step += 1