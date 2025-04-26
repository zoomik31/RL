import pygame
import math
import random
import torch
import pandas as pd

from Sprites import *
from model import DQL

BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)
RED = (255, 0, 0)
GREY = (123, 241, 123)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREEN = (0, 255, 0)

WIDTH = 840
HEIGHT = 840

WIDTH_SCREEN = 840
HEIGHT_SCREEN = 950

FPS = 60
EPS = 0.4


class Game():
    def __init__(self, screen, map):
        self.map = pd.read_excel(map, header=None)
        self.empty_space = pygame.sprite.Group()
        self.border = pygame.sprite.Group()
        self.forest = pygame.sprite.Group()
        self.size = WIDTH/len(self.map)
        self.screen = screen
        self.reward = 0
    
    def generate_map(self):
        for y in range(len(self.map)):
            layer = []
            for x in range(len(self.map[0])):
                if self.map[y][x] == 1:
                    brd = Border(x*self.size, y*self.size, self.size)
                    self.border.add(brd)
                
                elif self.map[y][x] == 2:
                    self.flag = Flag(x*self.size, y*self.size, self.size)

                elif self.map[y][x] == 3:
                    tree = Tree(x*self.size, y*self.size, self.size)
                    self.forest.add(tree)

                else:
                    emp = Empty(x*self.size, y*self.size, self.size)
                    self.empty_space.add(emp)
        
        self.car = Car(9 * self.size, 35 * self.size, self.size)
        self.dist = math.sqrt((self.car.rect.x-self.flag.rect.x)**2 + (self.car.rect.y-self.flag.rect.y)**2)

    def generate_button(self):
        self.map_button_1 = MapButton(self.maps["map 1"][1][0], self.maps["map 1"][1][1], "1 map", self.maps["map 1"][0])
        self.map_button_2 = MapButton(self.maps["map 2"][1][0], self.maps["map 2"][1][1], "2 map", self.maps["map 2"][0])
        self.map_button_3 = MapButton(self.maps["map 3"][1][0], self.maps["map 3"][1][1], "3 map", self.maps["map 3"][0])

        self.switch_button = SwitchButton(200, 900, "Switch map")

        self.save_button = SaveModelButton(600, 900, "Save model")
        
        self.map_button_1.create_button()
        self.map_button_2.create_button()
        self.map_button_3.create_button()

        self.switch_button.create_button()
        self.save_button.create_button()

    def barriers_check(self):
        if pygame.sprite.spritecollideany(self.car, self.forest) or pygame.sprite.spritecollideany(self.car, self.border):
            self.car.go_back()
            return True

    def flag_check(self):
        if pygame.sprite.collide_rect(self.car, self.flag):
            self.car.restart()
            return True

    def measure_distance(self):
        self.prev_dist = self.dist
        self.dist = math.sqrt((self.car.rect.x-self.flag.rect.x)**2 + (self.car.rect.y-self.flag.rect.y)**2)

    def get_state(self):

        state = []

        self.car.go_right()
        if self.barriers_check():
            self.block_right = True
        else: 
            self.car.go_back()
            self.block_right = False
        
        self.car.go_left()
        if self.barriers_check():
            self.block_left = True
        else: 
            self.car.go_back()
            self.block_left = False
        
        self.car.go_down()
        if self.barriers_check():
            self.block_down = True
        else: 
            self.car.go_back()
            self.block_down = False
        
        self.car.go_up()
        if self.barriers_check():
            self.block_up = True
        else: 
            self.car.go_back()
            self.block_up = False
        
        state = [int(self.car.rect.x < self.flag.rect.x), int(self.car.rect.y < self.flag.rect.y),
                    int(self.car.rect.x > self.flag.rect.x), int(self.car.rect.y > self.flag.rect.y),
                    int(self.block_up), int(self.block_right),int(self.block_down),int(self.block_left),
                    int(self.car.direction == "up"), int(self.car.direction == "right"), int(self.car.direction == "down"), int(self.car.direction == "left")
        ]
        return state

    def step(self, action):
        if action == 0:
            self.car.go_up()
        if action == 1:
            self.car.go_right()
        if action == 2:
            self.car.go_down()
        if action == 3:
            self.car.go_left()
        
        self.run_game()
        state = self.get_state()
        return state, self.reward

    def run_game(self):
        #Совершение действия
        self.reward = 0
        self.measure_distance()
        if self.barriers_check():
            self.reward = -100
        elif self.flag_check():
            self.reward = 100
        # elif self.dist < self.prev_dist:
        #     self.reward = 50
        # elif self.dist == self.prev_dist:
            self.reward = 0
        else:
            self.reward = 30


        self.border.draw(self.screen)
        self.forest.draw(self.screen)
        self.empty_space.draw(self.screen)
        self.screen.blit(self.flag.image, self.flag.rect)
        self.screen.blit(self.car.image, self.car.rect)
        pygame.display.flip() 
    
    def button_tracking(self, model):
        for event in pygame.event.get():
            #Закрытие игры
            if event.type == pygame.QUIT:
                pygame.quit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_DOWN:
                    game.car.go_down()
                elif event.key == pygame.K_RIGHT:
                    game.car.go_right()
                elif event.key == pygame.K_UP:
                    game.car.go_up()
                elif event.key == pygame.K_LEFT:
                    self.car.go_left()
            
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if self.map_button_1.button_rect.collidepoint(event.pos) and not(self.on_mission):
                    self.on_mission = True
                    self.generate_map(self.map_button_1.map)
                if self.map_button_2.button_rect.collidepoint(event.pos) and not(self.on_mission):
                    self.on_mission = True
                    self.generate_map(self.map_button_2.map)
                if self.map_button_3.button_rect.collidepoint(event.pos) and not(self.on_mission):
                    self.on_mission = True
                    self.generate_map(self.map_button_3.map)
                
                if self.switch_button.button_rect.collidepoint(event.pos) and self.on_mission:
                    self.on_mission = False
                
                if self.save_button.button_rect.collidepoint(event.pos):
                    self.save_button.save(model.state_dict(), 'model')

if __name__ == "__main__":
    pygame.init()
    maps = {"map 1": ['E:\VS_project\souless\map_1.xlsx', (420, 395)], 
            "map 2": ['E:\VS_project\souless\map_2.xlsx', (420, 455)],
            "map 3": ['E:\VS_project\souless\map_4.xlsx', (420, 515)]}
    screen = pygame.display.set_mode((WIDTH_SCREEN, HEIGHT_SCREEN))
    pygame.display.set_caption("game")
    clock = pygame.time.Clock()
    game = Game(screen, maps)
    game.generate_button()

    while True:
        screen.fill((155, 255, 155))
        clock.tick(FPS) 
        
        game.button_tracking()
        
        if game.on_mission:
            game.switch_button.draw_button(screen)
            game.run_game()
        else:
            game.map_button_1.draw_button(screen)
            game.map_button_2.draw_button(screen)
            game.map_button_3.draw_button(screen)
            pygame.display.flip() 