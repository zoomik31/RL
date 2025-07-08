import pygame
import math
import random
import torch
import pandas as pd
import psycopg2
from Sprites import *

ORANGE = (255,140,0)
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

EPS = 0.3

class Game():
    def __init__(self, screen):
        self.fps = 20

        self.empty_space = pygame.sprite.Group()
        self.border = pygame.sprite.Group()
        self.forest = pygame.sprite.Group()
        self.roads = pygame.sprite.Group()
        self.start_cells = pygame.sprite.Group()
        self.snow_cells = pygame.sprite.Group()
        self.snowdrifts = pygame.sprite.Group()
        self.divinglines = pygame.sprite.Group()
        self.puddles = pygame.sprite.Group()

        self.all_map = []
        self.size = 20
        self.done = False
        
        self.train_step = 1
        self.screen = screen
        self.reward = 0
        self.on_mission = False
    
    def generate_map(self, map):
        self.map = map
        car_created = False
        for row in self.map:
            if row[0] == 1:
                brd = Border(row[2]*self.size, row[1]*self.size, self.size)
                self.border.add(brd)
            
            elif row[0] == 2:
                self.flag = Flag(row[2]*self.size, row[1]*self.size, self.size)

            elif row[0] == 3:
                tree = Tree(row[2]*self.size, row[1]*self.size, self.size)
                self.forest.add(tree)

            elif row[0] == 4:
                road = Road(row[2]*self.size, row[1]*self.size, self.size)
                self.roads.add(road)

            elif row[0] == 5:
                self.car = Car(row[2]*self.size, row[1]*self.size, self.size, RED)
                
                start_cell = Start(row[2]*self.size, row[1]*self.size, self.size)
                self.start_cells.add(start_cell)

            elif row[0] == 6:
                snow = Snow(row[2]*self.size, row[1]*self.size, self.size)
                self.snow_cells.add(snow)
            
            elif row[0] == 7:
                snowdrift = SnowDrift(row[2]*self.size, row[1]*self.size, self.size)
                self.snowdrifts.add(snowdrift)
            
            elif row[0] == 8:
                div_line = DivingLine(row[2]*self.size, row[1]*self.size, self.size)
                self.divinglines.add(div_line)
            
            elif row[0] == 9:
                puddle = Puddle(row[2]*self.size, row[1]*self.size, self.size)
                self.puddles.add(puddle)
            else:
                self.emp = Empty(row[2]*self.size, row[1]*self.size, self.size)
                self.empty_space.add(self.emp)
    
        self.dist = math.sqrt((self.car.rect.x-self.flag.rect.x)**2 + (self.car.rect.y-self.flag.rect.y)**2)

        self.generate_button()
            
    def generate_button(self):
        self.back_button = BackButton(200, 900, "choise map")
        self.save_button = SaveModelButton(600, 900, "save model")

        self.back_button.create_button()
        self.save_button.create_button()
    


    def barriers_check(self):
        if pygame.sprite.spritecollideany(self.car, self.forest) or pygame.sprite.spritecollideany(self.car, self.border) or pygame.sprite.spritecollideany(self.car, self.snowdrifts) or pygame.sprite.spritecollideany(self.car, self.puddles):
            return True

    def flag_check(self):
        if pygame.sprite.collide_rect(self.car, self.flag):
            self.done = True
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
        state.append(self.dist)
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

        return state, self.reward, self.done

    def run_game(self):
        self.button_tracking()
        self.reward = 0
        self.measure_distance()
        if self.barriers_check():
            self.reward = -100
        elif self.flag_check():
            self.reward = 150
        # elif self.dist < self.prev_dist:
        #     if self.dist < 300:
        #         self.reward = 30
        #     elif self.dist < 100:
        #         self.reward = 50
        #     else:
        #         self.reward = 10
        # elif self.dist == self.prev_dist:
        #     self.reward = 0
        # else:
        #     if self.dist < 300:
        #         self.reward = 10
        #     elif self.dist < 100:
        #         self.reward = 30
        #     else:
        #         self.reward = 5

        elif self.dist < self.prev_dist: 
            self.reward = 30
        else:
            self.reward = 0 

        self.border.draw(self.screen)
        self.forest.draw(self.screen)
        self.roads.draw(self.screen)
        self.empty_space.draw(self.screen)
        self.snow_cells.draw(self.screen)
        self.snowdrifts.draw(self.screen)
        self.divinglines.draw(self.screen)
        self.puddles.draw(self.screen)
        self.start_cells.draw(self.screen)
        self.screen.blit(self.car.image, self.car.rect)
        self.screen.blit(self.flag.image, self.flag.rect)
        # self.screen.blit(self.car.image, self.car.rect)
        pygame.display.flip() 

    def print_state(self):
        print(self.dist)

    def button_tracking(self):
        for event in pygame.event.get():
            #Закрытие игры
            if event.type == pygame.QUIT:
                pygame.quit()
   
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

                if self.back_button.button_rect.collidepoint(event.pos) and self.on_mission:
                    self.on_mission = False

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
        clock.tick(game.fps) 
        
        game.button_tracking()
        
        
        if game.on_mission:
            game.run_game()
            
        
        else:
            game.map_button_1.draw_button(screen)
            game.map_button_2.draw_button(screen)
            game.map_button_3.draw_button(screen)
            pygame.display.flip() 