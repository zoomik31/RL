import pygame
import math
import random
import torch
import pandas as pd
import psycopg2
from Sprites import *

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

EPS = 0.3

class Game():
    def __init__(self, screen):
        self.fps = 20

        self.empty_space = pygame.sprite.Group()
        self.border = pygame.sprite.Group()
        self.forest = pygame.sprite.Group()

        self.all_map = []
        self.size = 20
        
        self.train_step = 1
        self.screen = screen
        self.reward = 0
        self.on_mission = False
    
    def generate_map(self, map):
        car_created = False
        for row in map:
            layer = []
            if row[0] == 1:
                brd = Border(row[2]*self.size, row[1]*self.size, self.size)
                self.border.add(brd)
                layer.append(1)
            
            elif row[0] == 2:
                self.flag = Flag(row[2]*self.size, row[1]*self.size, self.size)
                layer.append(2)

            elif row[0] == 3:
                tree = Tree(row[2]*self.size, row[1]*self.size, self.size)
                self.forest.add(tree)
                layer.append(3)

            elif row[0] == 5:
                car_created = True
                self.car = Car(row[2]*self.size, row[1]*self.size, self.size)
                layer.append(0)
            else:
                self.emp = Empty(row[2]*self.size, row[1]*self.size, self.size)
                self.empty_space.add(self.emp)
                layer.append(0)

        self.all_map.append(layer)
        if car_created == False:
            self.car = Car(9 * self.size, 35 * self.size, self.size)
        # self.dist = math.sqrt((self.car.rect.x-self.flag.rect.x)**2 + (self.car.rect.y-self.flag.rect.y)**2)
            
    def generate_button(self):
        self.map_button_1 = MapButton(self.maps["map 1"][1][0], self.maps["map 1"][1][1], "1 map", self.maps["map 1"][0])
        self.map_button_2 = MapButton(self.maps["map 2"][1][0], self.maps["map 2"][1][1], "2 map", self.maps["map 2"][0])
        self.map_button_3 = MapButton(self.maps["map 3"][1][0], self.maps["map 3"][1][1], "3 map", self.maps["map 3"][0])
        
        self.back_button = BackButton(200, 900, "choise map")
        self.save_button = SaveModelButton(600, 900, "save model")

        self.map_button_1.create_button()
        self.map_button_2.create_button()
        self.map_button_3.create_button()

        self.back_button.create_button()
        self.save_button.create_button()
    
    def generate_text(self, screen):
        self.dir_text = DirectionText(400, 900)
        self.dir_text.create_text(self.car.direction)

        self.dir_text.draw_dir_text(screen)


    def barriers_check(self):
        obstacles = pygame.sprite.Group()
        obstacles.add(self.forest)
        obstacles.add(self.border)
        self.car.forward_border = False
        self.car.back_border = False
        for obstacle in obstacles:
            top_left, top_right, bottom_right, bottom_left = self.car.vertices
            # if (
            #     ((obstacle.rect.x <= top_right[0] <= obstacle.rect.x+obstacle.size) and (obstacle.rect.y <= top_right[1] <= obstacle.rect.y+obstacle.size)) or
            #     ((obstacle.rect.x <= bottom_right[0] <= obstacle.rect.x+obstacle.size) and (obstacle.rect.y <= bottom_right[1] <= obstacle.rect.y+obstacle.size)) or
            #     ((obstacle.rect.x <= top_left[0] <= obstacle.rect.x+obstacle.size) and (obstacle.rect.y <= top_left[1] <= obstacle.rect.y+obstacle.size)) or
            #     ((obstacle.rect.x <= bottom_left[0] <= obstacle.rect.x+obstacle.size) and (obstacle.rect.y <= bottom_left[1] <= obstacle.rect.y+obstacle.size))

            # ):
            #     print('s')
            if (
                ((obstacle.rect.x <= top_right[0] <= obstacle.rect.x+obstacle.size) and (obstacle.rect.y <= top_right[1] <= obstacle.rect.y+obstacle.size)) or
                ((obstacle.rect.x <= bottom_right[0] <= obstacle.rect.x+obstacle.size) and (obstacle.rect.y <= bottom_right[1] <= obstacle.rect.y+obstacle.size))
                ):
                self.car.forward_border = True
                self.car.back_border = False
                self.car.emergency_braking()
                return True
            elif (
                ((obstacle.rect.x <= top_left[0] <= obstacle.rect.x+obstacle.size) and (obstacle.rect.y <= top_left[1] <= obstacle.rect.y+obstacle.size)) or
                ((obstacle.rect.x <= bottom_left[0] <= obstacle.rect.x+obstacle.size) and (obstacle.rect.y <= bottom_left[1] <= obstacle.rect.y+obstacle.size))
                ):
                self.car.forward_border = False
                self.car.back_border = True
                self.car.emergency_braking()
                return True
        return False

    def flag_check(self):
        top_left, top_right, bottom_right, bottom_left = self.car.vertices

        # Проходим по спрайтам и проверяем коллизию
        if (
                ((self.flag.rect.x <= top_left[0] <= self.flag.rect.x+self.flag.size) and (self.flag.rect.y <= top_left[1] <= self.flag.rect.y+self.flag.size)) or
                ((self.flag.rect.x <= top_right[0] <= self.flag.rect.x+self.flag.size) and (self.flag.rect.y <= top_right[1] <= self.flag.rect.y+self.flag.size)) or 
                ((self.flag.rect.x <= bottom_left[0] <= self.flag.rect.x+self.flag.size) and (self.flag.rect.y <= bottom_left[1] <= self.flag.rect.y+self.flag.size)) or 
                ((self.flag.rect.x <= bottom_right[0] <= self.flag.rect.x+self.flag.size) and (self.flag.rect.y <= bottom_right[1] <= self.flag.rect.y+self.flag.size))
            ):
                self.car.restart()
                return True
        return False

        
    def measure_main_reward_distance(self):
        self.prev_dist = self.dist
        self.dist = math.sqrt((self.car.rect.x-self.flag.rect.x)**2 + (self.car.rect.y-self.flag.rect.y)**2)
    
    
    def get_state(self):

        state = []

        # self.car.go_right()
        # if self.barriers_check():
        #     self.block_right = True
        # else: 
        #     self.car.go_back()
        #     self.block_right = False
        
        # self.car.go_left()
        # if self.barriers_check():
        #     self.block_left = True
        # else: 
        #     self.car.go_back()
        #     self.block_left = False
        
        # self.car.go_down()
        # if self.barriers_check():
        #     self.block_down = True
        # else: 
        #     self.car.go_back()
        #     self.block_down = False
        
        # self.car.go_up()
        # if self.barriers_check():
        #     self.block_up = True
        # else: 
        #     self.car.go_back()
        #     self.block_up = False
        
        # state = [int(self.car.rect.x < self.flag.rect.x), int(self.car.rect.y < self.flag.rect.y),
        #             int(self.car.rect.x > self.flag.rect.x), int(self.car.rect.y > self.flag.rect.y),
        #             int(self.block_up), int(self.block_right),int(self.block_down),int(self.block_left),
        #             int(self.car.direction == "up"), int(self.car.direction == "right"), int(self.car.direction == "down"), int(self.car.direction == "left")
        # ]

        for y in range(int(self.car.rect.y/self.size)-2, (int(self.car.rect.y/self.size)+3)):
            for x in range(int(self.car.rect.x/self.size)-2, int(self.car.rect.x/self.size)+3):
                state.append(self.all_map[y][x])
        
        state.append(int(self.car.direction == "up"))
        state.append(int(self.car.direction == "right"))
        state.append(int(self.car.direction == "down"))
        state.append(int(self.car.direction == "left"))

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
        self.reward = 0
        self.flag_check()
        barrier_chek = self.barriers_check()
        while self.barriers_check():
            if self.car.forward_border:
                self.car.move_forward(-1)
            elif self.car.back_border:
                self.car.move_forward(1)
            self.car.calculate_vertices()
        # self.measure_main_reward_distance()
        # if self.barriers_check():
        #     self.reward = -130
        # elif self.flag_check():
        #     self.reward = 150

        # elif self.dist < self.prev_dist:
        #     self.reward = 40
        # elif self.dist == self.prev_dist:
        #     self.reward = 0
        # else:
        #     self.reward = 10

        self.generate_text(self.screen)

        self.border.draw(self.screen)
        self.forest.draw(self.screen)
        self.empty_space.draw(self.screen)

        self.car.draw()
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
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_DOWN:
                    self.car.go_down()
                elif event.key == pygame.K_RIGHT:
                    self.car.go_right()
                elif event.key == pygame.K_UP:
                    self.car.go_up()
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