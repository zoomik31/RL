import pygame
import math
import random
import torch
import pandas as pd

from Sprites import *

GRAD_4 = (0, 0, 255)
GRAD_3 = (0, 0, 128)
GRAD_2 = (0, 0, 64)
GRAD_1 = (0, 0, 32)
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
    def __init__(self, screen, maps):
        self.maps = maps
        self.empty_space = pygame.sprite.Group()
        self.border = pygame.sprite.Group()
        self.forest = pygame.sprite.Group()
        # self.map_group = pygame.sprite.Group()

        self.all_map = []
        self.size = 20
        self.train_step = 1
        self.screen = screen
        self.reward = 0
        self.on_mission = False
    
    def generate_map(self, map):
        self.map = pd.read_excel(map, header=None)
        for y in range(len(self.map)):
            layer = []
            for x in range(len(self.map[0])):
                if self.map[y][x] == 1:
                    brd = Border(x*self.size, y*self.size, self.size)
                    self.border.add(brd)
                    # self.map_group.add(brd)
                    layer.append(1)
                
                elif self.map[y][x] == 2:
                    self.flag = Flag(x*self.size, y*self.size, self.size)
                    layer.append(2)
                    # self.map_group.add(self.flag)

                elif self.map[y][x] == 3:
                    tree = Tree(x*self.size, y*self.size, self.size)
                    self.forest.add(tree)
                    layer.append(3)
                    # self.map_group.add(tree)

                else:
                    emp = Empty(x*self.size, y*self.size, self.size)
                    self.empty_space.add(emp)
                    layer.append(0)
                    # self.map_group.add(emp)

            self.all_map.append(layer)
        
        self.car = Car(9 * self.size, 35 * self.size, self.size)
        self.dist = math.sqrt((self.car.rect.x-self.flag.rect.x)**2 + (self.car.rect.y-self.flag.rect.y)**2)
    
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
        
        # state.append(int(self.car.direction == "up"))
        # state.append(int(self.car.direction == "right"))
        # state.append(int(self.car.direction == "down"))
        # state.append(int(self.car.direction == "left"))
        # state.append(int(self.car.rect.x/20))
        # state.append(int(self.car.rect.y/20))
        # state.append(int(self.flag.rect.x/20))
        # state.append(int(self.flag.rect.y/20))
        # state.append(self.dist)

        # for y in range(int(self.car.rect.y/self.size)-2, (int(self.car.rect.y/self.size)+3)):
        #     for x in range(int(self.car.rect.x/self.size)-2, int(self.car.rect.x/self.size)+3):
        #         for cell in self.map_group:
        #             if cell.rect.collidepoint(x*self.size, y*self.size):
        #                 if cell.colour == WHITE:
        #                     state.append(0)
        #                 elif cell.colour[2] == 0:
        #                     state.append(10)
        #                 else:
        #                     state.append(int(round(256/cell.colour[2])))

        # state.append(int(self.car.direction == "up"))
        # state.append(int(self.car.direction == "right"))
        # state.append(int(self.car.direction == "down"))
        # state.append(int(self.car.direction == "left"))
        state.append(int(self.car.rect.x/20))
        state.append(int(self.car.rect.y/20))
        state.append(int(self.flag.rect.x/20))
        state.append(int(self.flag.rect.y/20))
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
        return state, self.reward
    
    def color_change(self):
        inv_cell = False
        for cells in self.empty_space:
            if pygame.sprite.collide_rect(self.car, cells):
                x, y = cells.rect.x, cells.rect.y
                if cells.colour == WHITE:
                    inv_cell = True
                    colour = GRAD_4
                elif cells.colour == GRAD_4:
                    inv_cell = False
                    colour = GRAD_3
                elif cells.colour == GRAD_3:
                    inv_cell = False
                    colour = GRAD_2
                elif cells.colour == GRAD_2 or cells.colour == GRAD_1:
                    inv_cell = False
                    colour = GRAD_1
                # self.map_group.remove(cells)
                self.empty_space.remove(cells)
                emp = Empty(x, y, self.size, colour)
                self.empty_space.add(emp)
                # self.map_group.add(emp)

        return inv_cell
            
    def zeroing_grad(self):
        for cells in self.empty_space:
            self.empty_space.remove(cells)
            # self.map_group.remove(cells)
        
        for y in range(len(self.map)):
            for x in range(len(self.map[0])):
                if self.map[y][x] == 0: 
                    emp = Empty(x*self.size, y*self.size, self.size)
                    self.empty_space.add(emp)
                    # self.map_group.add(emp)
    
    def grad_score(self):
        for cells in self.empty_space:
            if pygame.sprite.collide_rect(self.car, cells):
                if cells.colour == WHITE:
                    return 0
                else:
                    return int(round(255/cells.colour[2]))

    def run_game(self):

        inv_cell = self.color_change()
        #Совершение действия
        self.reward = 0
        self.measure_distance()

        if inv_cell:
            self.reward += 20
        else:
            self.reward += -5

        if self.dist < self.prev_dist:
            self.reward += 20
        elif self.dist == self.prev_dist:
            self.reward += 0
        else:
            self.reward += 5

        if self.barriers_check():
            self.reward = -120
        elif self.flag_check():
            self.reward = 200

        self.border.draw(self.screen)
        self.forest.draw(self.screen)
        self.empty_space.draw(self.screen)
        self.screen.blit(self.flag.image, self.flag.rect)
        self.screen.blit(self.car.image, self.car.rect)
        pygame.display.flip() 
    
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
                elif event.key == pygame.K_0:
                    self.zeroing_grad()
            
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
        clock.tick(FPS) 
        
        game.button_tracking()
        
        if game.on_mission:
        
            game.run_game()
        
        
        else:
            game.map_button_1.draw_button(screen)
            game.map_button_2.draw_button(screen)
            game.map_button_3.draw_button(screen)
            pygame.display.flip() 