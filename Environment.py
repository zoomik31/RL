import pygame
import math
import random
import torch
import pandas as pd

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
    def __init__(self, screen, maps):
        self.maps = maps
        self.fps = 90

        self.empty_space = pygame.sprite.Group()
        self.border = pygame.sprite.Group()
        self.forest = pygame.sprite.Group()
        self.reward_group = pygame.sprite.Group()

        self.all_map = []
        self.size = 20
        
        self.train_step = 1
        self.screen = screen
        self.reward = 0
        self.on_mission = False
    
    def generate_map(self, map):
        self.count_add_reward = []

        self.map = pd.read_excel(map, header=None)
        for y in range(len(self.map)):
            layer = []
            for x in range(len(self.map[0])):
                if self.map[y][x] == 1:
                    brd = Border(x*self.size, y*self.size, self.size)
                    self.border.add(brd)
                    layer.append(1)
                
                elif self.map[y][x] == 2:
                    self.flag = Flag(x*self.size, y*self.size, self.size)
                    layer.append(2)

                elif self.map[y][x] == 3:
                    tree = Tree(x*self.size, y*self.size, self.size)
                    self.forest.add(tree)
                    layer.append(3)
                
                elif self.map[y][x] == 4:
                    self.count_add_reward.append((x, y, self.size))
                    add = AddRewards(x*self.size, y*self.size, self.size)
                    self.reward_group.add(add)
                    layer.append(4)

                else:
                    self.emp = Empty(x*self.size, y*self.size, self.size)
                    self.empty_space.add(self.emp)
                    layer.append(0)

            self.all_map.append(layer)
        
        self.car = Car(9 * self.size, 35 * self.size, self.size)
        self.dist_main_reward = math.sqrt((self.car.rect.x-self.flag.rect.x)**2 + (self.car.rect.y-self.flag.rect.y)**2)
        
        self.dist_add_reward = []
        for add in self.reward_group:
            self.dist_add_reward.append(math.sqrt((self.car.rect.x-add.rect.x)**2 + (self.car.rect.y- add.rect.y)**2))
            
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
        if pygame.sprite.spritecollideany(self.car, self.forest) or pygame.sprite.spritecollideany(self.car, self.border):
            self.car.go_back()
            return True

    def flag_check(self):
        if pygame.sprite.collide_rect(self.car, self.flag):
            self.car.restart()
            return True
    
    def add_reward_check(self):
        if pygame.sprite.spritecollideany(self.car, self.reward_group):
            for sprite in self.reward_group:
                    if sprite.rect.center == self.car.rect.center:
                        self.all_map[int(self.car.rect.y/self.size)][int(self.car.rect.x/self.size)] = 0
                        self.reward_group.remove(sprite)
            
            return True
        

    def measure_main_reward_distance(self):
        self.prev_dist_main_reward = self.dist_main_reward
        self.dist_main_reward = math.sqrt((self.car.rect.x-self.flag.rect.x)**2 + (self.car.rect.y-self.flag.rect.y)**2)
    
    def measure_add_reward_distance(self):
        self.prev_dist_add_reward = self.dist_add_reward.copy()
        del_zero = []

        for i in self.dist_add_reward:
            if i == 0:
                self.dist_add_reward.remove(0.0)
        
        for i, add in enumerate(self.reward_group):
            # print(self.dist_add_reward[i], i)
            # if int(self.dist_add_reward[i]) == 0:
            #     print(self.dist_add_reward, i)
            #     self.dist_add_reward.remove(0.0)
            #     del_zero.append(i)
            # else:
                # if not del_zero:
                    self.dist_add_reward[i] = math.sqrt((self.car.rect.x-add.rect.x)**2 + (self.car.rect.y-add.rect.y)**2)
                # else:
                #     self.dist_add_reward[i-1] = math.sqrt((self.car.rect.x-add.rect.x)**2 + (self.car.rect.y-add.rect.y)**2)
        if del_zero:
            self.prev_dist_add_reward.pop(del_zero[0]-1)
        
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
        # state.append(self.dist_main_reward)
        # state.append(float(min(self.dist_add_reward)))

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
        self.measure_main_reward_distance()
        self.measure_add_reward_distance()
        if self.barriers_check():
            self.reward = -130
        elif self.flag_check():
            self.reward = 150

        elif self.add_reward_check():
            self.reward = 80 
    
        elif self.reward_group:
            if float(min(self.dist_add_reward)) < float(min(self.prev_dist_add_reward)):
                self.reward = 40
            elif float(min(self.dist_add_reward)) == float(min(self.prev_dist_add_reward)):
                self.reward = 0
            else:
                self.reward = 10
        
        elif self.dist_main_reward < self.prev_dist_main_reward:
            self.reward = 40
        elif self.dist_main_reward == self.prev_dist_main_reward:
            self.reward = 0
        else:
            self.reward = 10

        self.generate_text(self.screen)

        self.border.draw(self.screen)
        self.forest.draw(self.screen)
        self.reward_group.draw(self.screen)
        self.empty_space.draw(self.screen)

        self.screen.blit(self.flag.image, self.flag.rect)
        self.screen.blit(self.car.image, self.car.rect)
        pygame.display.flip() 
    
    def reset_add_reward(self):
        for sprite in self.reward_group:
            self.reward_group.remove(sprite)

        for x, y, size in self.count_add_reward:
            add = AddRewards(x*size, y*size, size)
            self.all_map[y][x] = 4
            self.reward_group.add(add)
        
        self.dist_add_reward = []
        for add in self.reward_group:
            self.dist_add_reward.append(math.sqrt((self.car.rect.x-add.rect.x)**2 + (self.car.rect.y- add.rect.y)**2))



    def print_state(self):
        print(self.dist_add_reward)

    def button_tracking(self):
        for event in pygame.event.get():
            #Закрытие игры
            if event.type == pygame.QUIT:
                pygame.quit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_DOWN:
                    self.reset_add_reward()
                elif event.key == pygame.K_RIGHT:
                    self.car.go_right()
                elif event.key == pygame.K_UP:
                    self.car.go_up()
                elif event.key == pygame.K_LEFT:
                    self.print_state()

                    
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