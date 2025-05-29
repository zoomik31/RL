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
        self.fps = 20

        self.empty_space = pygame.sprite.Group()
        self.border = pygame.sprite.Group()
        self.forest = pygame.sprite.Group()

        self.bottom_surface = pygame.surface.Surface((840, 110))

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

                else:
                    self.emp = Empty(x*self.size, y*self.size, self.size)
                    self.empty_space.add(self.emp)
                    layer.append(0)

            self.all_map.append(layer)
        
        self.car = Car(self.screen, 9 * self.size, 35 * self.size, self.size)
        self.dist = math.sqrt((self.car.center_x-self.flag.rect.x)**2 + (self.car.center_y-self.flag.rect.y)**2)
            
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
        self.dir_text = DirectionText(400, 60)
        self.dir_text.create_text(str(abs(round(self.car.speed))))

        self.dir_text.draw_dir_text(screen)


    def barriers_check(self):
        obstacles = pygame.sprite.Group()
        obstacles.add(self.forest)
        obstacles.add(self.border)
        self.car.forward_border = False
        self.car.back_border = False
        for obstacle in obstacles:
            top_left, top_right, bottom_right, bottom_left = self.car.vertices

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

    def sensors_check(self):
        self.objects_nearby = []
        obstacles = pygame.sprite.Group()
        obstacles.add(self.forest)
        obstacles.add(self.border)
        
        for i, sensor in enumerate(self.car.sensors):
            sensor_x, sensor_y = sensor
            for obstacle in obstacles:
                if (
                    (obstacle.rect.x <= sensor_x <= obstacle.rect.x+obstacle.size) and (obstacle.rect.y <= sensor_y <= obstacle.rect.y+obstacle.size)
                    ):
                    dist = self.obstacle_distance(obstacle)
                    self.objects_nearby.append((obstacle.rect.x, obstacle.rect.y, dist))
                    break
            
            if  (
                (self.flag.rect.x <= sensor_x <= self.flag.rect.x+self.flag.size) and (self.flag.rect.y <= sensor_y <= self.flag.rect.y+self.flag.size)
                ):
                dist = self.obstacle_distance(self.flag)
                self.objects_nearby.append((self.flag.rect.x, self.flag.rect.y, dist))
                
            while len(self.objects_nearby) != i+1:
                self.objects_nearby.append((0, 0, 0))    

    def obstacle_distance(self, obstacle):
        return math.sqrt((self.car.center_x-obstacle.rect.center[0])**2 + (self.car.center_y-obstacle.rect.center[1])**2)

    def measure_main_reward_distance(self):
        self.prev_dist = self.dist
        self.dist = math.sqrt((self.car.center_x-self.flag.rect.x)**2 + (self.car.center_y-self.flag.rect.y)**2)
    
    
    def get_state(self):
        self.sensors_check()
        list_state = self.objects_nearby.copy()

        list_state.append((self.car.x, self.car.y, self.car.speed, self.car.angle))
        list_state.append((self.flag.rect.center[0], self.flag.rect.center[1], self.dist))
        
        state = [i for sub in list_state for i in sub]

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
        barrier_check = self.barriers_check()
        while self.barriers_check():
            if self.car.forward_border:
                self.car.move_forward(-1)
            elif self.car.back_border:
                self.car.move_forward(1)
            self.car.calculate_vertices()
        
        self.measure_main_reward_distance()
        if barrier_check:
            self.reward = -130
        elif self.flag_check():
            self.reward = 150
        elif self.dist < self.prev_dist:
            self.reward = 40
        elif self.dist == self.prev_dist:
            self.reward = 0
        else:
            self.reward = 10
        
        self.button_tracking()
        self.border.draw(self.screen)
        self.forest.draw(self.screen)
        self.empty_space.draw(self.screen)

        self.car.draw()
        self.screen.blit(self.flag.image, self.flag.rect)
        self.screen.blit(self.bottom_surface, (0, 841))
        self.bottom_surface.fill(GREEN)
        self.generate_text(self.bottom_surface)
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