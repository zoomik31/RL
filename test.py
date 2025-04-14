import pygame
import math
import random
import torch
import pandas as pd

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
        position = []
        self.empty_space = pygame.sprite.Group()
        self.border = pygame.sprite.Group()
        self.forest = pygame.sprite.Group()
        self.size = WIDTH/len(self.map)
        self.screen = screen
    
    def draw(self):
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

    def game(self):
        self.border.draw(self.screen)
        self.forest.draw(self.screen)
        self.empty_space.draw(self.screen)
        self.screen.blit(self.flag.image, self.flag.rect)
        pygame.display.flip() 

if __name__ == "__main__":
    pygame.init()
    map = 'E:\VS_project\souless\map_1.xlsx'
    screen = pygame.display.set_mode((WIDTH_SCREEN, HEIGHT_SCREEN))
    pygame.display.set_caption("game")
    clock = pygame.time.Clock()
    game = Game(screen, map)
    game.draw()
    while True:
        screen.fill((155, 255, 155))
        clock.tick(FPS) 
        game.game()
