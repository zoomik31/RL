import pygame
import torch

BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)
RED = (255, 0, 0)
GREY = (123, 241, 123)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREEN = (0, 255, 0)

class Cell(pygame.sprite.Sprite):
    def __init__(self, x, y, size, colour = RED):
        pygame.sprite.Sprite.__init__(self)
        self.obstacles_size = (size, size)
        self.colour = colour
        self.image = pygame.Surface(self.obstacles_size)
        self.image.fill(self.colour)
        self.rect = self.image.get_rect()
        self.rect.x = x
        self.rect.y = y


class Border(Cell):
    def __init__(self, x, y, size, colour=BLACK):
        super().__init__(x, y, size, colour)

class Flag(Cell):
    def __init__(self, x, y, size, colour=YELLOW):
        super().__init__(x, y, size, colour)

class Empty(Cell):
    def __init__(self, x, y, size, colour=WHITE):
        super().__init__(x, y, size, colour)

class Tree(Cell):
    def __init__(self, x, y, size, colour=GREEN):
       super().__init__(x, y, size, colour)

class Button(pygame.Surface):
    def __init__(self, x, y, text):
        pygame.Surface.__init__(self, (150, 50))
        self.colour = WHITE
        self.x = x
        self.y = y
        self.font = pygame.font.Font(None, 24)
        self.text_colour = BLACK
        self.font_antialias = True
        self.text = text
    
    def create_button(self):
        self.fill(self.colour)

        self.text_map = self.font.render(self.text, self.font_antialias, self.text_colour)
        
        self.text_rect = self.text_map.get_rect(
        center=(self.get_width() /2, 
                self.get_height()/2))
        
        self.button_rect = self.get_rect(center=(self.x, self.y))
    
    def draw_button(self, screen):
        self.blit(self.text_map, self.text_rect)

        screen.blit(self, (self.button_rect.x, self.button_rect.y))


class MapButton(Button):
    def __init__(self, x, y, text, map):
        super().__init__(x, y, text)
        self.map = map

class SwitchButton(Button):
    def __init__(self, x, y, text):
        super().__init__(x, y, text)

class SaveModelButton(Button):
    def __init__(self, x, y, text):
        super().__init__(x, y, text)
    
    def save(self, model, name):
        print('model save')
        torch.save(model, (f'{name}.pth'))
        
class Car(pygame.sprite.Sprite):
    def __init__(self, x=0, y=0, size=10):
        pygame.sprite.Sprite.__init__(self)
        self.car_size = (size, size)
        self.image = pygame.Surface(self.car_size)
        self.image.fill(RED)
        self.rect = self.image.get_rect()
        self.start_x = x
        self.start_y = y
        self.rect.x = self.start_x
        self.rect.y = self.start_y
        self.size = size
        self.direction = 'stop'

    def restart(self):
        self.rect.x = self.start_x
        self.rect.y = self.start_y
    
    def stand_stil(self):
        self.direction = "stop"
        pass

    def go_right(self):
        self.direction = "right"
        self.rect.x += (self.size)

    def go_left(self):
        self.direction = "left"
        self.rect.x -= (self.size)
        
    def go_down(self):
        self.direction = "down"
        self.rect.y += (self.size)

    def go_up(self):
        self.direction = "up"
        self.rect.y -= (self.size)

    def go_back(self):
        if self.direction == 'right':
            self.rect.x -= self.size
        if self.direction == 'left':
            self.rect.x += self.size
        if self.direction == 'down':
            self.rect.y -= self.size
        if self.direction == 'up':
            self.rect.y += self.size
