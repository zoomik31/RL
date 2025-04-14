import pygame

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
