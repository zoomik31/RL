import pygame
import torch
import math

BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)
RED = (255, 0, 0)
GREY = (123, 241, 123)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREEN = (0, 255, 0)
PINK = (255,39,203)

class Cell(pygame.sprite.Sprite):
    def __init__(self, x, y, size, color = RED):
        pygame.sprite.Sprite.__init__(self)
        self.size = size
        self.color = color
        self.image = pygame.Surface((self.size, self.size))
        self.image.fill(self.color)
        self.rect = self.image.get_rect()
        self.rect.x = x
        self.rect.y = y

class Border(Cell):
    def __init__(self, x, y, size, color=BLACK):
        super().__init__(x, y, size, color)

class Flag(Cell):
    def __init__(self, x, y, size, color=YELLOW):
        super().__init__(x, y, size, color)
    
    def draw(self, screen):
        w_half, h_half = self.size/2, self.size/2
        vertices = [
            (-w_half, -h_half),
            (w_half, -h_half),
            (w_half, h_half),
            (-w_half, h_half)]
        pygame.draw.polygon(screen, self.color, vertices)

class Empty(Cell):
    def __init__(self, x, y, size, color=WHITE):
        super().__init__(x, y, size, color)

class Tree(Cell):
    def __init__(self, x, y, size, color=GREEN):
       super().__init__(x, y, size, color)

class AddRewards(Cell):
    def __init__(self, x, y, size, color=PINK):
       super().__init__(x, y, size, color)

class Button(pygame.Surface):
    def __init__(self, x, y, text):
        pygame.Surface.__init__(self, (150, 50))
        self.color = WHITE
        self.x = x
        self.y = y
        self.font = pygame.font.Font(None, 24)
        self.text_colour = BLACK
        self.font_antialias = True
        self.text = text
    
    def create_button(self):
        self.fill(self.color)

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

class BackButton(Button):
    def __init__(self, x, y, text):
        super().__init__(x, y, text)

class SaveModelButton(Button):
    def __init__(self, x, y, text):
        super().__init__(x, y, text)
    
    def save_model(self, model):
        torch.save(model.state_dict(), 'model.pth')
        
class DirectionText(pygame.Surface):
    def __init__(self, x, y):
        pygame.Surface.__init__(self, (150, 50))
        self.color = WHITE
        self.x = x
        self.y = y
        self.font = pygame.font.Font(None, 24)
        self.text_colour = BLACK
        self.font_antialias = True
    
    def create_text(self, text):
        self.fill(self.color)

        self.text_dir = self.font.render(text, self.font_antialias, self.text_colour)
        
        self.text_rect = self.text_dir.get_rect(
        center=(self.get_width() /2, 
                self.get_height()/2))
        
        self.dir_text_space = self.get_rect(center=(self.x, self.y))
    
    def draw_dir_text(self, screen):
        self.blit(self.text_dir, self.text_rect)

        screen.blit(self, (self.dir_text_space.x, self.dir_text_space.y))


class Car(pygame.sprite.Sprite):
    def __init__(self, x=0, y=0, size=10, angle=0):
        pygame.sprite.Sprite.__init__(self)
        self.car_size = (size*2, size)
        self.color = RED
        self.start_x = x
        self.start_y = y
        self.start_angle = angle
        self.x = self.start_x
        self.y = self.start_y
        self.size = size
        self.direction = 'stop'
        self.angle = 0
        self.calculate_vertices()

    def restart(self):
        self.rect.x = self.start_x
        self.rect.y = self.start_y
    
    def stand_stil(self):
        self.direction = "stop"
        pass

    def go_right(self):
        self.rotate(-3)

    def go_left(self):
        self.rotate(3)

    def go_down(self):
        self.direction = "down"
        self.rect.y += (self.size)

    def go_up(self):
        pass

    def go_back(self):
        if self.direction == 'right':
            self.rect.x -= self.size
        if self.direction == 'left':
            self.rect.x += self.size
        if self.direction == 'down':
            self.rect.y -= self.size
        if self.direction == 'up':
            self.rect.y += self.size
    
    def rotate(self, degrees):
        """ Поворачивает спрайт на заданный угол """
        self.angle += degrees
        self.calculate_vertices()

    def calculate_vertices(self):
        """ Вычисляет вершины прямоугольника с учётом текущего угла поворота"""
        w_half = self.car_size[0] / 2
        h_half = self.car_size[1] / 2
        center_x = self.x + w_half
        center_y = self.y + h_half
        
        # Углы прямоугольника (до поворота)
        vertices = [
            (-w_half, -h_half),
            (w_half, -h_half),
            (w_half, h_half),
            (-w_half, h_half)
        ]
        
        # Применяем поворот каждой точки
        rad_angle = math.radians(-self.angle)  # отрицательный угол для соответствия правилам координат
        cos_a = math.cos(rad_angle)
        sin_a = math.sin(rad_angle)
        
        for i in range(len(vertices)):
            vx, vy = vertices[i]
            nx = vx * cos_a - vy * sin_a
            ny = vx * sin_a + vy * cos_a
            
            # Перемещаем точку обратно к центру спрайта
            vertices[i] = (nx + center_x, ny + center_y)
        
        self.vertices = vertices
    
    def draw(self, screen):
        """ Рисует повернутый спрайт на экране """
        pygame.draw.polygon(screen, self.color, self.vertices)