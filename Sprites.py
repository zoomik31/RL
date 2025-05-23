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

class Empty(Cell):
    def __init__(self, x, y, size, color=WHITE):
        super().__init__(x, y, size, color)

class Tree(Cell):
    def __init__(self, x, y, size, color=GREEN):
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
    def __init__(self, screen, x=0, y=0, size=10, angle=0):
        pygame.sprite.Sprite.__init__(self)
        self.car_size = (size*2, size)
        self.color = RED
        self.start_x = x
        self.start_y = y
        self.start_angle = angle
        self.x = self.start_x
        self.y = self.start_y
        self.size = size
        self.direction = 'Stop'
        self.forward_border = False
        self.back_border = False
        self.angle = 0
        self.speed = 0
        self.sensors_color = PINK
        self.sensors_len = self.car_size[1]
        
        self.screen = screen
        self.calculate_vertices()

    def restart(self):
        self.x = self.start_x
        self.y = self.start_y
        self.angle = self.start_angle
        self.speed = 0
        self.calculate_vertices()
    
    def stand_stil(self):
        pass

    def emergency_braking(self):
        self.speed = 0

    def go_right(self):
        if -1 < self.speed < 1:
            pass
        elif self.speed > 0:
            self.rotate(-5)
        else:
            self.rotate(5)

    def go_left(self):
        if -1 < self.speed < 1:
            pass
        elif self.speed > 0:
            self.rotate(5)
        else:
            self.rotate(-5)

    def go_down(self):
        self.move_forward(-1)

    def go_up(self):
        self.move_forward(2)

    def go_back(self):
        pass
    
    def rotate(self, degrees):
        """ Поворачивает спрайт на заданный угол """
        self.angle += degrees
        self.calculate_vertices()

    def calculate_vertices(self):
        """ Вычисляет вершины прямоугольника с учётом текущего угла поворота"""
        self.w_half = self.car_size[0] / 2
        self.h_half = self.car_size[1] / 2
        self.center_x = self.x + self.w_half
        self.center_y = self.y + self.h_half
        
        # Углы прямоугольника (до поворота)
        vertices = [
            (-self.w_half, -self.h_half),
            (self.w_half, -self.h_half),
            (self.w_half, self.h_half),
            (-self.w_half, self.h_half)
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
            vertices[i] = (nx + self.center_x, ny + self.center_y)
        
        self.vertices = vertices
        self.calculate_sensors()
    
    def calculate_sensors(self):
        vertexes = []
        vertexes.append((self.w_half, (int((self.w_half+self.w_half)+(-self.h_half+self.h_half)))/2))
        vertexes.append(self.vertices[1])
        vertexes.append(((int((-self.w_half+self.w_half)+(-self.h_half-self.h_half)))/2, self.h_half))
        vertexes.append(self.vertices[0])
        vertexes.append((-self.w_half, (int((-self.w_half-self.w_half)+(-self.h_half+self.h_half)))/2))
        vertexes.append(self.vertices[3])
        vertexes.append(((int((-self.w_half + self.w_half)+(self.h_half+self.h_half)))/2, -self.h_half))
        vertexes.append(self.vertices[2])

        for i, vertex in enumerate(vertexes):
            radians = math.radians((self.angle+180) + i*45)
            rel_x = (vertex[0] + self.sensors_len)- vertex[0]
            rel_y = (vertex[1] + self.sensors_len)-vertex[0]

            new_rel_x = rel_x * math.sin(radians) - rel_y * math.cos(radians)
            new_rel_y = rel_x * math.cos(radians) + rel_y * math.sin(radians)

            # Нормализация вектора направления и умножение на длину луча
            # length = math.sqrt(dir_x ** 2 + dir_y ** 2)
            # norm_dir_x = dir_x / 10
            # norm_dir_y = dir_y / 10

            # Конечная точка луча
            end_point = (
                vertex[0] + (new_rel_x),
                vertex[1] + new_rel_y
            )
            self.draw_sensors(vertex, end_point)

    def draw_sensors(self, vertex, end_point):
        pygame.draw.line(self.screen, self.sensors_color, vertex, end_point, 3)



    def move_forward(self, speed = 3):
        """ Движение вперёд под текущим углом поворота."""
        if round(self.speed, 2) == 0:
            self.calculate_vertices()
            pass
        elif round(self.speed, 2) < 0.33:
            self.speed += speed/9
            rad_angle = math.radians(self.angle)
            dx = self.speed * math.cos(rad_angle)
            dy = self.speed * math.sin(rad_angle)
            
            # Изменяем позицию спрайта согласно компонентам скорости
            self.x += dx
            self.y -= dy  # Минус потому что ось Y направлена сверху вниз
            self.calculate_vertices()  # Перерассчитываем вершины после смещения


        elif speed == 3:
            self.speed += -speed/9
            rad_angle = math.radians(self.angle)
            dx = self.speed * math.cos(rad_angle)
            dy = self.speed * math.sin(rad_angle)
            
            # Изменяем позицию спрайта согласно компонентам скорости
            self.x += dx
            self.y -= dy  # Минус потому что ось Y направлена сверху вниз
            self.calculate_vertices()  # Перерассчитываем вершины после смещения
        
        if speed == 2 or speed == 1:
            self.speed += speed
            rad_angle = math.radians(self.angle)
            dx = self.speed * math.cos(rad_angle)
            dy = self.speed * math.sin(rad_angle)
            
            # Изменяем позицию спрайта согласно компонентам скорости
            self.x += dx
            self.y -= dy  # Минус потому что ось Y направлена сверху вниз
            self.calculate_vertices()  # Перерассчитываем вершины после смещения
        
        elif speed == -1:
            self.speed += speed
            rad_angle = math.radians(self.angle)
            dx = self.speed * math.cos(rad_angle)
            dy = self.speed * math.sin(rad_angle)
            
            # Изменяем позицию спрайта согласно компонентам скорости
            self.x += dx
            self.y -= dy  # Минус потому что ось Y направлена сверху вниз
            self.calculate_vertices()  # Перерассчитываем вершины после смещения
        print(self.speed)
    
    
    def draw(self):
        """ Рисует повернутый спрайт на экране """
        self.move_forward()
        pygame.draw.polygon(self.screen, self.color, self.vertices)

   