import math
import pygame

class RotatableSprite(pygame.sprite.Sprite):
    def __init__(self, x, y, width, height, color=(255, 0, 0)):
        super().__init__()
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.color = color
        self.angle = 0  # начальный угол поворота
        self.calculate_vertices()  # вычисляем вершины фигуры

    def calculate_vertices(self):
        """ Вычисляет вершины прямоугольника с учётом текущего угла поворота"""
        w_half = self.width / 2
        h_half = self.height / 2
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

    def rotate(self, degrees):
        """ Поворачивает спрайт на заданный угол """
        self.angle += degrees
        self.calculate_vertices()

# Основная игра
pygame.init()
screen = pygame.display.set_mode((800, 600))
clock = pygame.time.Clock()

sprite = RotatableSprite(x=400, y=300, width=100, height=50)

running = True
while running:
    clock.tick(60)
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
    
    keys = pygame.key.get_pressed()
    if keys[pygame.K_LEFT]:
        sprite.rotate(1)  # поворачиваем налево
    elif keys[pygame.K_RIGHT]:
        sprite.rotate(-1)  # поворачиваем направо
    
    screen.fill((255, 255, 255))  # очищаем экран белым цветом
    sprite.draw(screen)  # рисуем спрайт
    pygame.display.flip()

pygame.quit()