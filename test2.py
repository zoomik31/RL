import pygame
import math

# Инициализируем окно
pygame.init()
screen = pygame.display.set_mode((800, 600))
clock = pygame.time.Clock()

# Цвета
BLACK = (0, 0, 0)
RED = (255, 0, 0)

# Начальная и конечная точки вектора
vector_start = (200, 300)
vector_end = (400, 300)

# Угол поворота в градусах
angle_in_degrees = 45

def rotate_vector(start, end, angle):
    """ Функция для поворота вектора """
    # Переводим угол из градусов в радианы
    theta = math.radians(angle)
    
    # Рассчитываем дельту координат
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    
    # Формула поворота
    new_dx = dx * math.cos(theta) - dy * math.sin(theta)
    new_dy = dx * math.sin(theta) + dy * math.cos(theta)
    
    # Новая конечная точка
    new_end = (
        start[0] + int(new_dx),
        start[1] + int(new_dy)
    )
    return new_end

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
    
    # Очистка экрана
    screen.fill(BLACK)
    
    # Исходная линия
    pygame.draw.line(screen, RED, vector_start, vector_end, 3)
    
    # Новый конец линии после поворота
    rotated_end = rotate_vector(vector_start, vector_end, angle_in_degrees)
    
    # Нарисовать новую линию
    pygame.draw.line(screen, (0, 255, 0), vector_start, rotated_end, 3)
    
    # Обновление дисплея
    pygame.display.update()
    clock.tick(60)

pygame.quit()