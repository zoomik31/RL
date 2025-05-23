import pygame
import random
import math

# Простая машина с возможностью поворота
class SimpleCar:
    def __init__(self, x, y, width, height):
        self.x = x                   # Центр массы
        self.y = y
        self.width = width           # Ширина машины
        self.height = height         # Высота машины
        self.angle = 0               # Текущий угол поворота
        self.color = (255, 0, 0)    # Красный цвет машины
        self.ray_length = 200       # Длина луча

    def update(self):
        # Здесь логика перемещения и поворота машины
        pass

    def draw(self, screen):
        # Преобразование координат вершин с учётом угла поворота
        half_w = self.width / 2
        half_h = self.height / 2
        vertices = [
            (-half_w, -half_h),
            (half_w, -half_h),
            (half_w, half_h),
            (-half_w, half_h)
        ]

        # Поворачиваем каждую вершину
        rad_angle = math.radians(self.angle)
        cos_a = math.cos(rad_angle)
        sin_a = math.sin(rad_angle)
        transformed_vertices = []
        for vertex in vertices:
            px, py = vertex
            nx = px * cos_a - py * sin_a
            ny = px * sin_a + py * cos_a
            transformed_vertices.append((nx + self.x, ny + self.y))

        # Рисуем машину
        pygame.draw.polygon(screen, self.color, transformed_vertices)

        # Дополнительный метод для рисования лучей
        self.draw_rays_from_vertices(screen, transformed_vertices)

    def draw_rays_from_vertices(self, screen, vertices):
        # Расстояние от центра машины до каждой вершины
        ray_color = (0, 0, 255)  # Голубой цвет для лучей
        for vertex in vertices:
            # Направление луча совпадает с направлением от центра машины к вершине
            dir_x = vertex[0] - self.x
            dir_y = vertex[1] - self.y

            # Нормализация вектора направления и умножение на длину луча
            length = math.sqrt(dir_x ** 2 + dir_y ** 2)
            norm_dir_x = dir_x / length
            norm_dir_y = dir_y / length

            # Конечная точка луча
            end_point = (
                vertex[0] + norm_dir_x * self.ray_length,
                vertex[1] + norm_dir_y * self.ray_length
            )

            # Рисуем линию (луч) от вершины до конца
            pygame.draw.line(screen, ray_color, vertex, end_point, 2)

    def check_collision_with_obstacles(self, obstacles):
        # Получаем актуальные вершины машины
        vertices = self.get_vertices()

        # Полигональная проверка столкновений с препятствиями
        for obstacle in obstacles:
            # Если любая точка машины находится внутри препятствия или наоборот
            if any(point_inside_rect(vertex, obstacle.rect) for vertex in vertices):
                return True
        return False

    def get_vertices(self):
        # Аналогичен процедуре из метода draw(), возвращает список вершин
        half_w = self.width / 2
        half_h = self.height / 2
        vertices = [
            (-half_w, -half_h),
            (half_w, -half_h),
            (half_w, half_h),
            (-half_w, half_h)
        ]

        rad_angle = math.radians(self.angle)
        cos_a = math.cos(rad_angle)
        sin_a = math.sin(rad_angle)
        transformed_vertices = []
        for vertex in vertices:
            px, py = vertex
            nx = px * cos_a - py * sin_a
            ny = px * sin_a + py * cos_a
            transformed_vertices.append((nx + self.x, ny + self.y))
        return transformed_vertices

# Проверка попадания точки внутрь прямоугольника
def point_inside_rect(point, rect):
    x, y = point
    return (
        rect.left <= x <= rect.right and
        rect.top <= y <= rect.bottom
    )

# Обычный класс препятствия
class Obstacle(pygame.sprite.Sprite):
    def __init__(self, x, y, size):
        super().__init__()
        self.image = pygame.Surface((size, size))
        self.image.fill((0, 255, 0))
        self.rect = self.image.get_rect(center=(x, y))

# Настройки окна
pygame.init()
screen = pygame.display.set_mode((800, 600))
clock = pygame.time.Clock()

# Группа препятствий
obstacle_group = pygame.sprite.Group()
for _ in range(50):
    obstacle = Obstacle(random.randrange(0, 800), random.randrange(0, 600), 50)
    obstacle_group.add(obstacle)

# Машина
car = SimpleCar(x=400, y=300, width=70, height=30)

# Игровой цикл
running = True
while running:
    clock.tick(60)
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Проверка столкновений
    collision_occurred = car.check_collision_with_obstacles(obstacle_group.sprites())
    if collision_occurred:
        print("Машина столкнулась с препятствием!")

    # Обновляем состояние машины
    car.update()

    # Визуализация
    screen.fill((255, 255, 255))
    obstacle_group.draw(screen)
    car.draw(screen)
    pygame.display.flip()

pygame.quit()