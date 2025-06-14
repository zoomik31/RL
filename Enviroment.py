import pygame
import math
import pandas as pd
import psycopg2
from Sprites import *

ORANGE = (255,140,0)
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

class Game():
    def __init__(self, screen):
        self.empty_space = pygame.sprite.Group()
        self.border = pygame.sprite.Group()
        self.forest = pygame.sprite.Group()
        self.roads = pygame.sprite.Group()
        self.cars = pygame.sprite.Group()
        self.start_cells = pygame.sprite.Group()
        self.snow_cells = pygame.sprite.Group()
        self.snowdrifts = pygame.sprite.Group()
        self.divinglines = pygame.sprite.Group()
        self.puddles = pygame.sprite.Group()
        self.size = 20
        self.screen = screen
        self.on_mission = False
        self.num_agents = 2

    def generate_map(self, map):
        car_created = 0
        for row in map:
            if row[0] == 1:
                brd = Border(row[2]*self.size, row[1]*self.size, self.size)
                self.border.add(brd)
            
            elif row[0] == 2:
                self.flag = Flag(row[2]*self.size, row[1]*self.size, self.size)

            elif row[0] == 3:
                tree = Tree(row[2]*self.size, row[1]*self.size, self.size)
                self.forest.add(tree)

            elif row[0] == 4:
                road = Road(row[2]*self.size, row[1]*self.size, self.size)
                self.roads.add(road)

            elif row[0] == 5:
                car_created +=1
                car = Car(row[2]*self.size, row[1]*self.size, self.size, RED)
                self.cars.add(car)
                start_cell = Start(row[2]*self.size, row[1]*self.size, self.size)
                self.start_cells.add(start_cell)

            elif row[0] == 6:
                snow = Snow(row[2]*self.size, row[1]*self.size, self.size)
                self.snow_cells.add(snow)
            
            elif row[0] == 7:
                snowdrift = SnowDrift(row[2]*self.size, row[1]*self.size, self.size)
                self.snowdrifts.add(snowdrift)
            
            elif row[0] == 8:
                div_line = DivingLine(row[2]*self.size, row[1]*self.size, self.size)
                self.divinglines.add(div_line)
            
            elif row[0] == 9:
                puddle = Puddle(row[2]*self.size, row[1]*self.size, self.size)
                self.puddles.add(puddle)
            else:
                self.emp = Empty(row[2]*self.size, row[1]*self.size, self.size)
                self.empty_space.add(self.emp)

        # Два агента разного цвета
        if car_created == 0:
            print('hi')
            x, y = 9, 35
            car = Car(x*self.size, y*self.size, self.size, RED)
            self.cars.add(car)
            car = Car((x+2)*self.size, y*self.size, self.size, ORANGE)
            self.cars.add(car)

            start_cell = Start(x*self.size, y*self.size, self.size)
            self.start_cells.add(start_cell)
            start_cell = Start((x+2)*self.size, y*self.size, self.size)
            self.start_cells.add(start_cell)

        if car_created == 1:
            car = Car(car.rect.x, car.rect.y - self.size, self.size, ORANGE)
            self.cars.add(car)

            start_cell = Start(car.rect.x, car.rect.y - self.size, self.size)
            self.start_cells.add(start_cell)

        self.dists = [self.calculate_distance(car) for car in self.cars]
        self.prev_dists = self.dists.copy()

        self.generate_button()

    def calculate_distance(self, car):
        return math.sqrt((car.rect.x-self.flag.rect.x)**2 + (car.rect.y-self.flag.rect.y)**2)

    def generate_button(self):
        self.back_button = BackButton(200, 900, "choise map")
        self.save_button = SaveModelButton(600, 900, "save model")

        self.back_button.create_button()
        self.save_button.create_button()
    
    def barriers_check(self, idx):
        car = self.cars.sprites()[idx]
        # Проверка на столкновение
        if pygame.sprite.spritecollideany(self.car, self.forest) or pygame.sprite.spritecollideany(self.car, self.border) or pygame.sprite.spritecollideany(self.car, self.snowdrifts) or pygame.sprite.spritecollideany(self.car, self.puddles):
            return True
        # Проверка столкновения с другим агентом
        for i, other in enumerate(self.cars):
            if i != idx and car.rect.colliderect(other.rect):
                return True
        return False

    def flag_check(self, idx):
        car = self.cars.sprites()[idx]
        if pygame.sprite.collide_rect(car, self.flag):
            car.restart()
            return True
        return False

    def get_joint_state(self):
        # Объединяем состояния обоих агентов в один вектор
        joint_state = []
        for idx, car in enumerate(self.cars):
            state = self.get_state(idx, car)
            # Добавляем координаты флага
            state.append(self.flag.rect.x / WIDTH)
            state.append(self.flag.rect.y / HEIGHT)
            # Добавляем координаты другого агента
            other = self.cars.sprites()[1 - idx]
            state.append(other.rect.x / WIDTH)
            state.append(other.rect.y / HEIGHT)
            state.append(idx)
            joint_state += state
        return joint_state


    def get_state(self, idx, car):
        car = car
        state = []
        pos = car.rect.topleft  # Сохраняем исходную позицию

        # Проверяем возможность движения во всех направлениях
        for move in [car.go_right, car.go_left, car.go_down, car.go_up]:
            move()
            blocked = self.barriers_check(idx)
            car.rect.topleft = pos  # Возвращаем машину назад
            state.append(int(blocked))

        state += [
            int(car.rect.x < self.flag.rect.x),
            int(car.rect.y < self.flag.rect.y),
            int(car.rect.x > self.flag.rect.x),
            int(car.rect.y > self.flag.rect.y)
        ]
        state += [
            int(car.direction == "up"),
            int(car.direction == "right"),
            int(car.direction == "down"),
            int(car.direction == "left")
        ]
        return state

    def swap_start_positions(self):
        cars = list(self.cars)
        # Сохраняем стартовые позиции
        pos0 = cars[0].start_x, cars[0].start_y
        pos1 = cars[1].start_x, cars[1].start_y
        # Меняем местами
        cars[0].start_x, cars[0].start_y = pos1
        cars[1].start_x, cars[1].start_y = pos0
        # Перемещаем физически
        cars[0].rect.topleft = pos1
        cars[1].rect.topleft = pos0

    def step(self, actions):
        rewards = []
        next_states = []
        barier_checked = []
        self.prev_dists = self.dists.copy()
        # Делаем ходы для обоих агентов
        for idx, action in enumerate(actions):
            car = self.cars.sprites()[idx]
            old_pos = car.rect.topleft
            if action == 0:
                car.go_up()
            elif action == 1:
                car.go_right()
            elif action == 2:
                car.go_down()
            elif action == 3:
                car.go_left()
            barier_checked.append(self.barriers_check(idx))
            if barier_checked[idx]:
                car.rect.topleft = old_pos

        for idx in range(self.num_agents):
            car = self.cars.sprites()[idx]
            reward = 0
            prev_dist = self.prev_dists[idx]
            new_dist = self.calculate_distance(car)
            self.dists[idx] = new_dist
            if barier_checked[idx]:
                reward = -100
            elif self.flag_check(idx):
                reward = 100
            elif new_dist < prev_dist:
                reward = 50
            elif new_dist == prev_dist:
                reward = 0
            else:
                reward = -15
            rewards.append(reward)
            next_states.append(self.get_state(idx, car))

        # --- КОЛЛЕКТИВНАЯ НАГРАДА ЗА СЛЕДОВАНИЕ ---
        car0 = self.cars.sprites()[0]
        car1 = self.cars.sprites()[1]
        dist = math.sqrt((car0.rect.x - car1.rect.x) ** 2 + (car0.rect.y - car1.rect.y) ** 2)
        if dist < 2 * self.size:
            rewards = [r + 25 for r in rewards]
        else:
            rewards = [r - 15 for r in rewards]
        # Штраф если второй агент обгоняет первого (по расстоянию до флага)
        dist0 = self.calculate_distance(car0)
        dist1 = self.calculate_distance(car1)
        if dist1 < dist0 - self.size:
            rewards[1] -= 20


        self.run_game()
        return next_states, rewards

    def run_game(self):
        self.screen.fill((155, 255, 155))
        self.border.draw(self.screen)
        self.forest.draw(self.screen)
        self.roads.draw(self.screen)
        self.empty_space.draw(self.screen)
        self.snow_cells.draw(self.screen)
        self.snowdrifts.draw(self.screen)
        self.divinglines.draw(self.screen)
        self.puddles.draw(self.screen)
        self.start_cells.draw(self.screen)
        self.screen.blit(self.flag.image, self.flag.rect)
        for car in self.cars:
            self.screen.blit(car.image, car.rect)
        self.save_button.draw_button(self.screen)
        self.back_button.draw_button(self.screen)
        pygame.display.flip()

    def button_tracking(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                for btn in self.buttons:
                    if btn.button_rect.collidepoint(event.pos) and not self.on_mission:
                        self.on_mission = True
                        self.generate_map(btn.map)
                 
                if self.back_button.button_rect.collidepoint(event.pos) and self.on_mission:
                    self.on_mission = False