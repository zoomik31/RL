from Sprites import *
from Environment import Game as OneAgentGame
from Enviroment import Game as MultyAgentEnvGame
import psycopg2
from model import *
import time
import torch

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
class MainGame():
    def __init__(self, screen):
        self.screen = screen
        self.map_type = ["Зимние дороги", "Внутрихозяйственные", "Лесные дороги", "Общего пользования", "Болото"]
        self.buttons_map = []
        self.num_agent = 0
        self.map_created = False
        self.chosed_type_map = False
        self.on_mission = False
        self.agent_1 = None
        self.episode = 1
        self.steps = 0
        self.max_steps = 2500
        connection = psycopg2.connect(user="postgres",
                                  password="admin",
                                  host="127.0.0.1",
                                  port="5432",
                                  database="map")

        self.cursor = connection.cursor()

    def create_button(self):
        self.one_agent = NumAgent(420, 395, "1 agent")
        self.multyagent = NumAgent(420, 515, "2 agent")
        
        self.map_type_1 = MapType(420, 335, "Зимние дороги")
        self.map_type_2 = MapType(420, 395, "Внутрихозяйственные", size=20)
        self.map_type_3 = MapType(420, 455, "Лесные дороги")
        self.map_type_4 = MapType(420, 515, "Общего пользования", size=20)
        self.map_type_5 = MapType(420, 575, "Болото")
        
        self.one_agent.create_button()
        self.multyagent.create_button()
        
        self.map_type_1.create_button()
        self.map_type_2.create_button()
        self.map_type_3.create_button()
        self.map_type_4.create_button()
        self.map_type_5.create_button()

    def sort_map(self, name):
        self.cursor.execute("SELECT type_id, name from cell_types")
        record = self.cursor.fetchall()
        needed_id = []
        for row in record:
            if name in row[1]:
                needed_id.append(row[0])
        self.maps = []
        self.cursor.execute("SELECT cell_id, map_name from all_map")
        record = self.cursor.fetchall()
        for row in record:
            if row[0] in needed_id:
                self.maps.append(row[1])
        self.map_btns = []
        for i, map in enumerate(self.maps):
            map_btn = MapButton(420, 395+(i+1)*60, map)
            map_btn.create_button()
            self.map_btns.append(map_btn)
        self.map_created = True
        print(record)
        return self.map_created

    def multyagent_learning(self):
        if self.agent_1 is None:
            self.agent_1 = DQL(self.env)
            # torch.load('model.pt', weights_only = False) # model_checkpoint
            self.agent_2 = DQL(self.env, num_layers=13)

        self.agent_1.game_main(torch.load('model.pt', weights_only = False))
        self.agent_2.game_side()
        
        self.env.save_button.draw_button(screen)
        self.env.back_button.draw_button(screen)
        
        if self.env.train_step > 3500:
            if self.env.train_step % 200 == 0:
                self.env.car_1.restart()
                self.env.car_2.restart()  
            elif self.env.train_step % 200 == 0:
                self.agent_2.train()
                print(self.env.train_step)
                self.env.car_1.restart()
                self.env.car_2.restart()
            elif self.env.train_step % 20 == 0:
                self.agent_2.train()
        self.env.train_step += 1

    def dql_learning(self):
        if self.agent is None:
            self.agent = DQL(self.env, num_layers=13)
        self.env.save_button.draw_button(screen)
        self.env.back_button.draw_button(screen)
        self.agent.game()
            
        if self.env.train_step % 20 == 0:
            print(self.env.train_step)
            self.agent.train()
        
        self.env.train_step += 1

    def button_tracking(self):
        for event in pygame.event.get():
            #Закрытие игры
            if event.type == pygame.QUIT:
                pygame.quit()
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if self.one_agent.button_rect.collidepoint(event.pos) and self.num_agent == 0:
                    self.num_agent = 1
                    self.env = OneAgentGame(self.screen)
                elif self.multyagent.button_rect.collidepoint(event.pos) and self.num_agent == 0:
                    self.num_agent = 2
                    self.env = MultyAgentEnvGame(self.screen)
                elif self.map_type_1.button_rect.collidepoint(event.pos) and self.chosed_type_map == False:
                    self.chosed_type_map = True
                    self.sort_map(self.map_type_1.text)
                elif self.map_type_2.button_rect.collidepoint(event.pos) and self.chosed_type_map == False:
                    self.chosed_type_map = True
                    self.sort_map(self.map_type_2.text)
                elif self.map_type_3.button_rect.collidepoint(event.pos) and self.chosed_type_map == False:
                    self.chosed_type_map = True
                    self.sort_map(self.map_type_3.text)
                elif self.map_type_4.button_rect.collidepoint(event.pos) and self.chosed_type_map == False:
                    self.chosed_type_map = True
                    self.sort_map(self.map_type_4.text.lower())
                elif self.map_type_5.button_rect.collidepoint(event.pos) and self.chosed_type_map == False:
                    self.chosed_type_map = True
                    self.sort_map(self.map_type_5.text)
                elif self.chosed_type_map == True:
                    for map in self.map_btns:
                        if map.button_rect.collidepoint(event.pos):
                            self.on_mission = True
                            self.cursor.execute(f"SELECT map, x, y from {map.text}")
                            record = self.cursor.fetchall()
                            self.env.generate_map(record)
            


                
if __name__ == "__main__":
    pygame.init()
    screen = pygame.display.set_mode((WIDTH_SCREEN, HEIGHT_SCREEN))
    pygame.display.set_caption("game")
    clock = pygame.time.Clock()
    main_menu = MainGame(screen)
    main_menu.create_button()
    connection = psycopg2.connect(user="postgres",
                                  password="admin",
                                  host="127.0.0.1",
                                  port="5432",
                                  database="map")
    

    while True:
        main_menu.button_tracking()
        screen.fill((155, 255, 155))
        clock.tick(30)
        if main_menu.num_agent == 0:
            main_menu.one_agent.draw_button(screen)
            main_menu.multyagent.draw_button(screen)
            pygame.display.flip()

        elif main_menu.chosed_type_map == False and main_menu.num_agent != 0:
            main_menu.map_type_1.draw_button(screen)
            main_menu.map_type_2.draw_button(screen)
            main_menu.map_type_3.draw_button(screen)
            main_menu.map_type_4.draw_button(screen)
            main_menu.map_type_5.draw_button(screen)
            pygame.display.flip()
        
        elif main_menu.chosed_type_map == True and main_menu.on_mission == False:
            for map in main_menu.map_btns:
                map.draw_button(screen)
            pygame.display.flip()
        
        elif main_menu.on_mission and main_menu.num_agent == 2:
            main_menu.multyagent_learning()
        
        elif main_menu.on_mission and main_menu.num_agent == 1:
            main_menu.dql_learning()
