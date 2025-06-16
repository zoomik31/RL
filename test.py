import pygame
import math
import pandas as pd
from Environment import *
from model import *

def game(env, model):
    state = env.get_state_main()
    answer = model.forward((torch.FloatTensor(state)))
    action = torch.argmax(answer).item()
    
    next_state, reward = env.step_main(action)

if __name__ == "__main__":
    pygame.init()
    maps = {"map 1": ['E:\VS_project\souless\map_1.xlsx', (420, 395)], 
            "map 2": ['E:\VS_project\souless\map_2.xlsx', (420, 455)],
            "map 3": ['E:\VS_project\souless\map_3.xlsx', (420, 515)]}
    screen = pygame.display.set_mode((WIDTH_SCREEN, HEIGHT_SCREEN))
    pygame.display.set_caption("game")
    clock = pygame.time.Clock()
    env = Game(screen, maps)
    # train_func = DQL()
    # model.load_state_dict(torch.load('model.pt'))
    agent_1 = torch.load('model.pt', weights_only = False) # model_checkpoint
    agent_2 = DQL(env, num_layers=36)
    # model.eval()
    env.generate_button()

    while True:
        screen.fill((155, 255, 155))
        clock.tick(FPS) 
        
        env.button_tracking(agent_1)
        
        if env.on_mission:
            env.back_button.draw_button(screen)
            env.save_button.draw_button(screen)
            game(env, agent_1)
            agent_2.game()

            if env.train_step > 3500:
                if env.train_step % 200 == 0:
                    env.car_1.restart()
                    env.car_2.restart()  
                # agent_2.train()
                # print(env.train_step)
                # env.car_1.restart()
                # env.car_2.restart()
            elif env.train_step % 200 == 0:
                agent_2.train()
                print(env.train_step)
                env.car_1.restart()
                env.car_2.restart()
            elif env.train_step % 20 == 0:
                agent_2.train()

            

            env.train_step += 1
        else:
            if env.train_step > 1:
                train_func.rollback()
            env.map_button_1.draw_button(screen)
            env.map_button_2.draw_button(screen)
            env.map_button_3.draw_button(screen)
            pygame.display.flip()