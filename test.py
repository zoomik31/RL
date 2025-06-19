import pygame
import math
import pandas as pd
from Environment import *
from model import *

def game(env, model):
    state = env.get_state()
    answer = model.forward((torch.FloatTensor(state)))
    action = torch.argmax(answer).item()
    
    next_state, reward = env.step(action)

if __name__ == "__main__":
    pygame.init()
    maps = {"map 1": ['E:\VS_project\souless\map_1.xlsx', (420, 395)], 
            "map 2": ['E:\VS_project\souless\map_2.xlsx', (420, 455)],
            "map 3": ['E:\VS_project\souless\map_3.xlsx', (420, 515)]}
    screen = pygame.display.set_mode((WIDTH_SCREEN, HEIGHT_SCREEN))
    pygame.display.set_caption("game")
    clock = pygame.time.Clock()
    env = Game(screen, maps)
    train_func = DQL(env)
    # model.load_state_dict(torch.load('model.pt'))
    model = torch.load('model_best.pt', weights_only=False) # model_checkpoint
    # model.eval()
    env.generate_button()

    while True:
        screen.fill((155, 255, 155))
        clock.tick(FPS) 
        
        env.button_tracking(model)
        
        if env.on_mission:
            env.back_button.draw_button(screen)
            env.save_button.draw_button(screen)
            game(env, model)

            if env.train_step % 500 == 0:
                print(env.train_step)
                env.car.restart()

            env.train_step += 1
        else:
            if env.train_step > 1:
                train_func.rollback()
            env.map_button_1.draw_button(screen)
            env.map_button_2.draw_button(screen)
            env.map_button_3.draw_button(screen)
            pygame.display.flip()