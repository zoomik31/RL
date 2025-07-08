import torch as t
import torch.nn as nn
import torch.optim as optim
import random
from Enviroment import *

EPSILON = 0.4
GAMMA = 0.99

class JointDQL(nn.Module):
    def __init__(self, state_dim, action_dim=4, num_agents=2):
        super().__init__()
        self.num_agents = num_agents
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim * num_agents)
        )
        self.memory = []
        self.optimizer = optim.Adam(self.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()

    def forward(self, x):
        return self.net(x)

    def act(self, joint_state, epsilon=EPSILON):
        with t.no_grad():
            q = self.forward(t.FloatTensor(joint_state))
            actions = []
            for i in range(self.num_agents):
                q_agent = q[i*4:(i+1)*4]
                if random.random() < epsilon:
                    actions.append(random.randint(0, 3))
                else:
                    actions.append(q_agent.argmax().item())
            return actions

    def remember(self, joint_state, actions, rewards, next_joint_state):
        self.memory.append((
            t.FloatTensor(joint_state),
            t.LongTensor(actions),
            t.FloatTensor(rewards),
            t.FloatTensor(next_joint_state)
        ))

    def train(self, batch_size=64):
        if len(self.memory) < batch_size: return
        batch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states = zip(*batch)
        states = t.stack(states)
        actions = t.stack(actions)
        rewards = t.stack(rewards)
        next_states = t.stack(next_states)

        q_values = self.forward(states)
        next_q_values = self.forward(next_states).detach()

        loss = 0
        for i in range(self.num_agents):
            q_agent = q_values[:, i * 4:(i + 1) * 4]
            next_q_agent = next_q_values[:, i * 4:(i + 1) * 4]
            q_val = q_agent.gather(1, actions[:, i].unsqueeze(1)).squeeze()
            target = rewards[:, i] + GAMMA * next_q_agent.max(1)[0]
            loss += self.criterion(q_val, target)

        print(loss)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.memory = []

    def save_button_tracking(self):
        for event in pygame.event.get():
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if env.save_button.button_rect.collidepoint(event.pos):
                    env.save_button.save_model(self)
    
if __name__ == "__main__":
    pygame.init()
    screen = pygame.display.set_mode((WIDTH_SCREEN, HEIGHT_SCREEN))
    pygame.display.set_caption("game")
    clock = pygame.time.Clock()
    maps = {
        "map 1": ['E:\VS_project\souless\map_1.xlsx', (420, 395)],
        "map 2": ['E:\VS_project\souless\map_2.xlsx', (420, 455)],
        "map 3": ['E:\VS_project\souless\map_3.xlsx', (420, 515)]
    }
    env = Game(screen, maps)
    env.generate_button()
    agent = None
    episode = 1
    steps = 0
    max_steps = 2500

    while True:
        screen.fill((155, 255, 155))
        clock.tick(FPS)
        env.button_tracking()
        if env.on_mission:
            if agent is None:
                state_dim = len(env.get_joint_state())
                agent = JointDQL(state_dim=state_dim, action_dim=4, num_agents=2)

            joint_state = env.get_joint_state()
            actions = agent.act(joint_state)
            next_states, rewards = env.step(actions)
            next_joint_state = env.get_joint_state()

            agent.remember(joint_state, actions, rewards, next_joint_state)
            episode += 1
            steps += 1
            env.run_game()
            env.back_button.draw_button(screen)
            env.save_button.draw_button(screen)
            

            if episode % 200 == 0:
                print(episode)
                agent.train()

            # --- Ротация ролей и перезапуск эпизода ---
            if any(r >= 100 for r in rewards) or steps >= max_steps:
                env.swap_start_positions()
                for car in env.cars:
                    car.restart()
                steps = 0
            

            # pygame.display.flip()
            # env.back_button.draw_button(screen)
            # env.save_button.draw_button(screen)

        else:
            for btn in env.buttons:
                btn.draw_button(screen)
            pygame.display.flip()
