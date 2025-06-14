import pygame
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from collections import deque
import json
import os

# Initialize Pygame
pygame.init()

# Game Constants
WINDOW_WIDTH = 800
WINDOW_HEIGHT = 600
GRID_SIZE = 20
GRID_WIDTH = WINDOW_WIDTH // GRID_SIZE
GRID_HEIGHT = WINDOW_HEIGHT // GRID_SIZE

# Colors
BLACK = (0, 0, 0)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
WHITE = (255, 255, 255)
BLUE = (0, 0, 255)

# Directions
UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3

class SnakeGame:
    def __init__(self):
        self.reset()
        
    def reset(self):
        """Reset the game to initial state"""
        self.snake = [(GRID_WIDTH//2, GRID_HEIGHT//2)]
        self.direction = RIGHT
        self.food = self.generate_food()
        self.score = 0
        self.game_over = False
        self.steps_without_food = 0
        return self.get_state()
    
    def generate_food(self):
        """Generate food at random position not occupied by snake"""
        while True:
            food = (random.randint(0, GRID_WIDTH-1), random.randint(0, GRID_HEIGHT-1))
            if food not in self.snake:
                return food
    
    def get_state(self):
        """Get current game state as numpy array for AI"""
        head_x, head_y = self.snake[0]
        food_x, food_y = self.food
        
        # Danger detection (collision in next step for each direction)
        danger_straight = self.is_collision(self.get_next_head(self.direction))
        danger_right = self.is_collision(self.get_next_head((self.direction + 1) % 4))
        danger_left = self.is_collision(self.get_next_head((self.direction - 1) % 4))
        
        # Direction booleans
        dir_up = self.direction == UP
        dir_right = self.direction == RIGHT
        dir_down = self.direction == DOWN
        dir_left = self.direction == LEFT
        
        # Food location relative to head
        food_up = food_y < head_y
        food_down = food_y > head_y
        food_left = food_x < head_x
        food_right = food_x > head_x
        
        state = [
            # Danger
            danger_straight,
            danger_right,
            danger_left,
            
            # Direction
            dir_up,
            dir_right,
            dir_down,
            dir_left,
            
            # Food location
            food_up,
            food_down,
            food_left,
            food_right
        ]
        
        return np.array(state, dtype=np.float32)
    
    def get_next_head(self, direction):
        """Get next head position based on direction"""
        head_x, head_y = self.snake[0]
        if direction == UP:
            return (head_x, head_y - 1)
        elif direction == RIGHT:
            return (head_x + 1, head_y)
        elif direction == DOWN:
            return (head_x, head_y + 1)
        elif direction == LEFT:
            return (head_x - 1, head_y)
    
    def is_collision(self, pos):
        """Check if position causes collision"""
        x, y = pos
        # Wall collision
        if x < 0 or x >= GRID_WIDTH or y < 0 or y >= GRID_HEIGHT:
            return True
        # Snake collision
        if pos in self.snake:
            return True
        return False
    
    def step(self, action):
        """Execute one game step with given action"""
        # Action: 0=straight, 1=right turn, 2=left turn
        if action == 1: 
            self.direction = (self.direction + 1) % 4
        elif action == 2: 
            self.direction = (self.direction - 1) % 4
        
        # Move snake
        new_head = self.get_next_head(self.direction)
        
        # Check collision
        if self.is_collision(new_head):
            self.game_over = True
            reward = -10
            return self.get_state(), reward, self.game_over
        
        self.snake.insert(0, new_head)
        
        # Check if food eaten
        if new_head == self.food:
            self.score += 1
            self.food = self.generate_food()
            reward = 10
            self.steps_without_food = 0
        else:
            self.snake.pop()
            reward = 0
            self.steps_without_food += 1
        
        # Penalty for taking too long without eating
        if self.steps_without_food > 100 * len(self.snake):
            self.game_over = True
            reward = -10
        
        return self.get_state(), reward, self.game_over

class DQN(nn.Module):
    def __init__(self, input_size=11, hidden_size=256, output_size=3):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class DQNAgent:
    def __init__(self, state_size=11, action_size=3, lr=0.001):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=100000)
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = lr
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Neural networks
        self.q_network = DQN(state_size, 256, action_size).to(self.device)
        self.target_network = DQN(state_size, 256, action_size).to(self.device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        
        # Copy weights to target network
        self.update_target_network()
        
    def update_target_network(self):
        """Copy weights from main network to target network"""
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay memory"""
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state):
        """Choose action using epsilon-greedy policy"""
        if np.random.random() <= self.epsilon:
            return random.randrange(self.action_size)
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        q_values = self.q_network(state_tensor)
        return np.argmax(q_values.cpu().data.numpy())
    
    def replay(self, batch_size=32):
        """Train the model on a batch of experiences"""
        if len(self.memory) < batch_size:
            return
        
        batch = random.sample(self.memory, batch_size)
        states = torch.FloatTensor([e[0] for e in batch]).to(self.device)
        actions = torch.LongTensor([e[1] for e in batch]).to(self.device)
        rewards = torch.FloatTensor([e[2] for e in batch]).to(self.device)
        next_states = torch.FloatTensor([e[3] for e in batch]).to(self.device)
        dones = torch.BoolTensor([e[4] for e in batch]).to(self.device)
        
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        next_q_values = self.target_network(next_states).max(1)[0].detach()
        target_q_values = rewards + (0.95 * next_q_values * ~dones)
        
        loss = F.mse_loss(current_q_values.squeeze(), target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def save(self, filename):
        """Save the model"""
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon
        }, filename)
    
    def load(self, filename):
        """Load the model"""
        if os.path.exists(filename):
            checkpoint = torch.load(filename, map_location=self.device)
            self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
            self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.epsilon = checkpoint['epsilon']
            print(f"Model loaded from {filename}")

def train_agent(episodes=1000):
    """Train the DQN agent"""
    game = SnakeGame()
    agent = DQNAgent()
    scores = []
    
    print("Training AI Agent...")
    print("Episode | Score | Epsilon | Avg Score (last 100)")
    print("-" * 50)
    
    for episode in range(episodes):
        state = game.reset()
        total_reward = 0
        
        while not game.game_over:
            action = agent.act(state)
            next_state, reward, done = game.step(action)
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            
            if done:
                break
        
        scores.append(game.score)
        agent.replay()
        
        # Update target network every 100 episodes
        if episode % 100 == 0:
            agent.update_target_network()
        
        # Print progress
        if episode % 100 == 0:
            avg_score = np.mean(scores[-100:]) if len(scores) >= 100 else np.mean(scores)
            print(f"{episode:7} | {game.score:5} | {agent.epsilon:.3f} | {avg_score:.2f}")
    
    # Save the trained model
    agent.save('snake_dqn_model.pth')
    
    # Save training statistics
    with open('training_scores.json', 'w') as f:
        json.dump(scores, f)
    
    print(f"\nTraining completed! Model saved as 'snake_dqn_model.pth'")
    return agent, scores

def play_game_visual(agent=None, human_play=False):
    """Play the game with visual interface"""
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption("Snake Game - AI vs Human")
    clock = pygame.time.Clock()
    font = pygame.font.Font(None, 36)
    
    game = SnakeGame()
    state = game.reset()
    
    if agent:
        agent.epsilon = 0  # No exploration during play
    
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN and human_play:
                if event.key == pygame.K_UP and game.direction != DOWN:
                    game.direction = UP
                elif event.key == pygame.K_DOWN and game.direction != UP:
                    game.direction = DOWN
                elif event.key == pygame.K_LEFT and game.direction != RIGHT:
                    game.direction = LEFT
                elif event.key == pygame.K_RIGHT and game.direction != LEFT:
                    game.direction = RIGHT
        
        if not game.game_over:
            if not human_play and agent:
                action = agent.act(state)
                state, _, game.game_over = game.step(action)
            elif human_play:
                # For human play, always go straight (direction already set by keys)
                state, _, game.game_over = game.step(0)
        
        # Clear screen
        screen.fill(BLACK)
        
        # Draw snake
        for segment in game.snake:
            pygame.draw.rect(screen, GREEN, 
                           (segment[0] * GRID_SIZE, segment[1] * GRID_SIZE, 
                            GRID_SIZE, GRID_SIZE))
        
        # Draw food
        pygame.draw.rect(screen, RED, 
                        (game.food[0] * GRID_SIZE, game.food[1] * GRID_SIZE, 
                         GRID_SIZE, GRID_SIZE))
        
        # Draw score
        score_text = font.render(f"Score: {game.score}", True, WHITE)
        screen.blit(score_text, (10, 10))
        
        # Draw game over message
        if game.game_over:
            game_over_text = font.render("Game Over! Press ESC to exit", True, WHITE)
            screen.blit(game_over_text, (WINDOW_WIDTH//2 - 150, WINDOW_HEIGHT//2))
            
            # Check for ESC key to exit
            keys = pygame.key.get_pressed()
            if keys[pygame.K_ESCAPE]:
                running = False
        
        pygame.display.flip()
        clock.tick(10 if human_play else 15)  # Slower for human, faster for AI
    
    pygame.quit()
    return game.score

def plot_training_results():
    """Plot training results"""
    try:
        with open('training_scores.json', 'r') as f:
            scores = json.load(f)
        
        plt.figure(figsize=(12, 8))
        
        # Plot raw scores
        plt.subplot(2, 1, 1)
        plt.plot(scores, alpha=0.6)
        plt.title('Training Scores Over Episodes')
        plt.xlabel('Episode')
        plt.ylabel('Score')
        plt.grid(True)
        
        # Plot moving average
        plt.subplot(2, 1, 2)
        window_size = 100
        if len(scores) >= window_size:
            moving_avg = [np.mean(scores[i:i+window_size]) 
                         for i in range(len(scores)-window_size+1)]
            plt.plot(range(window_size-1, len(scores)), moving_avg)
            plt.title(f'Moving Average Score (window size: {window_size})')
            plt.xlabel('Episode')
            plt.ylabel('Average Score')
            plt.grid(True)
        
        plt.tight_layout()
        plt.savefig('training_results.png')
        plt.show()
        
        print(f"Final average score (last 100 episodes): {np.mean(scores[-100:]):.2f}")
        print(f"Maximum score achieved: {max(scores)}")
        
    except FileNotFoundError:
        print("Training results not found. Please train the model first.")

def live_training_visual(episodes=500, games_per_episode=3):
    """Watch AI learn in real-time with visual feedback"""
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption("Snake AI - Live Learning")
    clock = pygame.time.Clock()
    font = pygame.font.Font(None, 36)
    small_font = pygame.font.Font(None, 24)
    
    game = SnakeGame()
    agent = DQNAgent()
    
    # Training statistics
    scores = []
    episode_scores = []
    current_episode = 0
    games_played = 0
    best_score = 0
    recent_avg = 0
    
    print("🎮 Live Training Started!")
    print("Watch the AI learn to play Snake in real-time!")
    print("Press ESC to stop training early")
    print("-" * 50)
    
    running = True
    training_active = True
    
    while running and current_episode < episodes:
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    training_active = False
                    print("\nTraining stopped by user!")
        
        if not training_active:
            # Just display the current game state
            screen.fill(BLACK)
            
            # Draw snake
            for segment in game.snake:
                pygame.draw.rect(screen, GREEN, 
                               (segment[0] * GRID_SIZE, segment[1] * GRID_SIZE, 
                                GRID_SIZE, GRID_SIZE))
            
            # Draw food
            pygame.draw.rect(screen, RED, 
                            (game.food[0] * GRID_SIZE, game.food[1] * GRID_SIZE, 
                             GRID_SIZE, GRID_SIZE))
            
            # Draw final statistics
            final_text = font.render("Training Stopped - Press ESC to exit", True, WHITE)
            screen.blit(final_text, (10, 10))
            
            keys = pygame.key.get_pressed()
            if keys[pygame.K_ESCAPE]:
                break
            
            pygame.display.flip()
            clock.tick(60)
            continue
        
        # Start new episode
        if games_played == 0:
            current_episode += 1
            episode_scores = []
            state = game.reset()
        
        # Play one step
        if not game.game_over:
            action = agent.act(state)
            next_state, reward, done = game.step(action)
            agent.remember(state, action, reward, next_state, done)
            state = next_state
        else:
            # Game over - record score and start new game
            episode_scores.append(game.score)
            scores.append(game.score)
            games_played += 1
            
            if game.score > best_score:
                best_score = game.score
            
            # Train the agent
            agent.replay(32)
            
            # Check if episode is complete
            if games_played >= games_per_episode:
                games_played = 0
                # Update target network every episode
                if current_episode % 10 == 0:
                    agent.update_target_network()
                
                # Calculate recent average
                if len(scores) >= 100:
                    recent_avg = sum(scores[-100:]) / 100
                else:
                    recent_avg = sum(scores) / len(scores) if scores else 0
                
                # Print progress every 10 episodes
                if current_episode % 10 == 0:
                    avg_episode_score = sum(episode_scores) / len(episode_scores)
                    print(f"Episode {current_episode:4d} | Avg Score: {avg_episode_score:5.1f} | "
                          f"Best: {best_score:3d} | Epsilon: {agent.epsilon:.3f} | "
                          f"Recent Avg: {recent_avg:.1f}")
            else:
                # Start new game in same episode
                state = game.reset()
        
        # Visual rendering
        screen.fill(BLACK)
        
        # Draw snake
        for i, segment in enumerate(game.snake):
            color = GREEN if i == 0 else (0, 200, 0)  # Head slightly brighter
            pygame.draw.rect(screen, color, 
                           (segment[0] * GRID_SIZE, segment[1] * GRID_SIZE, 
                            GRID_SIZE, GRID_SIZE))
        
        # Draw food
        pygame.draw.rect(screen, RED, 
                        (game.food[0] * GRID_SIZE, game.food[1] * GRID_SIZE, 
                         GRID_SIZE, GRID_SIZE))
        
        # Draw training statistics
        stats_y = 10
        stats = [
            f"Episode: {current_episode}/{episodes}",
            f"Current Score: {game.score}",
            f"Best Score: {best_score}",
            f"Games in Episode: {games_played + 1}/{games_per_episode}",
            f"Exploration Rate: {agent.epsilon:.3f}",
            f"Recent Average: {recent_avg:.1f}",
            f"Total Games: {len(scores)}"
        ]
        
        for i, stat in enumerate(stats):
            color = WHITE if i != 4 else (255, 255 - int(agent.epsilon * 255), 0)  # Epsilon in color
            text = small_font.render(stat, True, color)
            screen.blit(text, (10, stats_y + i * 25))
        
        # Draw learning progress bar
        progress = current_episode / episodes
        bar_width = 300
        bar_height = 20
        bar_x = WINDOW_WIDTH - bar_width - 10
        bar_y = 10
        
        # Background bar
        pygame.draw.rect(screen, (50, 50, 50), (bar_x, bar_y, bar_width, bar_height))
        # Progress bar
        pygame.draw.rect(screen, (0, 255, 0), (bar_x, bar_y, int(bar_width * progress), bar_height))
        # Border
        pygame.draw.rect(screen, WHITE, (bar_x, bar_y, bar_width, bar_height), 2)
        
        progress_text = small_font.render(f"Progress: {progress*100:.1f}%", True, WHITE)
        screen.blit(progress_text, (bar_x, bar_y + 25))
        
        # Draw instructions
        instruction_text = small_font.render("Press ESC to stop training", True, (200, 200, 200))
        screen.blit(instruction_text, (10, WINDOW_HEIGHT - 30))
        
        # Learning indicator
        if games_played == 0 and current_episode > 1:  # Between episodes
            learning_text = font.render("🧠 LEARNING...", True, (255, 255, 0))
            screen.blit(learning_text, (WINDOW_WIDTH//2 - 80, WINDOW_HEIGHT//2))
        
        pygame.display.flip()
        clock.tick(15)  # Moderate speed for watching
    
    pygame.quit()
    
    # Save the model after live training
    if training_active:
        agent.save('snake_dqn_live_model.pth')
        print(f"\n🎉 Live training completed!")
        print(f"📊 Total games played: {len(scores)}")
        print(f"🏆 Best score achieved: {best_score}")
        print(f"📈 Final average (last 100): {recent_avg:.1f}")
        print(f"💾 Model saved as 'snake_dqn_live_model.pth'")
        
        # Save live training scores
        with open('live_training_scores.json', 'w') as f:
            json.dump(scores, f)
    
    return scores

def main():
    """Main function to run the program"""
    print("Snake Game with DQN AI Agent")
    print("=" * 40)
    print("1. Train AI Agent (Background)")
    print("2. Watch AI Play (Pre-trained)")
    print("3. Play Yourself")
    print("4. View Training Results")
    print("5. 🎮 Watch AI Learn Live!")
    print("6. Exit")
    
    while True:
        choice = input("\nEnter your choice (1-6): ").strip()
        
        if choice == '1':
            print("\nStarting background training...")
            episodes = int(input("Enter number of episodes (default 1000): ") or "1000")
            train_agent(episodes)
            
        elif choice == '2':
            agent = DQNAgent()
            if os.path.exists('snake_dqn_model.pth'):
                agent.load('snake_dqn_model.pth')
                print("\nWatching trained AI play...")
            elif os.path.exists('snake_dqn_live_model.pth'):
                agent.load('snake_dqn_live_model.pth')
                print("\nWatching live-trained AI play...")
            else:
                print("\nNo trained model found! Training a quick model...")
                train_agent(100)
                agent.load('snake_dqn_model.pth')
            
            score = play_game_visual(agent, human_play=False)
            print(f"AI scored: {score}")
            
        elif choice == '3':
            print("\nYour turn! Use arrow keys to control the snake.")
            score = play_game_visual(human_play=True)
            print(f"You scored: {score}")
            
        elif choice == '4':
            plot_training_results()
            
        elif choice == '5':
            print("\n🎮 Starting Live Training!")
            print("You'll watch the AI learn from scratch in real-time.")
            episodes = int(input("Enter number of episodes to watch (default 200): ") or "200")
            games_per_ep = int(input("Games per episode (default 3): ") or "3")
            live_training_visual(episodes, games_per_ep)
            
        elif choice == '6':
            print("Thanks for playing!")
            break
            
        else:
            print("Invalid choice. Please enter 1-6.")

if __name__ == "__main__":
    main()