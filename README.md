# snake-RL

A reinforcement learning (RL) agent trained to play the classic game of Snake, built as a hands-on learning experience in deep reinforcement learning using PyTorch.

## Purpose

This project was my **first real exploration into reinforcement learning**. It helped me understand key RL concepts like experience replay, exploration vs exploitation, Q-learning, and training agents in a game environment.

I followed this [excellent tutorial](https://www.youtube.com/watch?v=L8ypSXwyBds&t=1054s) to build the baseline agent, and then extended it with my own improvements to enhance learning performance.

---

## Key Modifications

After getting the base agent working, I experimented with various parameters to improve training:

- 🔄 **Extended Exploration Phase**  
  Increased epsilon decay period from 80 → **400 episodes**, giving the agent more time to explore before settling into greedy behavior.

- 🧠 **Improved Learning Capacity**
  - **Max memory** (replay buffer) increased to allow the agent to remember more previous states
  - **Batch size** increased to speed up convergence during training

These tweaks led to **better and more consistent high scores**, showing clearer learning behavior.

---

## What I Learned

- Core RL concepts: Q-learning, exploration vs. exploitation, reward shaping
- Neural network design for game state evaluation
- How modifying training parameters affects learning dynamics
- Debugging and tuning agent performance in a dynamic environment