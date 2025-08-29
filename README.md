# AESER: Adaptive, Energy- & Security-aware Efficient Routing

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.8.0%2Bcpu-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

A single-file implementation of a Graph Neural Network (GNN) + Proximal Policy Optimization (PPO) reinforcement learning router with energy dynamics and security-aware anomaly detection.

## 🚀 Features

- **Graph-based Routing Environment**: Connected random graphs with energy dynamics
- **Security-aware Penalties**: Lightweight autoencoder for anomaly detection
- **Energy-aware Routing**: Battery management with transmission/reception costs
- **Action Masking**: Ensures only valid neighbor actions are chosen
- **PPO Training**: Stores raw PyG observations for correct minibatch processing
- **Deterministic Evaluation**: Reproducible results with comprehensive metrics

## 📋 Requirements

- Python 3.12+
- PyTorch 2.8.0+ (CPU-only)
- PyTorch Geometric 2.6.1+
- Gym 0.26.2
- Other dependencies (see requirements.txt)

## 🛠️ Installation

### Option 1: Using requirements.txt (Recommended)

```bash
# Clone the repository
git clone <repository-url>
cd scripy

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Option 2: Manual installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install PyTorch CPU
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install PyTorch Geometric
pip install torch-geometric -f https://data.pyg.org/whl/torch-2.8.0+cpu.html

# Install other dependencies
pip install gym==0.26.2 numpy pandas matplotlib tqdm networkx
```

## 🎯 Usage

### Basic Usage

```bash
# Activate virtual environment
source venv/bin/activate

# Run the model
python aeser_model_cpu.py
```

### Configuration

The model can be configured by modifying parameters in the `main()` function:

```python
# Environment parameters
env = AESERGraphEnv(
    N=18,                    # Number of nodes
    p_edge=0.22,            # Edge probability
    max_steps=40,           # Maximum steps per episode
    arrival_reward=1.0,     # Reward for reaching target
    step_cost=0.02,         # Cost per step
    energy_weight=0.6,      # Energy penalty weight
    security_weight=0.2,    # Security penalty weight
    seed=7                  # Random seed
)

# Training parameters
EPOCHS = 160
STEPS_PER_EPOCH = 512
```

## 🏗️ Architecture

### Core Components

1. **AESERGraphEnv**: Gym environment implementing the routing problem
   - Graph generation with Erdos-Renyi model
   - Energy dynamics (battery management)
   - Security scoring (anomaly detection)
   - Action masking for valid moves

2. **GNNEncoder**: Graph Neural Network for node representation
   - Supports SAGE and GCN convolution layers
   - Configurable hidden dimensions and layers
   - Global mean pooling for graph-level features

3. **ActorCritic**: PPO actor-critic network
   - Actor: Node selection policy
   - Critic: State value estimation
   - Action masking integration

4. **AESER_PPO**: PPO training implementation
   - GAE (Generalized Advantage Estimation)
   - Clipped policy optimization
   - Minibatch training with PyG data

### Security Features

- **TrafficAE**: Lightweight autoencoder for anomaly detection
- **Anomaly Scoring**: Normalized reconstruction error
- **Security Penalties**: Integrated into reward function

## 📊 Outputs

The model generates three output files:

1. **`aeser_train_curve.png`**: Training progress visualization
2. **`aeser_results.csv`**: Evaluation metrics in CSV format
3. **`aeser_eval_summary.txt`**: Human-readable evaluation summary

### Metrics

- **Average Reward**: Mean episode reward across test episodes
- **Success Rate**: Percentage of successful routing episodes
- **Average Steps**: Mean steps per episode

## 🔧 Technical Details

### Environment Dynamics

- **Graph**: Connected random graph with configurable edge probability
- **Energy**: Battery levels decrease with transmission/reception
- **Security**: Anomaly scores affect routing decisions
- **Termination**: Episode ends on target arrival, step limit, or battery failure

### Training Process

1. **Collection**: Gather trajectories using current policy
2. **GAE**: Compute advantages using Generalized Advantage Estimation
3. **PPO Update**: Multiple epochs of minibatch updates
4. **Evaluation**: Deterministic evaluation on test episodes

### Action Masking

- **Neighbor Constraint**: Only adjacent nodes can be selected
- **Battery Constraint**: Nodes with sufficient battery are preferred
- **Fallback Logic**: Graceful degradation when constraints conflict

## 🚨 Troubleshooting

### Common Issues

1. **Import Errors**: Ensure virtual environment is activated
2. **PyTorch Version Mismatch**: Use matching PyG wheels for your Torch version
3. **Memory Issues**: Reduce `STEPS_PER_EPOCH` or `N` for smaller graphs

### Performance Tips

- **CPU Optimization**: Model automatically sets optimal thread count
- **Batch Size**: Adjust `minibatch` parameter based on available memory
- **Graph Size**: Smaller graphs train faster but may be less realistic

## 📚 References

- [PyTorch Geometric Documentation](https://pytorch-geometric.readthedocs.io/)
- [OpenAI Gym Documentation](https://www.gymlibrary.dev/)
- [PPO Paper](https://arxiv.org/abs/1707.06347)
- [Graph Neural Networks](https://arxiv.org/abs/1812.08434)

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- PyTorch Geometric team for the excellent GNN library
- OpenAI for the Gym framework
- The reinforcement learning community for PPO implementation insights

---

**Note**: This is a research implementation. For production use, consider additional testing, validation, and security measures.
# AESER-IOT
# AESER-IOT
