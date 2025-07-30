# HFSS Optimization Framework

HFSS-AI-Optimizer is a design optimization tool that integrates various optimization algorithms, AI surrogate models, and automated control of HFSS (High-Frequency Structure Simulator). This tool aims to achieve efficient parameter optimization for radio frequency/microwave devices by leveraging surrogate modeling techniques such as Bayesian Neural Networks (BNN) and Gaussian Process Regression (GPR), combined with active learning strategies. It significantly reduces the cost of manual parameter tuning and simulation time while improving design efficiency.

## Features

• 🧮 Multi-strategy optimization:
  - Bayesian Optimization (GP Minimize)
  - Particle Swarm Optimization (PSO)
  - CMA-ES Evolution Strategy
  - Differential Evolution

• 🤖 AI Agent Optimization:
  - Bayesian Neural Network (BNN) surrogate models
  - Active learning framework
  - Uncertainty-aware sampling

• ⚙️ Advanced HFSS control:
  - Robust connection management
  - Automatic port mapping
  - S-parameter extraction

• 📊 Visualization & Analysis:
  - Optimization progress tracking
  - Constraint violation analysis
  - Model validation tools

• 💾 Data Management:
  - Dataset versioning
  - Feature extraction pipeline
  - Results serialization

## Installation

### Prerequisites
- Ansys HFSS 2023 R1 or later
- Python 3.8+

### Install dependencies
```bash
pip install -r requirements.txt
```

### Key Dependencies

| Package               | Version  | Purpose                  |
|-----------------------|----------|--------------------------|
| numpy                 | >=1.20   | Numerical computing      |
| pandas                | >=1.3    | Data manipulation        |
| scikit-learn          | >=1.0    | Machine learning         |
| scikit-optimize       | >=0.9    | Bayesian optimization    |
| tensorflow            | >=2.8    | Neural networks          |
| tensorflow-probability| >=0.16   | Bayesian modeling        |
| pyDOE                 | >=0.3    | Experimental design      |
| matplotlib            | >=3.5    | Visualization            |
| ansys-aedt            | >=0.4    | HFSS integration         |

## Quick Start

### 1. Traditional Constrained Optimization
```python
from optimizers import HfssAdvancedConstraintOptimizer

optimizer = HfssAdvancedConstraintOptimizer(
    project_path="path/to/your/project.aedt",
    design_name="HFSSDesign1",
    variables=[
        {'name': 'Lp', 'bounds': (3, 30), 'unit': 'mm'},
        {'name': 'Wp', 'bounds': (3, 25), 'unit': 'mm'}
    ],
    constraints=[
        {
            'expression': 'dB(S11)',
            'target': -15,
            'operator': '<',
            'weight': 1.0,
            'freq_range': (5e9, 7e9)
        }
    ],
    global_port_map={'S11': ('1', '1')}
)

# Run Bayesian optimization
result = optimizer.optimize(optimizer_type="bayesian")

# Run PSO optimization
result = optimizer.optimize(optimizer_type="pso")
```

### 2. AI Agent Optimization
```python
from ai_optimizer import HfssAIAgentOptimizer

ai_optimizer = HfssAIAgentOptimizer(
    project_path="path/to/your/project.aedt",
    design_name="HFSSDesign1",
    variables=[...],
    constraints=[...],
    global_port_map={'S11': ('1', '1')},
    feature_freq_points=50,
    max_active_cycles=20
)

# Start AI-driven optimization
best_params = ai_optimizer.optimize()
```

## Project Structure
```
hfss-optimization/
├── optim_framework.py   # Main optimizer class
├── BNNoptim.py          # Bayesian Neural Network optimizer
├── GPRoptim.py          # Gaussian Process Regression optimizer
├── DataSet.py           # Feature management
├── Trainer.py           # Active learning framework
├── api.py               # HFSS control interface
├── examples/            # Usage examples
├── optim_results/       # Optimization outputs
├── requirements.txt     # Dependencies
└── README.md            # This document
```

## Key Components

### 1. HFSS Controller (api.py)
Robust interface for HFSS automation with:
- Connection management with auto-recovery
- Variable setting with unit conversion
- S-parameter extraction
- Simulation timeout handling

### 2. Optimization Framework (optim_framework.py)
Supports multiple optimization strategies:
```python
# Available optimizer types
optimizer.optimize(optimizer_type="bayesian")  # Default
optimizer.optimize(optimizer_type="pso")
optimizer.optimize(optimizer_type="cmaes")
optimizer.optimize(optimizer_type="de")
```

### 3. AI Optimizers (BNNoptim.py & GPRoptim.py)
Advanced surrogate models for optimization:
- Bayesian Neural Network (BNN) implementation
- Gaussian Process Regression (GPR) implementation
- Active learning cycles with uncertainty-aware sampling

## Contributing
We welcome contributions! Please follow these steps:
1. Fork the repository
2. Create a new branch (git checkout -b feature/your-feature)
3. Commit your changes (git commit -am 'Add awesome feature')
4. Push to the branch (git push origin feature/your-feature)
5. Open a pull request

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Support
For issues or questions, please open an issue on GitHub or contact:
- ayang1643816608@gmail.com

Optimize with confidence - This framework has been successfully applied to antenna design, filter optimization, and EMI mitigation projects with 40%+ reduction in simulation time compared to manual approaches.
