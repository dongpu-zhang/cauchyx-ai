# CauchyX PDE Agent — Natural Language PDE Solver for Claude Code

Solve partial differential equations by describing them in plain English or Chinese, directly inside a Claude Code conversation.

**Powered by [CauchyX AI](https://cauchyx.com) · NVIDIA Inception Member**

---

## What it does

Type `/pde <description>` in Claude Code and get:

- Automatic PDE type detection and parameter parsing from natural language
- Finite difference solvers (FTCS, Leapfrog, Godunov upwind, sparse direct)
- CauchyNet PINN backend (single-layer PyTorch with Cauchy activation φ(x)=(λ₁x+λ₂)/(x²+d²))
- NVIDIA PhysicsNeMo / Modulus backend (if installed)
- Dark-theme solution plots displayed inline in the conversation

---

## Supported PDEs

| Keywords | Equation | Scheme |
|---|---|---|
| `heat`, `diffusion`, `thermal` | u_t = α∇²u | FTCS FDM |
| `wave`, `vibration`, `string` | u_tt = c²u_xx | Leapfrog FDM |
| `poisson`, `laplace`, `electrostatic` | −∇²u = f | Sparse direct |
| `burgers`, `viscous`, `shock` | u_t + u·u_x = ν·u_xx | Godunov upwind |
| `advection`, `transport` | u_t + a·u_x = 0 | First-order upwind |
| `allen-cahn`, `phase field` | u_t = ε²∇²u + u−u³ | FTCS FDM |
| `ode`, `spring`, `pendulum`, `oscillator` | y''+2γωy'+ω²y=0 | SciPy RK45 |
| `CauchyNet`, `PINN`, `neural` | any above | CauchyNet PINN (PyTorch) |
| `PhysicsNeMo`, `modulus` | any above | NVIDIA PhysicsNeMo |

---

## Installation

### 1. Prerequisites

```bash
pip install numpy scipy matplotlib torch
# Optional: NVIDIA PhysicsNeMo
pip install physicsnemo   # or: pip install nvidia-modulus
```

### 2. Download the solver

```bash
# Clone the repo
git clone https://github.com/dongpu-zhang/cauchyx-ai.git
cd cauchyx-ai/pde-agent
```

Or download just `pde_solver.py` directly:

```bash
curl -O https://raw.githubusercontent.com/dongpu-zhang/cauchyx-ai/main/pde-agent/pde_solver.py
```

### 3. Install the Claude Code slash command

Copy `commands/pde.md` to your Claude Code custom commands directory:

**macOS / Linux:**
```bash
mkdir -p ~/.claude/commands
cp commands/pde.md ~/.claude/commands/pde.md
```

**Windows (PowerShell):**
```powershell
New-Item -ItemType Directory -Force "$env:USERPROFILE\.claude\commands"
Copy-Item commands\pde.md "$env:USERPROFILE\.claude\commands\pde.md"
```

### 4. Edit the command path

Open `~/.claude/commands/pde.md` and update the solver path to match where you saved `pde_solver.py`:

```
python /your/path/to/pde_solver.py "<description>"
```

### 5. Restart Claude Code

The `/pde` slash command will be available in your next Claude Code session.

---

## Usage

### In Claude Code conversation

```
/pde heat equation alpha=0.01 on [0,1] until t=0.5 IC=sin(pi*x) Dirichlet BC=0
/pde wave equation c=1.5 on [0,2] IC=sin(pi*x) zero BC until t=3
/pde 2D Poisson equation on unit square with zero Dirichlet boundary conditions
/pde Burgers equation nu=0.005 periodic BC IC=-sin(pi*x) until t=1
/pde Allen-Cahn epsilon=0.05 on [0,1] until t=2
/pde harmonic oscillator omega=2 gamma=0.1 until t=20
/pde CauchyNet heat 1D alpha=0.01 on [0,1] t=0.5
/pde PhysicsNeMo Navier-Stokes 2D
```

### From the command line

```bash
python pde_solver.py "wave equation c=1.5 on [0,2] IC=sin(pi*x) zero BC until t=3"
```

Output plot is saved to `pde_<type>_<dims>d.png` in the current directory.

---

## Parameter syntax

| Natural language | Extracted as |
|---|---|
| `alpha=0.01`, `α=0.01`, `diffusivity 0.01` | thermal diffusivity α |
| `c=1.5`, `wave speed 2` | wave speed c |
| `nu=0.005`, `viscosity 0.01` | kinematic viscosity ν |
| `[0,1]`, `[0,pi]`, `[0,2]` | spatial domain |
| `t=0.5`, `until t=2`, `to t=3` | end time |
| `sin(pi*x)`, `gaussian`, `step function` | initial condition |
| `Dirichlet`, `Neumann`, `periodic` | boundary condition |
| `2D`, `two-dimensional` | 2D problem |
| `nx=256` | grid resolution |
| `CauchyNet` or `PINN` | neural network solver |
| `PhysicsNeMo` or `modulus` | NVIDIA backend |

---

## CauchyNet PINN

The PINN backend uses a single wide hidden layer with the **Cauchy activation function** developed by CauchyX AI:

```
φ(x; λ₁, λ₂, d) = (λ₁x + λ₂) / (x² + d²)
```

This activation's heavy tails and rational form make it particularly effective for physics-informed problems with sharp gradients and multi-scale features.

Reference: *XNet: Replacing ReLU with a Width-First Cauchy PINN*, Neural Networks 2025, DOI: 10.1016/j.neunet.2024.106955

---

## Example output

Wave equation c=1.5 on [0,2], IC=sin(πx), Dirichlet BC=0, until t=3:

- 6 evenly-spaced time snapshots
- Leapfrog scheme, ν²=0.81
- ‖u‖_max ≈ 1.0 (full amplitude recovery at t=2)

---

## License

MIT License. Free to use, modify, and distribute.

---

## About CauchyX AI

CauchyX AI develops physics-informed machine learning infrastructure for scientific computing and industrial simulation.

- Website: https://cauchyx.com  
- GitHub: https://github.com/dongpu-zhang/cauchyx-ai  
- NVIDIA Inception Member

---

# 中文说明

## 这是什么

在 Claude Code 对话里，用自然语言描述偏微分方程，直接求解并展示结果图。

输入 `/pde <描述>` 即可：

- 自动识别 PDE 类型和参数
- 有限差分法求解器（热方程、波动方程、Burgers、Poisson 等）
- CauchyNet PINN 神经网络求解器（PyTorch，Cauchy 激活函数）
- NVIDIA PhysicsNeMo 后端（需另行安装）
- 深色主题结果图直接显示在对话窗口

## 安装步骤

### 1. 安装依赖

```bash
pip install numpy scipy matplotlib torch
```

### 2. 下载求解器

```bash
git clone https://github.com/dongpu-zhang/cauchyx-ai.git
```

取出 `pde-agent/pde_solver.py`，放到你常用的目录（如 `D:\claudecode\pde_solver.py`）。

### 3. 安装斜杠命令

将 `commands/pde.md` 复制到 Claude Code 自定义命令目录：

**Windows：**
```powershell
Copy-Item pde-agent\commands\pde.md "$env:USERPROFILE\.claude\commands\pde.md"
```

**macOS / Linux：**
```bash
cp pde-agent/commands/pde.md ~/.claude/commands/pde.md
```

### 4. 修改路径

打开 `~/.claude/commands/pde.md`，将其中的 `pde_solver.py` 路径改为你实际存放的路径。

### 5. 重启 Claude Code

重启后 `/pde` 命令即可使用。

## 使用示例

```
/pde 一维热方程 alpha=0.01 区间[0,1] t=0.5 初始条件sin(pi*x) Dirichlet边界
/pde 波动方程 c=1.5 在[0,2]上 初始sin(pi*x) 零边界 到t=3
/pde 二维Poisson方程 单位正方形 零Dirichlet边界
/pde Burgers方程 nu=0.005 周期边界 初始-sin(pi*x) 到t=1
/pde CauchyNet 热方程 1D alpha=0.01 在[0,1] t=0.5
/pde 谐振子 omega=2 gamma=0.1 到t=20
```
