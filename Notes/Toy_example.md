# **Neural Approximation of the Infinite-Horizon Viability Kernel for 3D Dubins Car**
### *Stationary Discounted HJ-VI Implementation Documentation*

This document provides **all implementation details** needed to train a neural network to approximate the **infinite-horizon discounted viability kernel** for the **3D Dubins car** using the **stationary discounted HJ-VI**.

It is the continuous-time, infinite-horizon extension of **DeepReach**, specialized for your Dubins car example.

---

# 1. Problem Setup

We consider the Dubins car dynamics with disturbance:

\[
\dot{x}_{1}=u_{1}\cos(x_{3})+d_{1},\quad
\dot{x}_{2}=u_{1}\sin(x_{3})+d_{2},\quad
\dot{x}_{3}=u_{2}.
\]

State:
- \(x_1, x_2\): position  
- \(x_3\): heading  

Control bounds:
\[
u_1 \in [0.05, 1],\quad u_2 \in [-1,1]
\]

Disturbance bounds:
\[
d_1,d_2 \in [-0.01, 0.01]
\]

---

## 1.1 State Constraints \(g(x)\)

We define the **safe region** via:

\[
g(x)=\max\{|x_1|-L,\; |x_2|-L,\;
r^2-(x_1-C_x)^2-(x_2-C_y)^2 \}.
\]

Safe region:
\[
\mathcal X_{\text{safe}}=\{x\mid g(x)\le 0\}.
\]

Unsafe when:
- Outside square \([-L,L]^2\)
- Inside circular obstacle of radius \(r\) at \((C_x,C_y)\)

---

# 2. Infinite-Horizon Discounted Viability Kernel

We use the performance index:

\[
J(x,u,d)=\sup_{s\ge 0}e^{-\gamma s}g(x(s)),
\]

The **infinite-horizon value function** is:

\[
V(x)=\sup_\delta\inf_{u(\cdot)} J(x,u,\delta[u]),
\]

The **viability kernel** is:

\[
\mathcal V_\infty=\{x: V(x)\le 0\}.
\]

---

# 3. Stationary Discounted HJ-VI

From the dynamic programming derivation:

\[
0=\max\Big(
g(x)-V(x),\;
\min_{u\in U}\max_{d\in D}
\big[\nabla V(x)\cdot f(x,u,d)-\gamma V(x)\big]
\Big).
\tag{HJVI}_\infty
\]

This PDE:
- Is **stationary** (no time)
- Has a **unique bounded viscosity solution** because \(\gamma>0\)
- Yields viability kernel as the 0-level set

---

# 4. Neural Network Approximation

We parameterize:

\[
V_\theta:\mathbb R^3\to\mathbb R
\]

Use **SIREN (sinusoidal) networks**, because:
- We need accurate **gradients**
- SIRENs outperform ReLU/tanh for HJ PDEs (as shown by DeepReach)

---

# 5. HJ-VI Residual Loss for Training

For a sampled state \(x\), compute:

- Network value: \(V_\theta(x)\)
- Spatial gradient: \(\nabla_x V_\theta(x)\)
- Hamiltonian:
  \[
  H_\theta(x)=\min_u\max_d\big[\nabla V_\theta(x)\cdot f(x,u,d)-\gamma V_\theta(x)\big]
  \]
- Viability constraint residual:
  \[
  R(x)=\max(g(x)-V_\theta(x),\; H_\theta(x))
  \]

The training loss:

\[
\boxed{
\mathcal L(\theta)=\mathbb E_x|R(x;\theta)|
}
\]

This enforces **both branches** of the HJ-VI:
- If \(V<g\): enforce dynamic safe evolution  
- If \(V\ge g\): enforce \(V=g\) (state inside unsafe region)

---

# 6. State Sampling Distribution

We sample states \(x=(x_1,x_2,x_3)\) from:

\[
x_1,x_2\sim U[-L-\delta,\; L+\delta],\quad
x_3\sim U[-\pi,\pi].
\]

Optionally:
- **oversample near the constraint** \(g(x)\approx 0\)
- use mixture sampling:
  \[
  \rho = 0.7\,\text{Uniform} + 0.3\,\text{BoundarySampling}
  \]

---

# 7. Hamiltonian Implementation (Same as helperOC)

For each sample \(x\), gradient \(p=(p_1,p_2,p_3)=\nabla_xV\):

\[
H(x) = 
\min_{u_1,u_2}\max_{d_1,d_2}
\big[
p_1(u_1\cos x_3+d_1)
+p_2(u_1\sin x_3+d_2)
+p_3u_2
-\gamma V(x)
\big]
\]

Closed-form optimal controls/disturbances:

u1_opt = uMin1 if p1cos(x3)+p2sin(x3)>0
uMax1 otherwise

u2_opt = uMin2 if p3>0
uMax2 otherwise

d1_opt = dMax1 if p1>0 else dMin1
d2_opt = dMax2 if p2>0 else dMin2


Hamiltonian:

\[
H=p_1(u_1^\star\cos x_3+d_1^\star)
+p_2(u_1^\star\sin x_3+d_2^\star)
+p_3u_2^\star
-\gamma V
\]

This matches exactly your MATLAB Hamiltonian implementation.

---

# 8. Full Training Algorithm

### **Algorithm**
1. Initialize SIREN network \(V_\theta(x)\)
2. Repeat until convergence:
   1. Sample batch of states \(x\sim\rho\)
   2. Compute:
      - \(V_\theta(x)\)
      - gradients \(p=\nabla_x V_\theta(x)\)
      - optimal \(u^\star, d^\star\)
      - Hamiltonian \(H_\theta(x)\)
      - constraint residual \(R(x)\)
   3. Loss = mean absolute value of \(R(x)\)
   4. Update \(\theta\leftarrow\theta-\eta\nabla_\theta L\)

### **Stopping criteria**
- \(\max_x |R(x)| < 10^{-3}\)
- or training loss plateaus

---

