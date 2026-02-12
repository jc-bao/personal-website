[[PDF](https://drive.google.com/file/d/1wc-URim71ra3FrP07KplxTk9nae6JtRt/view?usp=sharing)][[Slides](https://drive.google.com/file/d/1Ebvetsh_TlQOdt4cpwxoUJVPZ8y-9io2/view?usp=sharing)]

Diffusion-based planner has been widely used in robot motion planning thanks to its ability to handle multi-modal demonstration, scalability to high-dim state space, and stability in training.
However, most diffusion-based planners are fully data-driven, which requires high-quality data and cannot adapt to new dynamics effectively due to the model-free nature.
The goal of this tutorial is to introduce a new **model-based** diffusion (MBD) planner that leverages model information to generate high-quality trajectories **without data** while enable the planner use diverse quality data more effectively.

## Background: Diffusion Model in Planning

### Diffusion model as a powerful sampler

A diffusion model is a generative model capable of generating samples from a given distribution. It serves as a powerful tool for distribution matching, extensively utilized in image generation, text generation, and other creative tasks.

![diffusion examples](assets/blog/mbd/assetsdalle.png)

However, sampling from the target distribution is hard due to its high-dim and sparse nature. Diffusion model introduces a forward process to corrupt the distribution with noise to make it smooth and easier to sample from.

$$
\begin{aligned}
p_{i|i-1}(\cdot| Y^{(i-1)}) \sim \mathcal{N}(\sqrt{\alpha_i} Y^{(i-1)}, \sqrt{1-\alpha_i} I) \\
p_{i|0}(\cdot | Y^{(0)}) \sim \mathcal{N}(\sqrt{\bar{\alpha}_i} Y^{(0)}, \sqrt{1-\bar{\alpha}_i} I), \quad \bar{\alpha}_i = \prod_{k=1}^{i} \alpha_k.
\end{aligned}
$$

![diffusion overview](assets/blog/mbd/assetsdiffusion_overview.jpg)

The reverse process is introduced to recover the original distribution by removing the noise.
For the standard diffusion model, the reverse process is achieved by learning the score function (the arrows in the figure below) from data.

$$
\begin{aligned}
    p_{i-1}(Y^{(i-1)}) & = \int p_{i-1|i}(Y^{(i-1)}|Y^{(i)}) p_{i}(Y^{(i)}) \, dY^{(i)},               \\
    p_{0}(Y^{(0)})     & = \int P_{N}(Y^{(N)}) \prod_{i=N}^{1} p_{i-1|i}(Y^{(i-1)}| Y^{(i)}) dY_{1:N}
\end{aligned}
$$

By using reverse SDE, the diffusion model can recover the original distribution iteratively.
At the core of the diffusion model is the score function, representing the noise direction used to update samples to match the target distribution. Through learning this score function, the diffusion model can adeptly generate samples from the specified distribution.

$$
Y^{(i-1)}_{k} = \frac{1}{\sqrt{\alpha_i}}\left(Y^{(i)}_{k}+\frac{1-\alpha_i}{2} \nabla_{Y^{(i)}}\log p_{i}(Y^{(i)}_{k})\right)+\sqrt{1-\alpha_i} \mathbf{z}_i
$$

![diffusion reverse](assets/blog/mbd/assetsdiffusion_reverse.gif)

The diffusion model have several advantages over other generative models, including but not limited to the following:

- **Multimodal**: It effectively handles multimodal distributions, a challenge often encountered when directly predicting distributions.
- **Scalable**: This approach scales well with high-dimensional distribution matching problems, making it versatile for various applications.
- **Stable**: Grounded in solid math and a standard multi-stage diffusion training procedure, the model ensures stability during training.
- **Non-autoregressive**: Its capability to predict entire trajectory sequences in a non-autoregressive manner enables efficient handling of non-autoregressive and multimodal distribution matching challenges.

### Diffusion solving trajectory optimization as a sampling problem

The diffuser operates by concatenating the state and action, allowing us to diffuse the state-action sequence. This process is akin to diffusing a single-channel image.

| | Image Generation | Planning |
|---|---|---|
| Diffuse | ![](assets/blog/mbd/assetsdenoise_image.gif) | ![](assets/blog/mbd/assetsdenoise_traj.gif) |
| Data | ![](assets/blog/mbd/assetsimage_diffuse.jpg) | ![](assets/blog/mbd/assetstraj_diffuse.png) |

![diffusion trajectory](assets/blog/mbd/assetsinpaint.gif)

### Example: Throwing Ball Over a Wall with Diffusion

For simplicity, let's consider a simplied version of the problem, which is 1-step trajectory optimization problem without any constraints:

$$
\begin{aligned}
        \min_{u} & J(f(x_{\text{init}}, u)) = l(f(x_{\text{init}}, u), u)
\end{aligned}
$$

For a throwing ball over a wall to maximize the distance, the cost function visualized as below:

![throwing ball task](assets/blog/mbd/assetsthrow_ball.png)
![cost function](assets/blog/mbd/assetsthrow_ball_cost.png)

The cost function is non-convex, nonlinear, and discontinuous, which makes it hard to solve with traditional optimization methods.

**Diffusion-based Planer Approach TO by Coverting Optimization to Sampling Problem**: Diffusion model bypasses the difficulty by transforming the optimization problem into a sampling problem.

$$
p_0(Y) \propto \exp{\left(-\frac{J(Y)}{\lambda}\right)}
$$

When temperature is low, sampling from it is equivalent to solving the optimization problem (the darkest red curve in the figure below).

![target distribution](assets/blog/mbd/assetsthrow_ball_distri.png)

Back to the throwing ball example, the forward the backward process would look like the following, where the sharp peak is the original distribution and the smooth distribution is the corrupted Gaussian distribution.

![throw ball forward](assets/blog/mbd/assetsthrow_ball_forward.png)

Then, the backward process would start from the corrupted distribution and recover the original distribution iteratively with reverse SDE. With that, we are able to sample from the original sparse distribution effectively.

$$
Y^{(i-1)}_{k} = \frac{1}{\sqrt{\alpha_i}}\left(Y^{(i)}_{k}+\frac{1-\alpha_i}{2} \nabla_{Y^{(i)}}\log p_{i}(Y^{(i)}_{k})\right)+\sqrt{1-\alpha_i} \mathbf{z}_i
$$

![throw ball backward](assets/blog/mbd/assetsthrow_ball_backward.gif)

The key ingredient of the reverse process is the score function $\nabla_{Y^{(i)}}\log p_{i}(Y^{(i)}_{k})$, which model-free diffusion learned merely from data without any model information.

## Introduction: Why is Model-based Diffusion?

### Limitation of Model-Free Diffusion: not generalizable

Model-free diffusion learns the score function merely from data, which means even if the target distribution has changed, the model-free diffusion would still generate trajectories based on the old dynamics/tasks.

This means if the dynamics has changed or the data itself is not high-quality, the model-free diffusion would fail to generate high-quality trajectories. The following plot shows the change of target distribution when the dynamics has changed:

![throw ball dynamics change](assets/blog/mbd/assetsthrow_ball_new_params.png)

For text/image generative task, the target distribution is unknown, so it is hard to predict the target distribution given a new task. However, for trajectory optimization, the target distribution is known given the dynamics and objective, just hard to sample from.

### Model-based Diffusion Make Score Function Conditional on Model

Model-based Diffusion (MBD) is introduced to leverage the readily available model information to conduct the reverse process more effectively. (When refer to model information, I mean we can evalute the dynamics function $f$ and the cost function $l$ at any state and action pair. With that, we can evalute target distribution $p_0$ at any state and action pair.)

Model-based Diffusion achieve that with a novel data-free score computation method and a new backward process designed for the optimization problem. With that design, MBD can use diverse quality data more effectively and adapt to dynamic changes more effectively.

The following is the comparison between model-based diffusion and model-free diffusion:

| Aspect | Model-Based Diffusion (MBD) | Model-Free Diffusion (MFD) |
| --- | --- | --- |
| Target distribution | Known, but hard to sample | Unknown, but have data from it |
| Objective | Sample high-likelihood solution | Generate diverse samples |

## Method: What is Model-based Diffusion?

### Score Function Computation with Model

Recall that the score function is the key to the reverse process in the diffusion model which guide the sample to the target distribution.

![diffusion reverse](assets/blog/mbd/assetsdiffusion_reverse%201.gif)

But given the model, how can we do the score computation? The major challenge here is that the score function $\nabla_{Y^{(i)}}\log p_{i}(Y^{(i)}_{k})$, where $p_i = \int p_{i|0} p_0 dY^{(0)}$, is intractable to compute.

$$
\begin{aligned}
    \nabla_{Y^{(i)}}\log{p_i(Y^{(i)})} = & \frac{\nabla_{Y^{(i)}}\int  p_{i\mid 0 }(Y^{(i)} \mid Y^{(0)}) p_0(Y^{(0)}) \, dY^{(0)}}{\int  p_{i\mid 0 }(Y^{(i)} \mid Y^{(0)}) p_0(Y^{(0)}) \, dY^{(0)}}
    \\ = & \frac{\int  \nabla_{Y^{(i)}}p_{i\mid 0 }(Y^{(i)} \mid Y^{(0)}) p_0(Y^{(0)}) \, dY^{(0)}}{\int  p_{i\mid 0 }(Y^{(i)} \mid Y^{(0)}) p_0(Y^{(0)}) \, dY^{(0)}} \\
    = & \frac{\int -\frac{Y^{(i)}-\sqrt{\bar{\alpha}_{i}}Y^{(0)}}{1-\bar{\alpha}_{i}}p_{i\mid 0 }(Y^{(i)} \mid Y^{(0)}) p_0(Y^{(0)}) \, dY^{(0)}}{\int  p_{i\mid 0 }(Y^{(i)} \mid Y^{(0)}) p_0(Y^{(0)}) \, dY^{(0)}} \\
    = & -\frac{Y^{(i)}}{1-\bar{\alpha}_{i}} + \frac{\sqrt{\bar{\alpha}_{i}}}{1-\bar{\alpha}_{i}} \frac{\int Y^{(0)}p_{i\mid 0 }(Y^{(i)} \mid Y^{(0)}) p_0(Y^{(0)}) \, dY^{(0)}}{\int  p_{i\mid 0 }(Y^{(i)} \mid Y^{(0)}) p_0(Y^{(0)}) \, dY^{(0)}}
\end{aligned}
$$

where,

$$
p_{i\mid 0 }(Y^{(i)} \mid Y^{(0)})\propto \exp\left(-\frac{1}{2} \frac{\left(Y^{(i)} - \sqrt{\bar{\alpha}_{i}}Y^{(0)}\right)^\top  \left(Y^{(i)} - \sqrt{\bar{\alpha}_{i}}Y^{(0)}\right)}{1-\bar{\alpha}_i}\right)
$$

To address this challenge, we propose a novel method to compute the score function with the model information. The key idea is to use Monte Carlo estimation to approximate the score function.

$$
\begin{aligned}
    & \nabla_{Y^{(i)}}\log{p_i(Y^{(i)})} = -\frac{Y^{(i)}}{1-\bar{\alpha}_{i}} + \frac{\sqrt{\bar{\alpha}_{i}}}{1-\bar{\alpha}_{i}} \frac{\int Y^{(0)} \phi_i(Y^{(0)}) p_0(Y^{(0)}) \, dY^{(0)}}{\int  \phi_i(Y^{(0)})  p_0(Y^{(0)}) \, dY^{(0)}} \\
    \approx & -\frac{Y^{(i)}}{1-\bar{\alpha}_{i}} + \frac{\sqrt{\bar{\alpha}_{i}}}{1-\bar{\alpha}_{i}} \underbrace{\frac{\sum_{Y^{(0)}\in\mathcal{Y}^{(i)}} Y^{(0)} p_0(Y^{(0)})}{\sum_{Y^{(0)}\in \mathcal{Y}^{(i)}} p_0(Y^{(0)})}}_{\text{Monte Carlo Approximation}} \\
    :=  & -\frac{Y^{(i)}}{1-\bar{\alpha}_{i}} + \frac{\sqrt{\bar{\alpha}_{i}}}{1-\bar{\alpha}_{i}}\bar{Y}^{(0)}(\mathcal{Y}^{(i)})
\end{aligned}
$$

### Speed Up Backward Process in MBD

For the backward process, instead of use standard reverse SDE, we propose Monte Carlo Score Ascent to recover the original distribution faster by leveraging the fact that we only care about the high likelihood solution.

Reverse SDE:

$$
Y^{(i-1)} = \frac{1}{\sqrt{\alpha_i}}\left(Y^{i}+\frac{1-\alpha_i}{2} \nabla_{Y^{(i)}}\log p_{i}(Y^{(i)})\right)+\sqrt{1-\alpha_i} \mathbf{z}_i
$$

MCSA (ours):

$$
Y^{(i-1)} =  \frac{1}{\sqrt{\alpha_{i}}}\left(Y^{(i)} + (1-\bar{\alpha}_{i})\nabla_{Y^{(i)}}\log{p_{i}(Y^{(i)})}\right)
$$

Major difference:

- larger step size according to the smoothness of the distribution
- no noise term in the update

To see how MBD behaves differently from MFD, here is use two method to optimize a synthetic highly non-convex objective function.

![MCSA](assets/blog/mbd/assetsmcsa.png)

Given each intermediate distribution, the backward with MCSA converges faster due to larger step size and lower sampling noise while still capturing the multimodality.

### MBD v.s. MFD

With above two key components, MBD is able to generate high-quality trajectories without relying on data. Here is the full algorithm of MBD:

![General MBD Algorithm](assets/blog/mbd/assetsopt_algo.png)

Here is the full comparison between MBD and MFD:

| Aspect | Model-Based Diffusion (MBD) | Model-Free Diffusion (MFD) |
| --- | --- | --- |
| Target distribution | Known, but hard to sample | Unknown, but have data from it |
| Objective | Sample high-likelihood solution | Generate diverse samples |
| Score Approximation | From model + data (optional) | From data |
| Backward Process | Monte Carlo Score Ascent | Reverse SDE |

### Solve Full Trajectory Optimization Problem with MBD

The above discussion is based on constrain-free optimization problem. In the standard trajectory optimization problem, where the goal is to find a trajectory that minimizes a cost function.

$$
\begin{aligned}
    \min_{x_{1:T}, u_{1:T}} & J(x_{1:T} ; u_{1:T})= l_T(x_T) + \sum_{t=0}^{T-1} l_t(x_t, u_t) \\
    \text{s.t.}   \quad     & x_0 = x_{\text{init}} \\
                             & x_{t+1} = f_t(x_t, u_t), \quad \forall t = 0, 1, \dots, T-1, \\
                             & g_t(x_t, u_t) \leq 0, \quad \forall t = 0, 1, \dots, T-1.
\end{aligned}
$$

The target distribution $p_0(Y^{(0)})$ is proportional to dynamical feasibility $p_d(Y) \propto \prod_{t=1}^{T} \mathbf{1}(x_t = f_{t-1}(x_{t-1},u_{t-1}))$, optimality $p_J(Y) \propto \exp{(-\frac{J(Y)}{\lambda})}$ and the constraints $p_g(Y) \propto \prod_{t=1}^{T} \mathbf{1}(g_t(x_t, u_t)\leq 0)$, i.e.,

$$p_0(Y) \propto p_d(Y) p_J(Y) p_g(Y)$$

Directly sampling from the target distribution is hard due to the high-dim and sparse nature of the distribution. MBD navigate this challenge by only sampling from feasible trajectory space, which is similar to the shooting method:

$$
\begin{aligned}
    \nabla_{Y_{i}}\log{p_{i}(Y_{i})} =  & -\frac{Y_{i}}{1-\bar{\alpha}_{i}} + \frac{\sqrt{\bar{\alpha}_{i}}}{1-\bar{\alpha}_{i}} \frac{\int Y_{0} \phi_i(Y_{0}) p_d(Y_{0})p_g(Y_{0})p_J(Y_{0}) \, dY_{0}}{\int  \phi_i(Y_{0})  p_d(Y_{0})p_g(Y_{0})p_J(Y_{0}) \, dY_{0}} \\
    \approx & -\frac{Y_{i}}{1-\bar{\alpha}_{i}} + \frac{\sqrt{\bar{\alpha}_{i}}}{1-\bar{\alpha}_{i}} \frac{\sum_{Y_{0} \in \mathcal{Y}^{(i)}_d} Y_{0} p_J(Y_{0}) p_g(Y_{0})}{\sum_{Y_{0}\in \mathcal{Y}^{(i)}_d} p_J(Y_{0}) p_g(Y_{0})} \\
    = & -\frac{Y_{i}}{1-\bar{\alpha}_{i}} + \frac{\sqrt{\bar{\alpha}_{i}}}{1-\bar{\alpha}_{i}}\bar{Y}^{(0)}, \\
    \mathrm{where} \quad \bar{Y}^{(0)} = & \frac{\sum_{Y_{0} \in \mathcal{Y}^{(i)}_d} Y_{0} w(Y_{0})}{\sum_{Y_{0} \in \mathcal{Y}^{(i)}_d} w(Y_{0})}, \quad w(Y_{0}) = p_J(Y_{0})p_g(Y_{0})
\end{aligned}
$$

![MBD for Trajectory Optimization](assets/blog/mbd/assetstraj_opt_algo.png)

Up to now, we have introduced the full version of MBD for trajectory optimization. However, there is still one reminding issue: due to the combinatorial nature of $p_g(Y)$, we might still sampling from infeasible trajectory space. To address this issue, we introduce data to guide the diffusion process.

### Augment MBD with Diverse Data

Even though MBD is data-free, it can still leverage diverse data more effectively with the help of model. The key idea is to use data as a regularization term to guide the diffusion process to the high-likelihood solution while allowing to use model further to refine the solution.

With the demonstration data, we redefine our target distribution as:

$$p'_0(Y^{(0)}) \propto (1-\eta) p_d(Y^{(0)}) p_J(Y^{(0)}) p_g(Y^{(0)}) + \eta p_{\text{demo}}(Y^{(0)}) p_J(Y_{\text{demo}}) p_g(Y_{\text{demo}})$$

where

$$
\eta = \begin{cases}
        1 & p_d(Y^{0})p_J(Y^{0})p_g(Y^{0}) < p_\text{demo}(Y^{0})p_J(Y_\text{demo})p_g(Y_\text{demo}) \\
        0 & p_d(Y^{0})p_J(Y^{0})p_g(Y^{0}) \ge p_\text{demo}(Y^{0})p_J(Y_\text{demo})p_g(Y_\text{demo}).
        \end{cases}
$$

$\eta$ is a constant to balance the model and the demonstration.
Here, we have introduced two extra constant terms $p_J(Y_{\text{demo}})$ and $p_g(Y_{\text{demo}})$ to ensure that the demonstration likelihood is properly scaled to match the model likelihood $p_0(Y^{(0)})$.

Example: MBD with data vs. without data on a nonconvex function with constraints.
We want MBD converge to the optimal point 8 with the help of demonstration data. Although the demostration point is not optimal, MBD can still converge to the optimal point with the guidance of the demonstration data. Here data serves as a regularization term to guide the diffusion process to the negative optimal point while allowing to use model further to refine the solution.

![data augmented](assets/blog/mbd/assetsdata-augmented.png)

## Results: How does MBD Perform?

### MBD for Zeroth Order Optimization

Performance of MBD on high-dimensional black-box optimization benchmarks. MBD outperforms other Gaussian Process-based Bayesian Optimization methods by 23%.

![Black Box Optimization](assets/blog/mbd/assetsblackbox.png)

### MBD for Trajectory Optimization

We evalute our method on a set of challenging non-continous tasks, involving long-horizon push-T tash, which is generally considered as a challenging task even for RL.

![non-continous task](assets/blog/mbd/assetstask_joint.gif)

Our benchmark shows that MBD outperforms PPO by 59%:

![Reward Comparison](assets/blog/mbd/assetsresult_rew.png)

Achieving that performance, MBD only requires 10% computational time:

![Time Comparison](assets/blog/mbd/assetsresult_time.png)

For more visual results, please check [our website](https://lecar-lab.github.io/mbd/).

### MBD with Diverse Data

We also evaluate the performance of MBD with data augmentation on the Car UMaze Navigation and Humanoid Jogging tasks to see how partial and dynamically infeasible data can help the exploration of MBD and regularize the solution by steering the diffusion process.

![reference task](assets/blog/mbd/assetstask_reference.png)

## Takeaway

1. MBD achieve generalizable trajectory optimization by leveraging model information (i.e. we can evaluate the cost function and dynamics).
2. MBD can use diverse quality data more effectively to generate high-quality trajectories.
