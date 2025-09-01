## L6: Fully-Connected Networks,optimization,initialization

#### Fully-connected network

$Z_{i+1}=\sigma_i (Z_iW_i+b_i^T),\\Z_1=X,\\
  h_\theta (X)=Z_{L+1},\\W_i\in \R^{n_i \times n_{i+1}}   ,\ b_i\in R^{n_{i+1}}$  (broadcast, repeat num_input times without copying)


#### Optimization

- primitive goal:minimize average loss
- gradient descent:$\theta=\theta-\alpha \triangledown _{\theta}f(\theta)$,modify params in steepest descent dir(gradient)
- Newton's method:$\theta=\theta-\alpha (\triangledown^2 _{\theta}f(\theta))^{-1}\triangledown _{\theta}f(\theta)$    (Hessian matrix)   (inefficient and uncommon)
- Momentum: take account of moving average of multiple previous grads,$\\ u_{t+1}=\beta u_t + (1-\beta) \triangledown _{\theta}f(\theta_t)\ ,\\ \theta_{t+1}=\theta_t-\alpha u_{t+1}$
- Unbiased momentum: initial grads will be smaller than later ones(assume grads are all 1 and u0=0,then ut=will be 1-beta^k), $\theta_{t+1}=\theta_t-\frac{\alpha u_{t+1}}{1-\beta^{t+1}}$
- Nesterov Momentum: compute momentum at next point,$ u_{t+1}=\beta u_t + (1-\beta) \triangledown _{\theta}f(\theta_t-\alpha u_t)$
- Adam: adaptive gradient methods
  - idea:scale of grads can vary in different params,adam estimates the scale and re-scale the gradients accordingly$\\ u_{t+1}=\beta_1 u_t + (1-\beta_1) \triangledown _{\theta}f(\theta_t) ,\\ v_{t+1}=\beta_2 v_t + (1-\beta_2) \triangledown _{\theta}f(\theta_t)^2 \ \ (elementwise\ square)\\ \theta_{t+1}=\theta_t-\frac{\alpha u_{t+1}}{\sqrt{v_{t+1}}+\epsilon}   \  (elementwise\ sqrt\ and\  div)$

#### Weight initialization

- Problem: how to init $W_i,b_i$
- In the backpass,$G_i =(G_{i+1}\circ \sigma_i^{'}(Z_iW_i) )W_i^T$,Wi=0 will let grads be 0.
- Random $W_i \sim \mathcal{N}(0,\sigma^2I) $, choice of variance will affect norms of $Z_i$ and grads
- idea: weight won't change too much after init,choice of init matters
- Xavier init: variance before and after Y=WX+B should be the same,so var of W is 1/n
- Kaiming init:half weights after ReLU will be 0,so $\sigma^2=\frac{2}{n}I$