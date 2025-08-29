## L2:ML Refresher & Softmax Regression

### ML Revision

- Concept: Data-driven
- Elements:hypothesis class(how to map inputs to outputs,with parameters),loss function(how well answer fits),optimization method(how to minimize the loss)

----

### Example:Softmax Regression

- k-class training:x=input,y=1-k,output
- n:input dim, k:output dim, m:training set size
- e.g. in MNIST, n=28*28=784(grey scalar for every pixel),k=10(10 digits),m=60000
- hypothesis:$h:\R^n\to \R^k$,output means likelihood to be each class

--

linear hypothesis:$h_{\theta}(x)=\theta^T x ,\theta\in \R^{n \times k}$

Matrix batch annotation:
$X\in \R^{m*n} = \begin{bmatrix}
    x^{(1)T}\\
    x^{(2)T}\\
    ...\\
    x^{(m)T}
\end{bmatrix}$,m=batch size,n=input dim

$Y\in \{1,2,...,k\}^m =\begin{bmatrix}
    y^{(1)}\\
    ...\\
    y^{(m)}
\end{bmatrix}$

$h_{\theta}(X)=\begin{bmatrix}
    h_{\theta}(x^{(1)})^T\\
    ...\\
    h_{\theta}(x^{(m)})^T
\end{bmatrix}=X\cdot \theta$

Loss func:

Zero_one version:(l in 花体)
$\mathcal{l}_{err}(h(x),y)=\begin{cases}
    0 ,if \ \ argmax_i\ (h(x))_i = y\ \ \ \  (biggest\ entry\ corresponds\ to\ the\ label)\\
    1 ,otherwise
\end{cases}$

Drawback:not differentiable

Modified ver:softmax/cross_entropy loss

$z_i=p(label=i)=\frac{exp(h_i(x))}{\sum exp(h_j(x))}=normalize(exp(h(x)))$(to continuous prob)

$l_{ce}(h(x),y)=-log\ p(label=y)=-h_y(x)+log(\sum exp(h_j(x)))$

softmax regression: minimize $\frac{\sum l(h_{\theta}(x),y)}{m}$ over $\theta$ 

Optimization:gradient descent

Notation:
$\triangledown _\theta f(\theta) \in \R ^{n\times k} =\begin{bmatrix}
    \frac{\partial f(\theta)}{\partial \theta _{11}}\dots \frac{\partial f(\theta)}{\partial \theta _{1k}}\\
    \dots \\
    \frac{\partial f(\theta)}{\partial \theta _{n1}}\dots \frac{\partial f(\theta)}{\partial \theta _{nk}}\\
\end{bmatrix}$,
where theta in f() means args of f,subscript means grad over theta

idea:
$\theta=\theta-\alpha \triangledown _{\theta}f(\theta)$



SGD: sample a minibatch of B samples,then update params by average grad
$\theta=\theta-\frac{\alpha}{B} \sum_{i=1}^{B}\triangledown _{\theta} l(h_{\theta}(x^{(i)}),y^{(i)})$

Question: How to compute gradfor cross-entropy,

$\frac{\partial l(h,y)}{\partial h_i}=\begin{cases}
    -1+\frac{exp(h_i)}{\sum exp(h_j)} ,i=y\\
    \frac{exp(h_i)}{\sum exp(h_j)} ,  \ \ other
\end{cases}$

$\triangledown _h l(h,y)=normalize(exp(h(x)))-e_y(unit\ vec)=z-e_y$

$ \triangledown _{\theta} l(\theta^Tx,y)$?

~~By taking everything like scalar and validating dims,~~

=$x\cdot (z-e_y)^T$

for batch X,

=$X^T\cdot (Z-I_y)$/m    (Z norm by row,each row of I means e_y)