# Exact Penalty Subproblem
## Description
$$
g^Tx + \frac{1}{2} x^T H x + \sum_{i\in\mathcal{E}}\vert a_i^Tx+b_i\vert + \sum_{i\in\mathcal{I}}^m \max(0, a_i^Tx + b_i)
$$

## Input
- $g\in\mathbb{R}^n$
- $H\in\mathbb{R}^{n\times n}$
- $a_i\in\mathbb{R}^n, i\in\mathcal{E}\cup \mathcal{I}$
- $b_i\in\mathbb{R}, i\in\mathcal{E}\cup \mathcal{I}$

## Easy Testcase
```lingo
@VARIABLES:
	x1, x2;

MIN = 0.5 * (x1*x1+x2*x2)+x1;

2*x1 + 2*x2 = 3;
x1 <= 4;
x2 <= 2;
```
### QP Problem
- $n=2$
- $H=\begin{bmatrix}1 & 0\\0 & 1\end{bmatrix}, g=\begin{bmatrix}1\\0\end{bmatrix}$
  - $f(x) = x_1  + \frac{1}{2}(x_1^2 + x_2^2)$
- $\mathcal{E}=\{1\}, \mathcal{I}=\{2, 3\}$
- $A = \begin{bmatrix}2 & 2\\1 & 0\\0 & 1\end{bmatrix}, b=\begin{bmatrix}-3\\-4\\-2\end{bmatrix}$
  - $2x_1 + 2 x_2 = 3\quad\bigg(x_1+2x_2+(-3) = 0\bigg)$
  - $x_1\leq 4\quad\bigg(x_1+(-4)\leq 0\bigg)$
  - $x_2\leq 2\quad\bigg(x_2+(-2)\leq 0\bigg)$
### QP Solution(By LINGO)
- Objective value: 1.062500
- x1 = 0.25, x2 = 1.250000
### Exact Penalty Subproblem
```lingo
MIN = 0.5 * (x1*x1+x2*x2)+x1 + @abs(2*x1+2*x2-3) + @smax(x1-4, 0) + @smax(x2-2, 0);
```
优化结果和 QP 问题一致

## Mid Dimension Testcase
### Testcase
```
n: 4
m: 8
H: [[0.01, 0.0, 0.0, 0.0], [0.0, 0.01, 0.0, 0.0], [0.0, 0.0, 0.01, 0.0], [0.0, 0.0, 0.0, 0.17012579140824938]]
g: [1.8675579901499675, -0.977277879876411, 0.9500884175255894, -0.1513572082976979]
AI: [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, -0.10321885179355784], [0.0, 0.0, 0.41059850193837233, 0.0], [-0.0, -0.0, -0.0, -0.0], [-0.0, -0.0, -0.0, -0.0], [-0.0, -0.0, -0.0, 0.10321885179355784], [-0.0, -0.0, -0.41059850193837233, -0.0]]
bI: [-0.02021839744032572, -0.832619845547938, -0.7781567509498505, -0.8700121482468192, -0.4319554389060677, -0.07440336170733897, -0.9289639418021131, -0.9128707002984593]
AE: None
bE: None
```
### LINGO Exact Penalty Subproblem
```lingo
MIN = 0.5 * (0.01 * x1 * x1 + 0.01 * x2 * x2 + 0.01 * x3 * x3 + 0.17012579140824938 * x4 * x4 ) + (1.8675579901499675 * x1 + -0.977277879876411 * x2 + 0.9500884175255894 * x3 + -0.1513572082976979 * x4 ) + @smax(0, + -0.02021839744032572)  + @smax(0, + -0.832619845547938)  + @smax(0, -0.10321885179355784 * x4 + -0.7781567509498505)  + @smax(0, 0.41059850193837233 * x3 + -0.8700121482468192)  + @smax(0, + -0.4319554389060677)  + @smax(0, + -0.07440336170733897)  + @smax(0, 0.10321885179355784 * x4 + -0.9289639418021131)  + @smax(0, -0.41059850193837233 * x3 + -0.9128707002984593) ;
```

### Result:
```
Objective value:   -47.82093

Variable           Value        Reduced Cost
  X1        0.000000            1.867558
  X2        97.72761          -0.3326476E-06
  X3       0.1038948E-05       0.9500884
  X4       0.8880139           0.1377715E-04
```