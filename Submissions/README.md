# QP Solver
> Yunxiang He,  Kaixing Zhang, Ziyang Wu

所有的子函数均在 `QP_Solver.py` 中：
- 默认使用求解器 OSQP（基于 ADMM 的求解器），我们实现的 OSQP 可以直接解决 QP 问题，并且效率较高。
- 可以选择使用 ADAL 或者 IRWA 进行求解。这两个方法单独使用只能解决 QP 的 Exact Expenalty Subproblem，我们设计了合适的迭代方法来确保收敛到全局最优解。
- 对于具体的算法超参，我们在 `QP_Solver.py` 中提供了默认值，如果 TA 需要调整进行测试，请根据具体问题直接在代码里进行修改。
- 我们在 Submission 里并未提供我们的测试用例。