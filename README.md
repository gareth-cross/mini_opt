## mini_opt

`mini_opt` is a small C++ implementation of interior point optimization. I implemented this as a hobby project, and used it to experiment with inverse kinematic problems in Unreal Engine. The solver itself accepts a nonlinear L2 cost function, nonlinear equality constraints, and linear inequality constraints. It operates by linearizing the problem, and solving a sequence of QP problems using the interior-point method. A simple line-search method (either polynomial approximation, or armijo backtracking) is used to select the step size. Both L1 and L2 norms are supported for the equality constraints.

While I have endeavored to produce well-commented and unit-tested code, the solver behavior has primarily been evaluated in the context of one particular problem (IK). In practice, you would be unlikely to select interior point as a method of performing IK in games, owing to its complexity. My goal with the project was to learn more about the solver, and use Unreal Engine as a fun vehicle to visualize the problem. Note that this repository does not contain the UE4 project.

To build and run:
```
git submodule update --init
mkdir build
cd build
cmake .. -G Ninja -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build .
```

The reference used for this implementation is:
> "Numerical Optimization, Second Edition", Jorge Nocedal and Stephen J. Wright

### Project TODOs:
- [ ] Add support for non-linear inequality constraints. Only diagonal and linear constraints are supported at present.

---
Project is licensed under GPLv3. Copyright 2021 Gareth Cross.
