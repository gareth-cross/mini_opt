## mini_opt

`mini_opt` is a small C++ implementation of interior point optimization. I implemented this as a hobby project, and used it to experiment with inverse kinematic problems in Unreal Engine. The solver itself accepts a nonlinear L2 cost function, nonlinear equality constraints, and linear inequality constraints. It operates by linearizing the problem, and solving a sequence of QP problems using the interior-point method. A simple line-search method (either polynomial aproximation, or armijo backtracking) is used to select the step size. Both L1 and L2 norms are supported for the equality constraints.

While I have endeavored to produce well-commented and unit-tested code, the solver _performance_ has primarily been evaluated in the context of one particular problem (IK). In practice, you would be unlikely to select interior point as a method of performing IK in games, owing to its complexity. My goal with the project was to learn more about the solver, and use Unreal Engine as a fun vehicle to visualize the problem. Note that this repository does not contain the UE4 project.

To build and run:
```
mkdir build
cd build
cmake ..
make test
```
The project depends on [libfmt](https://github.com/fmtlib/fmt), [geometry_utils](https://github.com/gareth-cross/geometry_utils), gtest, and Eigen. The cmake script _should_ automatically fetch the first three for you using git. Eigen will need to be *installed manually*. If you have a local checkout of Eigen, you can pass the cmake script `-DEIGEN_DIRECTORY=...` to specify where it resides.

The reference used for this implementation is:
> "Numerical Optimization, Second Edition", Jorge Nocedal and Stephen J. Wright

---
Project is lisenced under GPLv3. Copyright 2021 Gareth Cross.