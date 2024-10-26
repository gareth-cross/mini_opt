# mini_opt

![Linux workflow status](https://github.com/gareth-cross/mini_opt/actions/workflows/linux.yml/badge.svg?branch=main)
![Windows workflow status](https://github.com/gareth-cross/mini_opt/actions/workflows/windows.yml/badge.svg?branch=main)

`mini_opt` is a small C++ implementation of constrained non-linear least squares. I implemented this for fun, and use it to solve toy problems that interest me. The solver supports:
- Non-linear cost functions and equality constraints.
- Box inequality constraints via interior-point method.
- Line search using polynomial approximation or armijo backtracking.
- Levenberg marquardt.

## Building

To build and run tests:

```
git submodule update --init
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build .
ctest
```

Optional serialization of optimization outputs (using [nlohmann](https://github.com/nlohmann/json)) is enabled by passing `-DMINI_OPT_SERIALIZATION=ON`.

## References

The reference used for this project is:

> "Numerical Optimization, Second Edition", Jorge Nocedal and Stephen J. Wright

The implementation is mostly based on Chapters 18 and 19 of this book.

## TODOs:

- [ ] Add support for non-linear inequality constraints. Only diagonal box constraints are supported at the moment.
- [ ] Add sparse versions of the solvers. For now I only support dense problems.

---

Project is licensed under MIT License.
