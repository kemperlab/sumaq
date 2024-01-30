# $\text{SUMA}\mathrm{Q}$: Spiced Up! More Accessible Quantum

![Build Status](https://img.shields.io/github/actions/workflow/status/kemperlab/sumaq/test.yml?style=for-the-badge)

A Python package for condensed matter physics simulations.

## Description

$\text{SUMA}\mathrm{Q}$ is a package designed to help users perform various calculations relevant to condensed matter physics. It comes with pre-defined Hamiltonians and allows users to build their own fermionic Hamiltonians. The package also supports generating, post-processing, and analyzing response functions. For time evolution of Hamiltonians, the package uses `qiskit` to implement quantum circuits and run jobs on real and fake backends. Additionally, there are built-in functions to perform common operations such as Fourier transforms, representing a matrix operator in terms of Pauli strings, or saving data in a human-readable format. It is important to note that this package is still in development.