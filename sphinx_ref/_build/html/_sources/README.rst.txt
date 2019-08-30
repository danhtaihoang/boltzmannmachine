Network Inference in Stochastic Systems
=======================================

Introduction
-----------------------------
We developed a data-driven approach for network inference in stochastic systems, Free Energy Minimazation (FEM). From data comprising of configurations of variables, we determine the interactions between them in order to infer a predictive stochastic model. FEM outperforms other existing methods such as variants of mean field approximations and Maximum Likelihood Estimation (MLE), especially in the regimes of large coupling variability and small sample sizes. Besides better performance, FEM is parameter-free and significantly faster than MLE.

Interactive notebook
-----------------------------
Use Binder to run our code online. You are welcome to change the parameters and edit the jupyter notebooks as you want. 

.. image:: https://mybinder.org/badge.svg
   :target: https://mybinder.org/v2/gh/danhtaihoang/network-inference/master?filepath=sphinx%2Fcodesource

Links
----------------------------
Code Documentation
    https://danhtaihoang.github.io/network-inference

Code Source
    https://github.com/danhtaihoang/network-inference

Reference
----------------------------
Danh-Tai Hoang, Juyong Song, Vipul Periwal, and Junghyo Jo, "Causality inference in stochastic systems from neurons to currencies: Profiting from small sample size", `Physical Review E, 99, 023311 (2019) <https://journals.aps.org/pre/abstract/10.1103/PhysRevE.99.023311>`_.

Highlights
----------------------------
- **Inferring network with specific structures**

.. image:: figs/fig1.png
  :width: 420

- **Inferring neuronal network from experimental neuronal activities**

.. image:: figs/fig2.png
  :width: 420

- **Inferring currency network from currency exchange rates**

.. image:: figs/fig3.png
  :width: 420
