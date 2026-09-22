.. _install:

Installation
============

Install using ``pip``
---------------------

MNeuEventLib can be installed like most other python packages using the ``pip`` package manager.

On a machine with ``pip`` installed, simply run

.. code-block::

   pip install MNeuEventLib


Install from source
-------------------

MNeuEventLib is built with the `maturin <https://www.maturin.rs/>`_ . build system
Compilation also requires the `rustup <https://rustup.rs/>`_ toolchain.

To install locally, activate a virtual environment and run:

.. code-block::

   maturin develop --release
