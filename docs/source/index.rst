.. MNeuEventLib documentation master file, created by
   sphinx-quickstart on Thu Jul  2 12:49:57 2026.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

MNeuEventLib
============

MNeuEventLib is a Python package (written in Rust) for processing ISIS event data into a NeXuS
version 2 histogram file. This processing may include filtering the events based on:

- the time they occurred;
- on the values of auxiliary logs such as sample logs, warnings, or vetos;
- or on a high-pass filter of event amplitudes per detector.

It is primarily for muon event data, and the histogram files are
currently only compatible with `WiMDA <https://shadow.nd.rl.ac.uk/wimda/>`_.

Get started with MNeuEventLib 
-----------------------------

Follow these guides to get started:

* :ref:`install`: Learn how to install MNeuEventLib.

* :ref:`tutorials`: Learn how to use MNeuEventLib through guided tutorials.



Learn more
----------

* :ref:`how-to`: Explore specific features and workflows.

* `API Reference <_static/api/doc/mneueventlib/index.html>`_: The API reference for developers is available here.


Get more help
-------------

The easiest way to get help with the project is to start discussions or open an issue on `Github <https://github.com/ISISMuon/MNeuEventLib>`_.

.. toctree::
   :hidden:

   Home <self>

.. toctree::
   :hidden:
   :titlesonly:

   install
   tutorials/index
   how-to/index
   api
