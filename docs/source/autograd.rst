.. currentmodule:: autograd

Autograd
*********

How it works
============
When you write::

    import numpy as np
    from slick_dnn.variable import Variable

    a = Variable(np.ones(3))
    b = Variable(np.ones(3))

    c = a + b

New Variable c is created.
It's :code:`c.data` is numpy array :code:`[2, 2, 2]`.
But it also tracks history of creation.

So it's :code:`backward_function` was set to :py:class:`Add.backward <slick_dnn.autograd.mathematical.Add.backward>`
and it's :code:`backward_variables` was set to :code:`[a, b]`

Fundamental classes
========================
.. automodule:: slick_dnn.autograd
    :members:
    :special-members: __call__

Mathematical
=============

All mathematical operations available for Variables

.. automodule:: slick_dnn.autograd.mathematical
    :members:


Activation Functions
====================

All activation functions available.

.. automodule:: slick_dnn.autograd.activations
    :members: