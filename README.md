Backpropagation in OCaml
========================

Trying to implement backpropagation in OCaml using monads. I started with a first implementation of neural networks ([ocaml-nn](https://github.com/smimram/ocaml-nn/)) which was traditional but a bit heavy to my taste, which motivated me to try a more structured approach based on monadic-like approach (although there are no monads in the end).

## The backprop "functor"

The main idea here is that a type `'a` which can be _backpropagated_ is represented by `'a t` defined (in the `Backpropagatable` module) as

```ocaml
type 'a t = 'a * ('a -> unit)
```

The first component is the result, and the second component is a continuation specifying what we will do with the partial derivative of the error with respect to this variable (typically, we will modify references to perform gradient descent). On such a value you can call `eval` in order to exact the value and `descent` to perform a step of gradient descent.

Functions operating on backpropagatable values are usually defined from _differentiable functions_, which are roughly pairs consisting of a function (`'a -> 'b`) and its differential (`'b -> 'a`).

## See also

Articles:

- [Backprop as Functor: A compositional perspective on supervised learning](https://arxiv.org/abs/1711.10455)
- [Backpropagation in the Simply Typed Lambda-calculus with Linear Negation](https://arxiv.org/abs/1909.13768v2)
- [Demystifying Differentiable Programming: Shift/Reset the Penultimate Backpropagator](https://arxiv.org/abs/1803.10228)
- [Algorithmic Differentiation](https://ocaml.xyz/algodiff.html)

Libraries:

- [backprop Haskell library](https://backprop.jle.im/)
