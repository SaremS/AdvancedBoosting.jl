# Parameterizable Transformations

Transformations that can contain parameters, e.g.
linear transformations or neural networks. The respective
transformation also needs to be differentiable with respect
to both input and parameters in order to properly optimize
the boosting model and additional parameters.

All transformations should be applicable using both the boosting
output and the actual input.

I.e. let ``f(\cdot)`` denote the function corresponding to a Gradient Boosting model, ``\mathbf{x}`` the vector of inputs and ``g`` a `ParameterizableTransform`. Then, ``g`` should be applicable as

```math
g(f(\mathbf{x}),\mathbf{x};\theta)
```

Where $\theta$ denotes a respective parameter vector of the transformation itself.

This increases flexibility and allows to, for example, build varying coefficient models where the varying coefficients are modelled by individual gradient boosting models, $f_1,...,f_n$:

```math
g(f_1(\mathbf{x}),...,f_n(\mathbf{x}),\mathbf{x};\theta)=\alpha_0 + f_1(\mathbf{x})\cdot x_1 + \cdots + f_n(\mathbf{x})\cdot x_n,
```

where $\theta\equiv [\alpha]$.


```@docs
ParameterizableTransform
IdentityTransform
SoftplusTransform
SigmoidTransform
MultiTransform
```
