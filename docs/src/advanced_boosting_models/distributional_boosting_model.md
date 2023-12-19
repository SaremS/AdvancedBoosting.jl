Models the conditional distribution 

```math
p(\mathbf{y}||\mathbf{x})=p(\mathbf{y};\theta(\mathbf{x}))
```

where

```math
\theta(\mathbf{x})=\left(g_1(f_1(\mathbf{x})),...,g_m(f_m(\mathbf{x}))\right)^T,
```

i.e., the parameters of the conditional distribution are expressed by Gradient Boosting models $f_1,...,f_m$ and respective link-functions $g_1,...,g_m$.

```@docs
DistributionalBoostingModel
```
