# Backpropagation

#### How do we find the directions to move $\theta$ so as to decrease the objective $R(\theta)$?

One need to calculate **gradient** of $R(\theta)$ evaluated at some current value $\theta = \theta^m$:

$\nabla R(\theta^m) = \frac{\partial R(\theta)}{\partial\theta} \biggr\rvert_{\theta = \theta^m}$

The idea of gradient descent is to move $\theta$ a little in the opposite direction:

$\theta^{m+1} \leftarrow \theta^m - \rho \nabla R(\theta^m)$,

where $\rho$ is the **learning rate**.

If the gradient vector is zero, then we may have arrived at a minimum of the objective.

#### Backpropagation:
* Allows us to compute gradients algorithmically
* Used by deep learning frameworks (TensorFlow, PyTorch, etc.) 

---

# Backpropagation: Example

<div class="grid grid-cols-[2fr_2fr]">
<div>
  <figure>
    <img src="/Backpropagation_Ex2.png" style="width: 350px !important;">
    <figcaption style="color:#b3b3b3ff; font-size: 11px; position: absolute;"><br>Ex. credits:
      <a href="https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1184/syllabus.html">https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1184</a>
    </figcaption>
  </figure>
</div>
<div>

$$
\boxed{
\begin{array}{rcl}
f(x, y, z) = (x + y) ~\mathrm{max}(y, z)\\
x = 1, ~y = 2, ~z = 0\\
\\
\mathrm{Find~}\frac{\partial{f}}{\partial{y}}
\end{array}
}
$$
</div>
</div>

<div class="grid grid-cols-[2fr_2fr]">
<div>

Forward propagation steps:<br>
$a = x + y$

$b = \mathrm{max}(y, z)$

$f = a \cdot b$

</div>
<div>

Local gradients:<br>
$\frac{\partial{a}}{\partial{x}} = 1$ $~~~\frac{\partial{a}}{\partial{x}} = 1$

$\frac{\partial{b}}{\partial{y}} = \boldsymbol{1} (y > z) = 1$ $~~~\frac{\partial{b}}{\partial{z}} = \boldsymbol{1} (y < z) = 0$

$$
\boxed{
\frac{\partial{f}}{\partial{a}} = b = 2~~~ \frac{\partial{f}}{\partial{b}} = a = 3
}
$$
</div>
</div>

$\frac{\partial{f}}{\partial{x}} = 2~~~~~~ \frac{\partial{f}}{\partial{y}} = 3 + 2 = 5~~~~~ \frac{\partial{f}}{\partial{z}} = 0$

---

# Backpropagation in PyTorch

```python {all}
# Initialize x, y and z to values 4, -3 and 5
x = torch.tensor(4., requires_grad=True)
y = torch.tensor(-3., requires_grad=True)
z = torch.tensor(5., requires_grad=True)

# Set q to sum of x and y, set f to product of q with z
q = x + y
f = q * z

# Compute the derivatives
f.backward()

# Print the gradients
print("Gradient of x is: " + str(x.grad))
print("Gradient of y is: " + str(y.grad))
print("Gradient of z is: " + str(z.grad))
```
The code above produces:
```python {all}
Gradient of x is: tensor(5.)
Gradient of y is: tensor(5.)
Gradient of z is: tensor(1.)
```