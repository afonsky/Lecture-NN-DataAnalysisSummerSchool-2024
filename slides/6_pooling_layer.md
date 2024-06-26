# Pooling Layer
<div></div>

The pooling layer (**POOL**) is a downsampling operation, typically applied after a convolution layer, which does some spatial invariance. Most popular kinds of pooling:
<div class="grid grid-cols-[1fr_1fr]">
<div>

* **Max pooling**
  * Selects maximum value of the current view
  * Preserve detected features
  * Most commonly used

```python {all}
m = nn.MaxPool2d(2, stride=2)
input = torch.randn(20, 16, 50)
output = m(input)
```
</div>
<div>
  <figure>
    <img src="/max-pooling-a.png" style="width: 200px !important;">
  </figure>  
</div>
</div>

<div class="grid grid-cols-[1fr_1fr]">
<div>

* **Average pooling**
  * Averages the values of the current view
  * Downsamples feature map
  * Used in LeNet
</div>
<div>
  <figure>
    <img src="/average-pooling-a.png" style="width: 200px !important;">
  </figure>
</div>
</div>
<span style="color:grey"><small>Gifs are from <a href="https://stanford.edu/~shervine/teaching/cs-230/cheatsheet-convolutional-neural-networks">Convolutional Neural Networks cheatsheet</a></small></span>