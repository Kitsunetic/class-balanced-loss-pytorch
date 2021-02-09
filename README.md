# class-balanced-loss-pytorch

```py
cb_loss = ClassBalancedLoss([1, 2, 3, 4, 5], 5, loss_type='focal', beta=0.9999, gamma=2.0)

x = torch.tensor(1, 5)
y = torch.ones(1, dtype=torch.long)

loss = cb_loss(x, y) # important: the order should be {input, target}
```
