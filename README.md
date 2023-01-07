# EVA8_assignments

This contains the jupyter notebooks for assignment solutions to EVA-8 exercies:

1. For assignment 2.5, the corresponding notebook is [assignment_2p5.ipynb](https://github.com/raghuch/EVA8_assignments/blob/main/assignment_2p5.ipynb). For adding an random integer input, I have chosen one-hot encoding from the torch.nn.functional package, and it takes a shape similar to the batch input:
```int_input = torch.randint(10, (data.shape[0], 1)).to(device)```

and finally in the forward function of the CNN, we return the two outputs (digit and sum) as:

``` pred = F.log_softmax(x)
        return pred, (pred.argmax(dim=1, keepdim=True) + int_input)```
        

The training output shows that the accuracy achieved is above 99%
