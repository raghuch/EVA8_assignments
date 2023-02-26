Assignment 9 solution for a naive transformer implementation with conv -> conv -> conv -> GAP -> Ultimus -> Ultimus -> Ultimus -> Ultimus -> FFC layer. I have used OCP to find the max LR and used one cycle policy along with Adam optimizer.

For this assignment, I haven't used any other transforms other than normalizing (so train set transforms and testset transforms are the same in this case).

The model file used for this implementation is present in the "models" sub dir of this repo [here](https://github.com/raghuch/EVA8_assignments/blob/main/models/model9_ultimus.py), and I have reused the CIFAR10 dataset class in utils dir.

I have used a jupyter notebook to import model, dataloaders etc.; the notebook is inside the "assignment9" dir inside the repo, [here](https://github.com/raghuch/EVA8_assignments/blob/main/assignment9/assignment9_notebook.ipynb).

The notebook itself contains the train and test losses and accuracies, but it is also present [here](https://github.com/raghuch/EVA8_assignments/blob/main/assignment9/assignment9_acc_plots.png)

![assignment9_acc_plots](https://user-images.githubusercontent.com/14867819/221433812-343f12f3-36c0-454d-a8e5-80f72c363e72.png)
