# EVA8_assignments

This contains the jupyter notebooks for assignment solutions to EVA-8 exercies:

1. For assignment 2.5, the corresponding notebook is [assignment_2p5.ipynb](https://github.com/raghuch/EVA8_assignments/blob/main/assignment_2p5.ipynb). For adding an random integer input, I have chosen one-hot encoding from the torch.nn.functional package, and it takes a shape similar to the batch input:
```int_input = torch.randint(10, (data.shape[0], 1)).to(device)```

and finally in the forward function of the CNN, we return the two outputs (digit and sum) as:

``` pred = F.log_softmax(x)
        return pred, (pred.argmax(dim=1, keepdim=True) + int_input)
```
        

The training output shows that the accuracy achieved is above 99%



Colab logs for assignment 2.5:

|Timestamp|Level|Message|
|---|---|---|
|Jan 7, 2023, 7:29:47 AM|INFO|Kernel started: 4aeff88b-a8cd-4711-a651-788eb0271d23|
|Jan 7, 2023, 7:29:18 AM|INFO|Use Control-C to stop this server and shut down all kernels \(twice to skip confirmation\)\.|
|Jan 7, 2023, 7:29:18 AM|INFO|http://172\.28\.0\.12:9000/|
|Jan 7, 2023, 7:29:18 AM|INFO|The Jupyter Notebook is running at:|
|Jan 7, 2023, 7:29:18 AM|INFO|Serving notebooks from local directory: /|
|Jan 7, 2023, 7:29:18 AM|INFO|google\.colab serverextension initialized\.|
|Jan 7, 2023, 7:29:18 AM|INFO|Use Control-C to stop this server and shut down all kernels \(twice to skip confirmation\)\.|
|Jan 7, 2023, 7:29:18 AM|WARNING|    	/root/\.jupyter/jupyter\_notebook\_config\.json|
|Jan 7, 2023, 7:29:18 AM|INFO|http://172\.28\.0\.2:9000/|
|Jan 7, 2023, 7:29:18 AM|WARNING|    	/root/\.local/etc/jupyter/jupyter\_notebook\_config\.json|
|Jan 7, 2023, 7:29:18 AM|INFO|The Jupyter Notebook is running at:|
|Jan 7, 2023, 7:29:18 AM|WARNING|    	/usr/etc/jupyter/jupyter\_notebook\_config\.json|
|Jan 7, 2023, 7:29:18 AM|INFO|Serving notebooks from local directory: /|
|Jan 7, 2023, 7:29:18 AM|INFO|google\.colab serverextension initialized\.|
|Jan 7, 2023, 7:29:18 AM|WARNING|    	/usr/local/etc/jupyter/jupyter\_notebook\_config\.json|
|Jan 7, 2023, 7:29:18 AM|WARNING|    	/usr/local/etc/jupyter/jupyter\_notebook\_config\.d/panel-client-jupyter\.json|
|Jan 7, 2023, 7:29:18 AM|WARNING|    	/root/\.jupyter/jupyter\_notebook\_config\.json|
|Jan 7, 2023, 7:29:18 AM|WARNING|    	/root/\.local/etc/jupyter/jupyter\_notebook\_config\.json|
|Jan 7, 2023, 7:29:18 AM|WARNING|    	/usr/etc/jupyter/jupyter\_notebook\_config\.json|
|Jan 7, 2023, 7:29:18 AM|WARNING|    	/etc/jupyter/jupyter\_notebook\_config\.json|
|Jan 7, 2023, 7:29:18 AM|WARNING|    	/usr/local/etc/jupyter/jupyter\_notebook\_config\.json|
|Jan 7, 2023, 7:29:18 AM|WARNING|    	/usr/local/etc/jupyter/jupyter\_notebook\_config\.d/panel-client-jupyter\.json|
|Jan 7, 2023, 7:29:18 AM|WARNING|    	/etc/jupyter/jupyter\_notebook\_config\.json|
|Jan 7, 2023, 7:29:18 AM|INFO|Writing notebook server cookie secret to /root/\.local/share/jupyter/runtime/notebook\_cookie\_secret|
|Jan 7, 2023, 7:29:18 AM|WARNING|Notebook version 5 is no longer maintained\. Please upgrade to version 6 or later\.|
|Jan 7, 2023, 7:29:18 AM|INFO|Writing notebook server cookie secret to /root/\.local/share/jupyter/runtime/notebook\_cookie\_secret|
|Jan 7, 2023, 7:29:18 AM|WARNING|Notebook version 5 is no longer maintained\. Please upgrade to version 6 or later\.|
