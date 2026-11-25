import torchvision.datasets as dset
dset.CIFAR100(root='data', train=True, download=True); 
dset.CIFAR100(root='data', train=False, download=True)