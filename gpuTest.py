import torch
if torch.cuda.is_available():
    print(1)
    print(torch.cuda.device_count())
    print(torch.cuda.current_device())
    print(torch.cuda.get_device_name(0))


import torch
 
a = torch.tensor([1, 2, 3.], requires_grad=True)
print(a.grad)
out = a.sigmoid()
print(out)
 
#添加detach(),c的requires_grad为False
c = out.detach().numpy()
d = c.copy()
d[0] = 1
print(d)
 
#这时候没有对c进行更改，所以并不会影响backward()
out.sum().backward()
print(a.grad)
