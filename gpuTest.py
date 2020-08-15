import torch
if torch.cuda.is_available():
    print(1)
    print(torch.cuda.device_count())
    print(torch.cuda.current_device())
    print(torch.cuda.get_device_name(0))

