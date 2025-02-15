import tensorflow as tf

print(f'\nTensorflow version = {tf.__version__}\n')
print(f'\n{tf.config.list_physical_devices("GPU")}\n')
print(f'\n{tf.sysconfig.get_build_info()}\n')


import torch

print(f'\nAvailable cuda = {torch.cuda.is_available()}')
print(f'\nGPUs availables = {torch.cuda.device_count()}')
#print(f'\nCurrent device = {torch.cuda.current_device()}')
print(f'\nCurrent Device location = {torch.cuda.device(0)}')
print(f'\nName of the device = {torch.cuda.get_device_name(0)}')