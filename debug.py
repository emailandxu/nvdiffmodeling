from matplotlib import pyplot as plt
from torchvision.utils import make_grid, save_image

save = lambda tensor, name : save_image(make_grid(tensor.permute(0, 3, 1, 2), 4), f"{name}.png")
