from matplotlib import pyplot as plt


show = lambda tensor, name: (plt.imshow(tensor.cpu().numpy()), plt.savefig(f"{name}.jpg"))