import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt


def visualize_image(image_tensor):
    """
    Visualizes an image tensor.

    Args:
        image_tensor (Tensor): Image tensor of shape (3, H, W).
    """
    image_pil = TF.to_pil_image(image_tensor)
    plt.imshow(image_pil)
    plt.axis('off')
    plt.show()