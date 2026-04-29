import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from fer2013 import EmotionCNN

# Gradcam class
class GradCAM:
	"""
    Gradient-weighted Class Activation Mapping (Grad-CAM) implementation.

    This class generates localization maps highlighting important regions in an input
    image for a given model prediction by using gradients flowing into a target layer.

    Attributes:
        model (torch.nn.Module): The neural network model.
        target_layer (torch.nn.Module): The layer from which activations and gradients are extracted.
        activations (torch.Tensor): Stored forward activations from the target layer.
        gradients (torch.Tensor): Stored backward gradients from the target layer.
    """

	def __init__(self, model, target_layer):
		"""
        Initialize GradCAM with a model and target layer.

        Args:
            model (torch.nn.Module): The trained model to inspect.
            target_layer (torch.nn.Module): The convolutional layer to compute Grad-CAM for.
        """

		self.model = model
		self.target_layer = target_layer
		self.activations = None
		self.gradients = None

		self.target_layer.register_forward_hook(self.forward_hook)
		self.target_layer.register_backward_hook(self.backward_hook)

	def forward_hook(self, module, input, output):
		"""
        Forward hook to capture activations from the target layer.

        Args:
            module (torch.nn.Module): The layer being hooked.
            input (tuple): Input to the layer.
            output (torch.Tensor): Output (activations) from the layer.
        """

		self.activations = output

	def backward_hook(self, module, grad_input, grad_output):
		"""
        Backward hook to capture gradients from the target layer.

        Args:
            module (torch.nn.Module): The layer being hooked.
            grad_input (tuple): Gradients with respect to the layer inputs.
            grad_output (tuple): Gradients with respect to the layer outputs.
        """

		self.gradients = grad_output[0]

	def generate(self, x):
		"""
        Generate a Grad-CAM heatmap for the predicted class of input `x`.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W).

        Returns:
            tuple:
                cam (np.ndarray): Normalized Grad-CAM heatmap resized to input dimensions.
                class_idx (int): Predicted class index.
        """

		self.model.zero_grad()

		output = self.model(x)
		class_idx = output.argmax(dim=1)

		loss = output[:, class_idx]
		loss.backward()

		weights = self.gradients.mean(dim=(2, 3), keepdim=True)
		cam = (weights * self.activations).sum(dim=1, keepdim=True)

		cam = F.relu(cam)
		cam = F.interpolate(cam, size=x.shape[2:], mode='bilinear', align_corners=False)

		cam = cam.squeeze().detach().cpu().numpy()
		cam = (cam - cam.min()) / (cam.max() + 1e-8)

		return cam, class_idx.item()

# Confingurations
DATA_DIR = "test"
MODEL_PATH = "best.pth"
BATCH_SIZE = 1

transform = transforms.Compose([
	transforms.Grayscale(),
	transforms.Resize((48, 48)),
	transforms.ToTensor(),
	transforms.Normalize((0.5,), (0.5,))
])

dataset = datasets.ImageFolder(DATA_DIR, transform=transform)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

class_names = dataset.classes

# load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = EmotionCNN().to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

gradcam = GradCAM(model, target_layer=model.features[13])

for i, (images, labels) in enumerate(loader):
	images = images.to(device)

	cam, pred_class = gradcam.generate(images)

	img_np = images[0].cpu().numpy().squeeze()
	true_label = class_names[labels.item()]
	pred_label = class_names[pred_class]

	fig, axes = plt.subplots(1, 2, figsize=(8, 4))

	# Original
	axes[0].imshow(img_np, cmap='gray')
	axes[0].set_title(f"Original\nTrue: {true_label}")
	axes[0].axis('off')

	# Overlayed with the gradcam heatmap
	axes[1].imshow(img_np, cmap='gray')
	axes[1].imshow(cam, cmap='inferno', alpha=0.4)
	axes[1].set_title(f"Grad-CAM\nPred: {pred_label}")
	axes[1].axis('off')

	plt.tight_layout()
	plt.show()

	# stop early if you don't want to see all images
	if i == 10:
		break