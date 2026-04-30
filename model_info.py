from torchinfo import summary

from fer2013 import EmotionCNN

model = EmotionCNN()
summary(
    model,
    input_size=(1, 1, 48, 48),
    col_names=("input_size", "output_size", "num_params", "mult_adds"),
    depth=4,
)
