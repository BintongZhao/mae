import random


class CropForWiFi:
    def __init__(self, output_shape=(128, 16)):
        self.output_shape = output_shape

    def __call__(self, spectrum):
        _, h, w = spectrum.shape
        new_h, new_w = self.output_shape

        if new_h > h or new_w > w:
            raise ValueError("Output shape must be smaller than input shape")

        top = random.randint(0, h - new_h)
        left = random.randint(0, w - new_w)

        return spectrum[:, top:top + new_h, left:left + new_w]