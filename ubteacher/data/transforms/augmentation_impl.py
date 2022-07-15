import random
from PIL import ImageFilter


class GaussianBlur:
    """ Gaussian blur augmentation """

    def __init__(self, sigma=[0.1, 0.2]):
        
        self.sigma = sigma

    def __call__(self, x):

        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x