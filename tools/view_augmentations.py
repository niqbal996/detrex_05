import numpy as np
import matplotlib.pyplot as plt
from glob import glob
import os
import cv2
import random
import albumentations as A
root = '/mnt/e/datasets/sugarbeet_syn_v6'

output_json_path = '/mnt/e/datasets/sugarbeet_syn_v6/filtered_annotations.json'
images = sorted(glob(os.path.join(root, 'images/*.png')))
reference_image = cv2.cvtColor(cv2.imread('/mnt/e/datasets/phenobench/train/images/05-15_00097_P0030692.png'), cv2.COLOR_BGR2RGB)
# reference_image = cv2.cvtColor(cv2.imread('/mnt/e/datasets/phenobench/train/images/05-26_00174_P0034117.png'), cv2.COLOR_BGR2RGB)
# reference_image = cv2.cvtColor(cv2.imread('/mnt/e/datasets/phenobench/train/images/06-05_00070_P0037989.png'), cv2.COLOR_BGR2RGB)

reference_images = [
    '/mnt/e/datasets/phenobench/train/images/05-15_00097_P0030692.png',
    '/mnt/e/datasets/phenobench/train/images/05-26_00174_P0034117.png',
    '/mnt/e/datasets/phenobench/train/images/06-05_00070_P0037989.png'
]

def cv_reader(path):
    return cv2.cvtColor(cv2.imread(random.choice(path)), cv2.COLOR_BGR2RGB)

transform = A.Compose([
    # A.ChromaticAberration(primary_distortion_limit=0.2, secondary_distortion_limit=0.2, mode='random', interpolation=1, always_apply=True),
    # A.UnsharpMask(blur_limit=(5,9), sigma_limit=0.0, alpha=(0.4, 0.7), threshold=10, always_apply=True),
    # A.RandomToneCurve(scale=0.1, per_channel=False, always_apply=True),   # yes
    A.Posterize(num_bits=4, always_apply=True),   # yes
    # A.ISONoise(color_shift=(0.02, 0.06), intensity=(0.1, 0.2), always_apply=True), # yes
    # A.FDA([reference_image], beta_limit=(0, 0.005), read_fn=lambda x: x, always_apply=True),
    # A.FDA([reference_images], beta_limit=(0, 0.005), read_fn=lambda x: cv_reader(x), always_apply=True),
    A.PixelDistributionAdaptation([reference_images], blend_ratio=(0.1, 0.3), read_fn=lambda x: cv_reader(x), transform_type='pca', always_apply=True),
    A.HistogramMatching([reference_images], blend_ratio=(0.1, 0.3), read_fn=lambda x: cv_reader(x), always_apply=True),
])

def plot_histogram(image, ax, title):
    colors = ('r', 'g', 'b')
    for i, color in enumerate(colors):
        hist = cv2.calcHist([image], [i], None, [256], [0, 256])
        ax.plot(hist, color=color)
    ax.set_xlim([0, 256])
    ax.set_title(title)

def plot_fourier_spectrum(image, ax, title):
    # Convert to grayscale if it's a color image
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image
    
    # Compute the 2D Fourier Transform
    f = np.fft.fft2(gray)
    
    # Shift the zero-frequency component to the center of the spectrum
    fshift = np.fft.fftshift(f)
    
    # Compute the magnitude spectrum
    magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1)
    
    # Display the magnitude spectrum
    im = ax.imshow(magnitude_spectrum, cmap='gray')
    ax.set_title(title)
    return im

os.makedirs(os.path.join(root, 'augmented_posterize'), exist_ok=True)
new_root = os.path.join(root, 'augmented_posterize')
for image_path in images[:16]:
    # Read an image with OpenCV and convert it to the RGB colorspace
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Randomly select a reference image
    # reference_image_path = random.choice(reference_images)
    # reference_image = cv_reader(reference_image_path)

    # Augment an image
    transformed = transform(image=image)
    transformed_image = transformed["image"]

    basename = os.path.basename(image_path)
    cv2.imwrite(os.path.join(new_root, basename[:-4]+'.jpg'), cv2.cvtColor(transformed_image, cv2.COLOR_RGB2BGR))
    # Create a figure with three rows: images, histograms, and Fourier spectrums
    # fig, axes = plt.subplots(3, 3, figsize=(15, 15))

    # # Display images
    # axes[0, 0].imshow(image)
    # axes[0, 0].set_title('Original Image')
    # axes[0, 0].axis('off')

    # axes[0, 1].imshow(reference_image)
    # axes[0, 1].set_title('Reference Image')
    # axes[0, 1].axis('off')

    # axes[0, 2].imshow(transformed_image)
    # axes[0, 2].set_title('Transformed Image')
    # axes[0, 2].axis('off')

    # # Plot histograms
    # plot_histogram(image, axes[1, 0], 'Original Histogram')
    # plot_histogram(reference_image, axes[1, 1], 'Reference Histogram')
    # plot_histogram(transformed_image, axes[1, 2], 'Transformed Histogram')

    # # Plot Fourier spectrums
    # im1 = plot_fourier_spectrum(image, axes[2, 0], 'Original Fourier Spectrum')
    # im2 = plot_fourier_spectrum(reference_image, axes[2, 1], 'Reference Fourier Spectrum')
    # im3 = plot_fourier_spectrum(transformed_image, axes[2, 2], 'Transformed Fourier Spectrum')

    # # Add colorbars for Fourier spectrums
    # fig.colorbar(im1, ax=axes[2, 0])
    # fig.colorbar(im2, ax=axes[2, 1])
    # fig.colorbar(im3, ax=axes[2, 2])

    # # Adjust the layout and display the plot
    # plt.tight_layout()
    # plt.show()