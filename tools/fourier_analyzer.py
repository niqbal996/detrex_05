import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from PIL import Image
import cv2
from scipy.fft import fft2, fftshift, ifft2, ifftshift

class FourierImageAnalyzer:
    def __init__(self, initial_scale_window=30, max_scale_window=100, alpha_factor=0.0):
        self.scale_window = initial_scale_window
        self.max_scale_window = max_scale_window
        self.alpha = alpha_factor
    def _compute_fft(self, image):
        """Compute FFT for each channel of the image."""
        return [fftshift(fft2(image[:,:,i])) for i in range(3)]

    def _compute_magnitude_spectrum(self, fft_result):
        """Compute magnitude spectrum with scaling."""
        return [np.log(np.abs(f) + 1) / self.scale_window for f in fft_result]

    def _compute_phase_spectrum(self, fft_result):
        """Compute phase spectrum."""
        return [np.angle(f) for f in fft_result]

    def analyze_and_plot(self, image_path1, image_path2):
        """Analyze and plot Fourier transforms of two images."""
        img1 = np.array(Image.open(image_path1))
        img2 = np.array(Image.open(image_path2))

        fft1 = self._compute_fft(img1)
        fft2 = self._compute_fft(img2)

        mag1 = self._compute_magnitude_spectrum(fft1)
        mag2 = self._compute_magnitude_spectrum(fft2)

        phase1 = self._compute_phase_spectrum(fft1)
        phase2 = self._compute_phase_spectrum(fft2)

        self._plot_results(img1, img2, mag1, mag2, phase1, phase2)

    def _plot_results(self, img1, img2, mag1, mag2, phase1, phase2):
        """Plot the original images and their Fourier transforms."""
        fig, axes = plt.subplots(4, 3, figsize=(15, 20))
        channel_names = ['Red', 'Green', 'Blue']

        # Plot original images
        axes[0, 0].imshow(img1)
        axes[0, 0].set_title('Original Image 1')
        axes[0, 0].axis('off')
        axes[0, 1].imshow(img2)
        axes[0, 1].set_title('Original Image 2')
        axes[0, 1].axis('off')
        axes[0, 2].axis('off')  # Empty subplot for alignment

        for i in range(3):
            # Magnitude spectra
            axes[1, i].imshow(mag1[i], cmap='viridis')
            axes[1, i].set_title(f'Magnitude Spectrum 1 - {channel_names[i]}')
            axes[1, i].axis('off')
            axes[2, i].imshow(mag2[i], cmap='viridis')
            axes[2, i].set_title(f'Magnitude Spectrum 2 - {channel_names[i]}')
            axes[2, i].axis('off')

            # Phase spectra
            axes[3, i].imshow(phase1[i], cmap='hsv')
            axes[3, i].set_title(f'Phase Spectrum 1 - {channel_names[i]}')
            axes[3, i].axis('off')

        plt.tight_layout()
        plt.show()

    def plot_combined_magnitude(self, image_path1, image_path2):
        """Plot combined RGB magnitude spectra for two images."""
        img1 = np.array(Image.open(image_path1))
        img2 = np.array(Image.open(image_path2))

        fft1 = self._compute_fft(img1)
        fft2 = self._compute_fft(img2)

        mag1 = self._compute_magnitude_spectrum(fft1)
        mag2 = self._compute_magnitude_spectrum(fft2)

        combined_mag1 = np.stack(mag1, axis=-1)
        combined_mag2 = np.stack(mag2, axis=-1)

        fig, axes = plt.subplots(2, 2, figsize=(12, 12))

        axes[0, 0].imshow(img1)
        axes[0, 0].set_title('Original Image 1')
        axes[0, 0].axis('off')

        axes[0, 1].imshow(combined_mag1)
        axes[0, 1].set_title('Combined Magnitude Spectrum 1')
        axes[0, 1].axis('off')

        axes[1, 0].imshow(img2)
        axes[1, 0].set_title('Original Image 2')
        axes[1, 0].axis('off')

        axes[1, 1].imshow(combined_mag2)
        axes[1, 1].set_title('Combined Magnitude Spectrum 2')
        axes[1, 1].axis('off')

        plt.tight_layout()
        plt.show()
    
    def inverse_fourier_transform(self, image_path):
        """
        Perform inverse Fourier transform to reconstruct the original image.
        
        Args:
        image_path (str): Path to the input image
        
        Returns:
        numpy.ndarray: Reconstructed RGB image
        """
        # Load the image
        img = np.array(Image.open(image_path)).astype(np.float32)
        
        # Initialize an array to store the reconstructed image
        reconstructed_img = np.zeros_like(img)

        for i in range(3):  # For each color channel
            # Compute the forward FFT
            f = fftshift(fft2(img[:,:,i]))
            
            # Compute the inverse FFT
            reconstructed_channel = ifft2(ifftshift(f)).real
            
            # Add the reconstructed channel to the result
            reconstructed_img[:,:,i] = reconstructed_channel

        # Normalize the reconstructed image to [0, 255] range
        reconstructed_img = ((reconstructed_img - reconstructed_img.min()) / 
                             (reconstructed_img.max() - reconstructed_img.min()) * 255).astype(np.uint8)

        return reconstructed_img

    def plot_original_and_reconstructed(self, image_path):
        """
        Plot the original image and its reconstruction from Fourier transform.
        
        Args:
        image_path (str): Path to the input image
        """
        # Load the original image
        original_img = np.array(Image.open(image_path))
        
        # Reconstruct the image using inverse Fourier transform
        reconstructed_img = self.inverse_fourier_transform(image_path)

        # Create a figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

        # Plot original image
        ax1.imshow(original_img)
        ax1.set_title('Original Image')
        ax1.axis('off')

        # Plot reconstructed image
        ax2.imshow(reconstructed_img)
        ax2.set_title('Reconstructed Image')
        ax2.axis('off')

        plt.tight_layout()
        plt.show()

        # Compute and print the mean squared error
        mse = np.mean((original_img.astype(np.float32) - reconstructed_img.astype(np.float32)) ** 2)
        print(f"Mean Squared Error: {mse}")

    def fourier_analysis(self, image, color_space):
        """
        Perform Fourier analysis on an image in the specified color space.
        
        Args:
        image: Can be either a string (path to image file) or a PIL Image object.
        color_space: String specifying the color space (e.g., 'RGB', 'YCbCr', 'HSV', etc.)
        """
        if isinstance(image, str):
            image = Image.open(image)
        
        if not isinstance(image, Image.Image):
            raise ValueError("Input must be either a file path or a PIL Image object")
        
        # Convert to specified color space
        image = image.convert(color_space)
        
        # Convert image to numpy array
        image_array = np.array(image)
    
        # Separate color channels
        channels = [image_array[:,:,i] for i in range(image_array.shape[2])]
    
        # Perform Fourier Transform on each channel
        f_transforms = []
        for channel in channels:
            f_transform = fftshift(fft2(channel))
            f_transforms.append(f_transform)
    
        return f_transforms

    def swap_low_frequency(self, image1, image2, channel, color_space):
        """
        Swap the low frequency magnitude spectrum of the specified channel(s) between two images.
        
        Args:
        image1, image2: Can be either strings (paths to image files) or PIL Image objects.
        channel: Integer specifying which channel to swap (0, 1, 2, or -1 for all channels)
        color_space: String specifying the color space to use
        
        Returns:
        PIL.Image: Reconstructed image with swapped low frequency spectrum in the original color space.
        """
        # Open images if paths are provided
        if isinstance(image1, str):
            image1 = Image.open(image1)
        if isinstance(image2, str):
            image2 = Image.open(image2)
        
        # Ensure images are the same size
        if image1.size != image2.size:
            image2 = image2.resize(image1.size)
        
        # Perform Fourier analysis on both images
        f_transforms1 = self.fourier_analysis(image1, color_space)
        f_transforms2 = self.fourier_analysis(image2, color_space)
        
        # Get dimensions
        rows, cols = f_transforms1[0].shape
        center_row, center_col = rows // 2, cols // 2
        
        # Create a window for low frequency
        window = np.zeros((rows, cols), dtype=bool)
        window[center_row-self.scale_window:center_row+self.scale_window, 
               center_col-self.scale_window:center_col+self.scale_window] = True
        

        
        # Determine which channels to swap
        channels_to_swap = range(len(f_transforms1)) if channel == -1 else [channel]
        
        # Swap low frequency magnitude components of the specified channel(s)
        for ch in channels_to_swap:
            magnitude1 = np.abs(f_transforms1[ch])
            magnitude2 = np.abs(f_transforms2[ch])
            phase1 = np.angle(f_transforms1[ch])
            
            temp1 = magnitude1[window].copy()
            temp2 = magnitude2[window].copy()
            temp1 = temp1 * (1 - self.alpha)
            temp2 = temp2 * self.alpha
            temp = temp1 + temp2
            magnitude1[window] = temp 
            
            # Reconstruct the channel
            f_transforms1[ch] = magnitude1 * np.exp(1j * phase1)
        
        # Inverse Fourier Transform
        reconstructed_channels = []
        for f_transform in f_transforms1:
            channel = np.real(ifft2(ifftshift(f_transform)))
            channel = np.clip(channel, 0, 255).astype(np.uint8)
            reconstructed_channels.append(channel)
        
        # Combine channels
        reconstructed_image = np.stack(reconstructed_channels, axis=-1)
        
        # Convert back to original color space
        reconstructed_image = Image.fromarray(reconstructed_image, color_space)
        
        return reconstructed_image

    def interactive_visualization(self, image1, image2, channel, color_space):
        """
        Create an interactive visualization of the low frequency spectrum swapping.

        Args:
        image1, image2: Can be either strings (paths to image files) or PIL Image objects.
        channel: Integer specifying which channel to swap (0, 1, 2, or -1 for all channels)
        color_space: String specifying the color space to use
        """
        # Open images if paths are provided
        if isinstance(image1, str):
            image1 = Image.open(image1)
        if isinstance(image2, str):
            image2 = Image.open(image2)
        
        # Ensure images are the same size
        if image1.size != image2.size:
            image2 = image2.resize(image1.size)
        
        # Convert to numpy arrays
        img1_array = np.array(image1)
        img2_array = np.array(image2)

        # Create the figure and subplots
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
        plt.subplots_adjust(bottom=0.25)

        # Display original images
        ax1.imshow(img1_array)
        ax1.set_title('Image 1')
        ax1.axis('off')

        ax2.imshow(img2_array)
        ax2.set_title('Image 2')
        ax2.axis('off')

        # Initial reconstruction
        reconstructed = self.swap_low_frequency(image1, image2, channel, color_space)
        im3 = ax3.imshow(np.array(reconstructed.convert('RGB')))
        ax3.set_title(f'Reconstructed Image (Channel {"All" if channel == -1 else channel})')
        ax3.axis('off')

        # Create the slider
        ax_slider = plt.axes([0.2, 0.1, 0.6, 0.03])
        slider = Slider(ax_slider, 'Scale Window', 0, self.max_scale_window, 
                        valinit=self.scale_window, valstep=1)

        # Update function for slider
        def update(val):
            self.scale_window = int(slider.val)
            reconstructed = self.swap_low_frequency(image1, image2, channel, color_space)
            im3.set_array(np.array(reconstructed.convert('RGB')))
            fig.canvas.draw_idle()

        slider.on_changed(update)

        plt.show()

        
    def visualize_spectrum(self, image):
        """
        Visualize the magnitude and phase spectrum of an image in RGB and YCbCr color spaces.

        Args:
        image: Can be either a string (path to image file) or a PIL Image object.
        """
        if isinstance(image, str):
            image = Image.open(image)
        
        # Convert to RGB and YCbCr
        rgb_image = image.convert('RGB')
        ycbcr_image = image.convert('YCbCr')

        # Create subplots
        fig, axs = plt.subplots(4, 4, figsize=(20, 20))
        fig.suptitle('Fourier Analysis: Magnitude and Phase Spectrum', fontsize=16)

        color_spaces = [('RGB', rgb_image), ('YCbCr', ycbcr_image)]

        for row, (color_space, img) in enumerate(color_spaces):
            # Convert image to numpy array
            img_array = np.array(img)

            for channel in range(3):
                # Perform Fourier Transform
                f_transform = fftshift(fft2(img_array[:,:,channel]))

                # Compute magnitude spectrum
                magnitude = np.abs(f_transform)
                magnitude_log = np.log1p(magnitude)

                # Compute phase spectrum
                phase = np.angle(f_transform)

                # Plot magnitude spectrum
                axs[row*2, channel].imshow(magnitude_log, cmap='viridis')
                axs[row*2, channel].set_title(f'{color_space} Channel {channel+1} Magnitude')
                axs[row*2, channel].axis('off')

                # Plot phase spectrum
                axs[row*2+1, channel].imshow(phase, cmap='twilight')
                axs[row*2+1, channel].set_title(f'{color_space} Channel {channel+1} Phase')
                axs[row*2+1, channel].axis('off')

            # Plot original image
            axs[row*2, 3].imshow(img)
            axs[row*2, 3].set_title(f'Original {color_space}')
            axs[row*2, 3].axis('off')

            # Remove unused subplot
            fig.delaxes(axs[row*2+1, 3])

        plt.tight_layout()
        plt.show()