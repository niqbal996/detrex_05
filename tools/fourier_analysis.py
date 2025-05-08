import torch
import numpy as np
from scipy.fft import fft2, fftshift
from PIL import Image
import cv2
import math
import matplotlib.pyplot as plt
from fourier_analyzer import FourierImageAnalyzer

def extract_ampl_phase(fft_im):
    # fft_im: size should be bx3xhxwx2
    fft_amp = fft_im[:,:,:,:,0]**2 + fft_im[:,:,:,:,1]**2
    fft_amp = torch.sqrt(fft_amp)
    fft_pha = torch.atan2( fft_im[:,:,:,:,1], fft_im[:,:,:,:,0] )
    return fft_amp, fft_pha

def low_freq_mutate( amp_src, amp_trg, L=0.1 ):
    _, _, h, w = amp_src.size()
    b = (  np.floor(np.amin((h,w))*L)  ).astype(int)     # get b
    amp_src[:,:,0:b,0:b]     = amp_trg[:,:,0:b,0:b]      # top left
    amp_src[:,:,0:b,w-b:w]   = amp_trg[:,:,0:b,w-b:w]    # top right
    amp_src[:,:,h-b:h,0:b]   = amp_trg[:,:,h-b:h,0:b]    # bottom left
    amp_src[:,:,h-b:h,w-b:w] = amp_trg[:,:,h-b:h,w-b:w]  # bottom right
    return amp_src

def low_freq_mutate_np( amp_src, amp_trg, L=0.1 ):
    a_src = np.fft.fftshift( amp_src, axes=(-2, -1) )
    a_trg = np.fft.fftshift( amp_trg, axes=(-2, -1) )

    _, h, w = a_src.shape
    b = (  np.floor(np.amin((h,w))*L)  ).astype(int)
    c_h = np.floor(h/2.0).astype(int)
    c_w = np.floor(w/2.0).astype(int)

    h1 = c_h-b
    h2 = c_h+b+1
    w1 = c_w-b
    w2 = c_w+b+1

    a_src[:,h1:h2,w1:w2] = a_trg[:,h1:h2,w1:w2]
    a_src = np.fft.ifftshift( a_src, axes=(-2, -1) )
    return a_src

def FDA_source_to_target(src_img, trg_img, L=0.1):
    # exchange magnitude
    # input: src_img, trg_img

    # get fft of both source and target
    fft_src = torch.rfft( src_img.clone(), signal_ndim=2, onesided=False ) 
    fft_trg = torch.rfft( trg_img.clone(), signal_ndim=2, onesided=False )

    # extract amplitude and phase of both ffts
    amp_src, pha_src = extract_ampl_phase( fft_src.clone())
    amp_trg, pha_trg = extract_ampl_phase( fft_trg.clone())

    # replace the low frequency amplitude part of source with that from target
    amp_src_ = low_freq_mutate( amp_src.clone(), amp_trg.clone(), L=L )

    # recompose fft of source
    fft_src_ = torch.zeros( fft_src.size(), dtype=torch.float )
    fft_src_[:,:,:,:,0] = torch.cos(pha_src.clone()) * amp_src_.clone()
    fft_src_[:,:,:,:,1] = torch.sin(pha_src.clone()) * amp_src_.clone()

    # get the recomposed image: source content, target style
    _, _, imgH, imgW = src_img.size()
    src_in_trg = torch.irfft( fft_src_, signal_ndim=2, onesided=False, signal_sizes=[imgH,imgW] )

    return src_in_trg

def FDA_source_to_target_np( src_img, trg_img, L=0.1 ):

    # exchange magnitude
    # input: src_img, trg_img

    src_img_np = src_img #.cpu().numpy()
    trg_img_np = trg_img #.cpu().numpy()

    # get fft of both source and target
    fft_src_np = np.fft.fft2( src_img_np, axes=(-2, -1) )
    fft_trg_np = np.fft.fft2( trg_img_np, axes=(-2, -1) )

    # extract amplitude and phase of both ffts
    amp_src, pha_src = np.abs(fft_src_np), np.angle(fft_src_np)
    amp_trg, pha_trg = np.abs(fft_trg_np), np.angle(fft_trg_np)

    # mutate the amplitude part of source with target
    amp_src_ = low_freq_mutate_np( amp_src, amp_trg, L=L )

    # mutated fft of source
    fft_src_ = amp_src_ * np.exp( 1j * pha_src )

    # get the mutated image
    src_in_trg = np.fft.ifft2( fft_src_, axes=(-2, -1) )
    src_in_trg = np.real(src_in_trg)

    return src_in_trg, (amp_src, pha_src), (amp_trg, pha_trg)

def plot_images(images, titles=None):
    """
    Plot multiple PIL images on a single figure.
    
    Parameters:
    images (list): List of PIL.Image objects to plot
    titles (list): Optional list of strings for image titles
    """
    num_images = len(images)
    
    # Calculate the number of rows and columns for the subplots
    num_cols = min(4, num_images)  # Max 4 images per row
    num_rows = math.ceil(num_images / num_cols)
    
    # Create a figure with subplots
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(4*num_cols, 4*num_rows))
    
    # If there's only one row, axes needs to be 2D for consistency
    if num_rows == 1:
        axes = axes.reshape(1, -1)
    # Plot each image
    for i, image in enumerate(images):
        if image.shape[0] == 3:
            image = np.transpose(image, (1, 2, 0))
        # if np.max(image) > 255:
        #     image[:, :, 0] /= np.max(image[:, :, 0])
        #     image[:, :, 1] /= np.max(image[:, :, 1])
        #     image[:, :, 2] /= np.max(image[:, :, 2])
        #     # image *= 255
        row = i // num_cols
        col = i % num_cols
        
        axes[row, col].imshow(image)
        axes[row, col].axis('off')  # Turn off axis numbers and ticks
        
        if titles and i < len(titles):
            axes[row, col].set_title(titles[i])
    
    # Remove any unused subplots
    for i in range(num_images, num_rows*num_cols):
        row = i // num_cols
        col = i % num_cols
        fig.delaxes(axes[row, col])
    
    # Adjust the layout and display the plot
    plt.tight_layout()
    plt.show()

processor = FourierImageAnalyzer(initial_scale_window=2, max_scale_window=100, alpha_factor=0.8)
# # processor.plot_original_and_reconstructed('/mnt/e/projects/gta.jpg')
# # processor.plot_swapped_frequency_images_interactive('/mnt/e/datasets/sugarbeet_syn_v6/images/2220.png', 
# #                                        '/mnt/e/datasets/phenobench/train/images/05-26_00158_P0034279.png')

processor.interactive_visualization('/mnt/e/datasets/sugarbeet_syn_v6/images/0009.png',  
                                    '/mnt/e/datasets/phenobench/train/images/05-26_00158_P0034279.png',
                                    # '/mnt/e/datasets/phenobench/train/images/05-15_00181_P0030949.png',
                                    # '/mnt/e/datasets/phenobench/train/images/06-05_00133_P0037987.png',
                                    channel=-1, color_space='YCbCr')
# processor.visualize_spectrum('/mnt/e/projects/hanover.png')
# processor.visualize_spectrum('/mnt/e/projects/gta.jpg')
# processor.visualize_spectrum('/mnt/e/datasets/sugarbeet_syn_v6/images/2220.png')
# processor.visualize_spectrum('/mnt/e/datasets/phenobench/train/images/05-26_00158_P0034279.png')
# processor.interactive_visualization('/mnt/e/projects/gta.jpg',  
#                                     '/mnt/e/projects/hanover.png', 
#                                     channel=-1, color_space='YCbCr')
im_src = Image.open("/mnt/e/datasets/sugarbeet_syn_v6/images/2220.png").convert('RGB')
im_trg = Image.open("/mnt/e/datasets/phenobench/train/images/05-26_00158_P0034279.png").convert('RGB')

# im_src = Image.open("/mnt/e/projects/gta.jpg").convert('RGB')
# im_trg = Image.open("/mnt/e/projects/hanover.png").convert('RGB')
# # visualize_fourier('/mnt/e/datasets/sugarbeet_syn_v6/images/2220.png')
# # visualize_fourier(im_trg)
# # visualize_fourier_rgb_combined('/mnt/e/datasets/sugarbeet_syn_v6/images/2220.png')
# # visualize_fourier_rgb('/mnt/e/datasets/phenobench/train/images/05-26_00158_P0034279.png')
# visualize_fourier_rgb('/mnt/e/datasets/sugarbeet_syn_v6/images/2220.png')
# im_src = im_src.resize( (1024,1024), Image.BICUBIC )
# im_trg = im_trg.resize( (1024,1024), Image.BICUBIC )

# im_src = np.asarray(im_src, np.float32)
# im_trg = np.asarray(im_trg, np.float32)

# im_src = im_src.transpose((2, 0, 1))
# im_trg = im_trg.transpose((2, 0, 1))

# src_in_trg, src_data, trg_data = FDA_source_to_target_np( im_src, im_trg, L=0.0001 )

# im_src = im_src.transpose((1, 2, 0)).astype(np.uint8)
# im_trg = im_trg.transpose((1, 2, 0)).astype(np.uint8)

# src_in_trg = src_in_trg.transpose((1,2,0)).astype(np.uint8)
# plot_images([im_src, im_trg, src_in_trg], ['source image','target image', 'da_image'])
# plot_images([src_data[0], src_data[1], trg_data[0], trg_data[1]], ['source phase', 'source magnitude', 'target phase', 'target magnitude'])
# cv_image = cv2.cvtColor(src_in_trg.astype(np.uint8), cv2.COLOR_RGB2BGR)
# cv2.imwrite('src_in_tar.png', cv_image)
# scipy.misc.toimage(src_in_trg, cmin=0.0, cmax=255.0).save('src_in_tar.png')