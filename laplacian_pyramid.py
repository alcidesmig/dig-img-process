import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve
from scipy.signal import correlate

# Decorator para imagens coloridas
def support_rgba(parameter_type):
    if type(parameter_type) == int:
        n_images = parameter_type
        if parameter_type <= 0:
            raise Exception(f'parameter_type isn\'t a "int > 0" nor a "list"')
    elif not parameter_type == list:
        raise Exception(f'parameter_type isn\'t a "int > 0" nor a "list"')

    def inner(f):
        def wrapper(*args, **kwargs):
            # Separa as imagens e os argumentos
            if type(parameter_type) == int:
                images = args[:n_images]
                args = args[n_images:]
            elif parameter_type == list:
                images = args[0]
                args = args[1:]

            # Tamanho do shape de uma imagem
            shape_len = len(images[0].shape)

            # Caso tenha mais de um canal
            if shape_len == 3:
                _, _, dim = images[0].shape
                
                # Separa os canais
                split_channel = lambda img : [img[:, :, i] for i in range(dim)]
                # Separa os canais
                images_channels = map(split_channel, images)
                # Aplica a função em cada canal
                if type(parameter_type) == int:
                    processed_channels = [f(*channels, *args, **kwargs) for channels in zip(*images_channels)]
                elif parameter_type == list:
                    processed_channels = [f(channels, *args, **kwargs) for channels in zip(*images_channels)]
                
                # Imagem final, com as dimenções do output
                w, h = processed_channels[0].shape
                final_img = np.zeros([w, h, dim], dtype=np.uint32)

                # Atribui os canais à imagem final
                for i in range(dim):
                    final_img[:, :, i] = processed_channels[i]
                
                return final_img

            # Caso tenha apenas um canal
            elif shape_len == 2:
                if type(parameter_type) == int:
                    processed_img = f(*images, *args, **kwargs)
                elif parameter_type == list:
                    processed_img = f(images, *args, **kwargs)

                return processed_img
            else:
                raise Exception(f'Object does not have the right amount of dimensions: {images[0].shape}')

        return wrapper
    return inner

@support_rgba(1)
def downsample(img):
    '''Gera uma nova imagem com metade do tamanho da imagem de entrada. A imagem de
       entrada é suavizada utilizando um filtro gaussiano e amostrada a cada 2 pixels'''
    # Filtro gaussiano
    
    filtro = np.array([
        [1,  4,  6,  4, 1],
        [4, 16, 24, 16, 4],
        [6, 24, 36, 24, 6],
        [4, 16, 24, 16, 4],
        [1,  4,  6,  4, 1]
    ])/256.
    
    img = img.astype(float)
    n_rows, n_cols = img.shape

    half_rows = (n_rows+1)//2
    half_cols = (n_cols+1)//2
    
    # Faz a convolução da imagem com o filtro gaussiano
    img_smooth = convolve(img, filtro, mode='same')
    # Define uma matriz de zeros para a nova imagem
    img_down = np.zeros([half_rows, half_cols])
    
    for row in range(0, half_rows):
        for col in range(half_cols):
            img_down[row, col] = img_smooth[2*row, 2*col]
            
    return img_down

@support_rgba(1)
def upsample_2x(img, filtro='vizinho'):
    '''Interpola imagem utilizando o filtro fornecido na variável filtro'''
    
    # Calculo dos filtros
    _w = np.zeros([7,7])
    _w[2:4,2:4] = 1
    _w2 = correlate(_w, _w, mode='same')
    _w2 = _w2/4
    _w3 = correlate(_w2, _w2, mode='same')
    _w3 = _w3/4
    _w_c = np.array([[-0.0625, 0, 0.5625, 1, 0.5625, 0, -0.0625]])
    _w_c2d = np.dot(_w_c.T, _w_c)

    _filtros = {
        'vizinho'   : _w,
        'cone'      : _w2,
        'corr_cone' : _w3,
        'bicubica'  : _w_c2d
    }

    if type(filtro) == str:
        filtro = _filtros[filtro]

    n_rows, n_cols = img.shape
    
    # Define uma matriz de zeros para a nova imagem
    img_up = np.zeros([2*n_rows-1, 2*n_cols-1])
    
    for row in range(n_rows-1):
        for col in range(n_cols-1):
            img_up[2*row, 2*col] = img[row, col]
    img_up[-1,::-2] = img[-1]
    img_up[::2,-1] = img[:,-1]

    img_interp = correlate(img_up, filtro, mode='same')
    
    return img_interp

@support_rgba(2)
def image_difference(img_a, img_b):
    wa, ha = img_a.shape
    wb, hb = img_b.shape
    w = min(wa, wb)
    h = min(ha, hb)
    return img_a[:w, :h] - img_b[:w, :h]

def laplacian_pyramid(img, pyramid_size, interp_filter='vizinho'):
    # Lista de imagens downsampled
    downsampled_images = [img]
    for i in range(pyramid_size):
        new_img = downsample(downsampled_images[-1])
        downsampled_images.append(new_img)

    # Lista de imagens upsampled
    upsampled_images = [upsample_2x(img, interp_filter) for img in downsampled_images[1:]]

    laplacian_images = []
    for dimg, uimg in zip(downsampled_images[:-1], upsampled_images):
        diff_img = image_difference(dimg, uimg)
        laplacian_images.append(diff_img)

    return laplacian_images

@support_rgba(list)
def compose_piramide(pyramid):
        rows, cols = pyramid[0].shape
        composite_image = np.zeros((rows+1, cols+cols//2+1), dtype=np.uint32)
        composite_image[:rows, :cols] = pyramid[0]
        
        i_row = 0
        for p in pyramid[1:]:
            n_rows, n_cols = p.shape
            composite_image[i_row:i_row + n_rows, cols:cols + n_cols] = p
            i_row += n_rows

        return composite_image

img = plt.imread('cameraman.tiff')
pyramid = laplacian_pyramid(img, 5)
plt.imshow(compose_piramide(pyramid))
plt.show()
