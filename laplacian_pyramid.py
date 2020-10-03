import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve, correlate

'''
Função para aplicar um filtro gaussiano e
diminuir o tamanho de uma imagem
- função disponibilizada pelo professor e editada
'''


def down_size(img):

    # Definindo filtro gaussiano
    gauss_filter = np.array([
        [1,  4, 6,  4,  1],
        [4, 16, 24, 16, 4],
        [6, 24, 36, 24, 6],
        [4, 16, 24, 16, 4],
        [1,  4,  6,  4, 1]
    ]) / 256

    img = img.astype(float)
    n_rows, n_cols = img.shape

    half_rows = (n_rows + 1) // 2
    half_cols = (n_cols + 1) // 2

    # Faz a convolução da imagem com o filtro gaussiano
    img_convolved = convolve(img, gauss_filter, mode='same')
    # Define uma matriz de zeros para a nova imagem
    new_img = np.zeros([half_rows, half_cols])

    # Preenche a nova imagem fazendo uma amostragem da imagem original a cada 2px
    for i in range(half_rows):
        for j in range(half_cols):
            new_img[i, j] = img_convolved[2*i, 2*j]

    return new_img


'''
Função para fazer a interpolação de uma imagem utilizando um determinado filtro
- função disponibilizada pelo professor e editada
'''


def upsample_2x(img, filtro):
    num_rows, num_cols = img.shape
    img_upsampled = np.zeros([2*num_rows-1, 2*num_cols-1])
    for row in range(num_rows-1):
        for col in range(num_cols-1):
            img_upsampled[2*row, 2*col] = img[row, col]
    img_upsampled[-1, ::-2] = img[-1]
    img_upsampled[::2, -1] = img[:, -1]

    sinal_interp = correlate(img_upsampled, filtro, mode='same')

    return sinal_interp


def gaussian_pyramid(img, pyramid_size):
    pyramid = [img]

    # Gera a pirâmide Gaussiana
    for i in range(pyramid_size-1):
        img = down_size(img)
        pyramid.append(img)

    return pyramid[::-1]


def laplacian_pyramid(img, pyramid_size, gauss_filter_order=0):
    # Gera a pirâmide Gaussiana que vai ser utilizada para montar a pirâmide Laplaciana
    gauss_pyr = gaussian_pyramid(img, pyramid_size + 1)

    # Define o filtro para interpolação,
    # no caso, foi escolhido um filtro cone para fazer uma interpolação linear (ordem 1)
    interpolation_filter = np.zeros([7, 7])
    interpolation_filter[2:4, 2:4] = 1
    for i in range(gauss_filter_order):
        interpolation_filter = correlate(
            interpolation_filter, interpolation_filter, mode='same')

    if gauss_filter_order:
        interpolation_filter /= 4
    pyramid = []

    # Gera a pirâmide Laplaciana a partir da pirâmida Gaussiana
    # Layer 0 da pir. Laplaciana = interpolacao da layer 0 da pir. Gaussiana - layer 1 da pir. Gaussiana
    for i in range(pyramid_size):
        interpolated_img = upsample_2x(gauss_pyr[i], interpolation_filter)
        # Alteração: remove última linha e última coluna da imagem da pirâmide Gaussiana para seguir padrão de tamanho
        laplacian_layer = gauss_pyr[i + 1][:-1, :-1] - interpolated_img
        pyramid.append(laplacian_layer)

    return pyramid


'''
Função para plotar todas as layers da pirâmide
'''


def show_pyramid(pyramid):
    fig = plt.figure()
    size = len(pyramid)
    j = 1
    for i in pyramid:
        aux = fig.add_subplot((size+1)/2, (size+1)/2, j)
        aux.imshow(i, 'gray')
        j += 1
    plt.show()


if __name__ == '__main__':
    img = plt.imread('cameraman.tiff')
    laplacian_pyr = laplacian_pyramid(img, 4, gauss_filter_order=0)
    show_pyramid(laplacian_pyr)
