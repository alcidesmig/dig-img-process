import numpy as np
import matplotlib.pyplot as plt

neighboors = [(i, j) for i in range(-1, 2) for j in range(-1, 2)]

def verify_1(img, point):
    values = [img[tuple(map(sum, zip(point, x)))] for x in neighboors]
    sum_values = sum(values) - values[4]
    condition_a = (2 <= sum_values <= 6)
    t_p1 = [4, 1, 2, 5, 8, 7, 6, 3, 0, 1]
    cont_p1 = 0
    for i in range(len(t_p1) - 1):
        if(values[t_p1[i]] == 0 and values[t_p1[i+1]] == 1):
            cont_p1 += 1
    condition_b = cont_p1 == 1
    condition_c = values[1] * values[5] * values[7] == 0
    condition_d = (values[5] * values[7] * values[3] == 0)
    return (condition_a and condition_b and condition_c and condition_d)

def verify_2(img, point):
    values = [img[tuple(map(sum, zip(point, x)))] for x in neighboors]
    sum_values = sum(values) - values[4]
    condition_a = (2 <= sum_values <= 6)
    t_p1 = [4, 1, 2, 5, 8, 7, 6, 3, 0, 1]
    cont_p1 = 0
    for i in range(len(t_p1) - 1):
        if(values[t_p1[i]] == 0 and values[t_p1[i+1]] == 1):
            cont_p1 += 1
    condition_b = cont_p1 == 1
    condition_c = values[1] * values[3] * values[5] == 0
    condition_d = (values[1] * values[7] * values[3] == 0)
    return (condition_a and condition_b and condition_c and condition_d)

def thinnerize(img):

	if len(img.shape) == 3:
		img = img[:, :, 0]
	if img.max() > 1:
		img /= 255
	nrows, ncols = img.shape
	
	img_bw = np.pad(img, pad_width=1, mode='constant', constant_values=0)
	
	removeds_1 = []
	removeds_2 = []
	
	while (len(removeds_1) + len(removeds_2) != 0):
	    
	    removeds_1 = []
	    removeds_2 = []
	    
	    for i in range(1, nrows):
	        for j in range(1, ncols):
	            if img_bw[i, j] != 0 and verify_neigh_1(img_bw, (i, j)):
	                removeds_1.append((i,j))
	    for i in removeds_1:
	        img_bw[i] = 0
	        
	    for i in range(1, nrows):
	        for j in range(1, ncols):
	            if img_bw[i, j] != 0 and verify_neigh_2(img_bw, (i, j)):
	                removeds_2.append((i,j))
	    for i in removeds_2:
	        img_bw[i] = 0
	
	plt.subplot(1, 2, 1)
	plt.imshow(img, 'gray')

	plt.subplot(1, 2, 2)
	plt.imshow(img_bw, 'gray')

def main():
	img = plt.imread('elefante.png')
	thinnerize(img)

if __name__ == '__main__':
	main()