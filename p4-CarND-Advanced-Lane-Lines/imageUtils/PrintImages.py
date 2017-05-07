import matplotlib.pyplot as plt

def print_images_sideways(image1,image2):
	
	f, (ax1, ax2) = plt.subplots(1, 2, figsize=(50, 50))
	plt.figure(figsize=(5,5))
	f.tight_layout()
	ax1.imshow(image1)
	ax1.set_title('Original Image', fontsize=50)
	ax2.imshow(image2)
	ax2.set_title('Undistorted Image', fontsize=50)
	plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)