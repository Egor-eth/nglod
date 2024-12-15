import cv2
img1 = cv2.imread('_results/render_app/imgs/n_bunny/rgb/mesh_cpu.png')
img2 = cv2.imread('_results/render_app/imgs/n_bunny/rgb/000000.png')

psnr = cv2.PSNR(img1, img2)
print("psnr=",psnr)
