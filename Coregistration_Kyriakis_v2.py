
import cv2
import numpy as np
from matplotlib import pyplot as plt
import os 
import sys
import warnings
warnings.filterwarnings("ignore")
#============INTERFACE==========#
Title ="# Co-registration Algorithm #" 

Head ="\n\t\t"+Title  
print("\n\t\t"+("#"*len(Title))+Head+"\n\t\t"+("#"*len(Title)))


# ========== IMAGES ========== #
head1 = "Directory of the Images:"
print("\n"+head1+"\n"+"="*len(head1))
Directory = input("\t--> ")
Img_List  = os.listdir(Directory)
print("\t--> {} images were loaded".format(len(Img_List)))



# ===========PARAMETERS============ # 
head2 = 'Choose Parameters:'
print('\n'+head2+'\n'+'='*len(head2)+'\n')

motion = input("\nMotion model (Default: Translationn):\n\t1. Translationn\n\t2. Euclidean\n(1/2) --> ")
thres   = input("\nThreshold  (Deafult: 1e-10): ")
itera  = input("Iterations (Deafult: 10 ): ")

#== DEFAULT
if thres == "":
	thres = 1e-10
if itera =="":
	itera = 10


thres = float(thres)
itera = int(itera)
# ================================= #

# ================================= #
# ================================= #

if motion == "2":
	motion_str = "Euclidean"
else:
	motion_str = "Translation"


#================================
print("""
Checking Parameters:
====================

	Images loaded = {} 
	---------------------------
	Motion     = {}
	Threshold  = {}
	Iterations = {}
	---------------------------

""".format(len(Img_List),motion_str,thres,itera))
#================================
proc = input("Proceed ? (y/n):")
if proc == "n" or proc =="N" or proc=="no":
	sys.exit()

head3 = "Running..."
print("\n"+head3+"\n"+"="*len(head3)+"\n")

# ========= LOADINGBAR ====== #
import time
import sys
toolbar_width = 2*len(Img_List)
if not os.path.exists("Aligned_Results"):
	os.makedirs("Aligned_Results")

# setup toolbar
sys.stdout.write("\rLoading: [%s]" % ("-" * toolbar_width))
sys.stdout.flush()
sys.stdout.write("\b" * (toolbar_width+1)) # return to start of line, after '['
# ============================ #


# ============= CODE ============ #
im1 =  cv2.imread(Directory + "\\"+Img_List[0])
im1_gra = cv2.cvtColor(im1,cv2.COLOR_BGR2GRAY)
image_zero = im1_gra


for i in range(1,len(Img_List)):
	im2 =  cv2.imread(Directory+"\\"+Img_List[i])
	im2_gray = cv2.cvtColor(im2,cv2.COLOR_BGR2GRAY)
	
	# Find size of image1
	sz = image_zero.shape

	# Define the motion model
	if motion == "2":
		warp_mode = cv2.MOTION_EUCLIDEAN
	else: 
		warp_mode = cv2.MOTION_TRANSLATION
	


	# Specify the number of iterations.
	number_of_iterations = itera
	 
	# Specify the threshold of the increment in the correlation coefficient between two iterations
	termination_eps = thres

	# ============================ Transaltion Align ==========================================#	
	# Define 2x3 or 3x3 matrices and initialize the matrix to identity
	warp_matrix = np.eye(2, 3, dtype=np.float32)
	 
	# Define termination criteria
	criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, number_of_iterations,  termination_eps)
	 
	# Run the ECC algorithm. The results are stored in warp_matrix.
	(cc, warp_matrix) = cv2.findTransformECC (image_zero,im2_gray,warp_matrix, warp_mode, criteria)
	
	# Use warpAffine for Translation, Euclidean and Affine
	im2_aligned = cv2.warpAffine(im2_gray, warp_matrix, (sz[1],sz[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP);
	# =========================================================================================


	sys.stdout.write('\u2588')
	sys.stdout.flush()


	# ============================== Affine Align ==========================================#
	warp_mode = cv2.MOTION_AFFINE
	# Define 2x3 or 3x3 matrices and initialize the matrix to identity
	warp_matrix = np.eye(2, 3, dtype=np.float32)

	# Define termination criteria
	criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, number_of_iterations,  termination_eps)
	 
	# Run the ECC algorithm. The results are stored in warp_matrix.
	(cc, warp_matrix) = cv2.findTransformECC (image_zero,im2_aligned,warp_matrix, warp_mode, criteria)

	# Use warpAffine for Translation, Euclidean and Affine
	im3_aligned = cv2.warpAffine(im2_aligned, warp_matrix, (sz[1],sz[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP);
	# =========================================================================================

	cv2.imwrite("Aligned_Results/"+"{}_Aligned".format(i)+".jpg", im3_aligned )
	diff = ((im3_aligned+1)/(image_zero+1))*100
	cv2.imwrite("Aligned_Results/"+"{}_Diff".format(i)+".jpg", np.concatenate((image_zero, im3_aligned,diff), axis=1) )
	
	sys.stdout.write('\u2588')
	sys.stdout.flush()
	image_zero = im3_aligned
	# ====== END LOOP =========

sys.stdout.write('\u2588\u2588')
sys.stdout.flush()
sys.stdout.write("\n")
print("\nFinished !")
