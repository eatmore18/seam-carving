import cv2
import numpy
import argparse
import imageio


USE_FORWARD_ENERGY = True
MAX_IMAGE_WIDTH = 400
VERTICAL_MODE = True

def seam_carve(image_path,output_image_path,seam_num):
    image = cv2.imread(image_path)
    if VERTICAL_MODE == False:
        print('yes')
        image = numpy.rot90(image,1,(1,0))
    width = image.shape[1]
    if width > MAX_IMAGE_WIDTH:
        image = resize(image,MAX_IMAGE_WIDTH)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.astype(numpy.int16)
    output_image = image
    output_image = remove_seams(output_image,seam_num)
    if VERTICAL_MODE == False:
        output_image = numpy.rot90(output_image,-1,(1,0))
    cv2.imwrite(output_image_path,output_image)
    output_image = cv2.imread(output_image_path)
    output_image = cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB)
    cv2.imwrite(output_image_path,output_image)

def remove_seams(image,seam_num):
    seam_images = []
    for i in range(seam_num):
        boolmask = get_min_seam_mask(image)
        gif_image = heighlight_mask(image,boolmask)
        gif_image = numpy.rot90(gif_image,-1,(1,0))
        seam_images.append(gif_image)
        image = remove_seam(image,boolmask)
        print(f'seam number { i+1 } is removed')
    imageio.mimsave('./images/output.gif', seam_images)
    return image

def get_min_seam_mask(image):
    height = image.shape[0]
    width = image.shape[1]  
    energy_func = forward_energy if USE_FORWARD_ENERGY else get_energy
    energy = energy_func(image)
    parents_so_far = numpy.full((height,width),0)
    path_length_so_far = numpy.full((height,width),numpy.inf)
    for i in range(height):
        for j in range(width):
            if i==0:
                path_length_so_far[i][j] = energy[i][j]
                parents_so_far[i][j] = 0
            else:
                if j==0:
                    if energy[i][j] + path_length_so_far[i-1][j] < path_length_so_far[i][j]:
                        path_length_so_far[i][j] = energy[i][j] + path_length_so_far[i-1][j]
                        parents_so_far[i][j] = 0
                    if energy[i][j] + path_length_so_far[i-1][j+1] < path_length_so_far[i][j]:
                        path_length_so_far[i][j] = energy[i][j] + path_length_so_far[i-1][j+1]
                        parents_so_far[i][j] = 1 
                elif j == (width - 1):
                    if energy[i][j] + path_length_so_far[i-1][j-1] < path_length_so_far[i][j]:
                        path_length_so_far[i][j] = energy[i][j] + path_length_so_far[i-1][j-1]
                        parents_so_far[i][j] = -1
                    if energy[i][j] + path_length_so_far[i-1][j] < path_length_so_far[i][j]:
                        path_length_so_far[i][j] = energy[i][j] + path_length_so_far[i-1][j]
                        parents_so_far[i][j] = 0
                else:
                    if energy[i][j] + path_length_so_far[i-1][j-1] < path_length_so_far[i][j]:
                        path_length_so_far[i][j] = energy[i][j] + path_length_so_far[i-1][j-1]
                        parents_so_far[i][j] = -1
                    if energy[i][j] + path_length_so_far[i-1][j] < path_length_so_far[i][j]:
                        path_length_so_far[i][j] = energy[i][j] + path_length_so_far[i-1][j]
                        parents_so_far[i][j] = 0
                    if energy[i][j] + path_length_so_far[i-1][j+1] < path_length_so_far[i][j]:
                        path_length_so_far[i][j] = energy[i][j] + path_length_so_far[i-1][j+1]
                        parents_so_far[i][j] = 1
    min_last_line_index = numpy.argmin(path_length_so_far[height-1])
    mask_bool = numpy.ones((height, width), dtype=numpy.bool)  
    i = height - 1
    j = min_last_line_index
    while i>=0 :
        mask_bool[i][j] = False
        j += parents_so_far[i][j]
        i -= 1
    return mask_bool

def get_energy(image):
    image = numpy.pad(image,((1,1),(1,1),(0,0)),'edge')
    height = image.shape[0]
    width = image.shape[1]
    energy = numpy.zeros((height-2 , width-2))
    for i in range(1, height - 1):
        for j in range(1, width - 1):
            left_pixel = image[i, j-1]
            right_pixel = image[i, j+1]
            up_pixel = image[i-1, j]
            down_pixel = image[i+1, j]
            delta_x = ( down_pixel[0] - up_pixel[0] )**2 + ( down_pixel[1] - up_pixel[1] )**2 + ( up_pixel[2] - down_pixel[2] )**2
            delta_y = ( left_pixel[0] - right_pixel[0] )**2 + ( left_pixel[1] - right_pixel[1] )**2 + ( left_pixel[2] - right_pixel[2] )**2
            energy[i-1, j-1] = delta_x + delta_y
    return energy

def forward_energy(image):
    height = image.shape[0]
    width = image.shape[1]

    image = cv2.cvtColor(image.astype(numpy.uint8), cv2.COLOR_BGR2GRAY).astype(numpy.float64)

    energy = numpy.zeros((height, width))
    m = numpy.zeros((height, width))
    
    U = numpy.roll(image, 1, axis=0)
    L = numpy.roll(image, 1, axis=1)
    R = numpy.roll(image, -1, axis=1)
    
    cU = numpy.abs(R - L)
    cL = numpy.abs(U - L) + cU
    cR = numpy.abs(U - R) + cU
    
    for i in range(1, height):
        mU = m[i-1]
        mL = numpy.roll(mU, 1)
        mR = numpy.roll(mU, -1)
        
        mULR = numpy.array([mU, mL, mR])
        cULR = numpy.array([cU[i], cL[i], cR[i]])
        mULR += cULR

        argmins = numpy.argmin(mULR, axis=0)
        m[i] = numpy.choose(argmins, mULR)
        energy[i] = numpy.choose(argmins, cULR)
      
        
    return energy



def remove_seam(image,boolmask):
    height = image.shape[0]
    width = image.shape[1]    
    return image[boolmask].reshape((height, width - 1 , 3))



def resize(image, width):
    dim = None
    h, w = image.shape[:2]
    dim = (width, int(h * width / float(w)))
    return cv2.resize(image, dim)

def heighlight_mask(image,boolmask):
    height = boolmask.shape[0]
    width  = boolmask.shape[1]
    for i in range(height):
        for j in range(width):
            if boolmask[i][j]==False:
                image[i][j] = [255,255,255]
    return image



if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("-size", type=int, required = True)
    arg_parser.add_argument("-input", required = True)
    arg_parser.add_argument("-output",default='./images/output.jpg')
    arg_parser.add_argument("-energyfn",help='forward or origin',default='forward')
    arg_parser.add_argument("-direction",help='horizontal or vertical', default='vertical')
    args = vars(arg_parser.parse_args())
    input_image_path = args['input']
    output_image_path = args['output']
    seams_num = args['size']
    direction = args['direction']
    energy_func = args['energyfn']
    if energy_func =='origin':
        USE_FORWARD_ENERGY = False
    if direction == 'horizontal':
        VERTICAL_MODE = False
    seam_carve(input_image_path,output_image_path,seams_num)
    
    

