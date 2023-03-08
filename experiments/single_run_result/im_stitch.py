import sys, os
path = os.path.dirname(os.path.realpath(__file__))

#%%
from PIL import Image
import sys, os
path = os.path.dirname(os.path.realpath(__file__))

def merge_image2(file1, file2):
    #Merge two images into one, displayed side by side
    #:param file1: path to first image file
    #:param file2: path to second image file
    #:return: the merged Image object
 
    image1 = Image.open(file1)
    image2 = Image.open(file2)

    (width1, height1) = image1.size
    (width2, height2) = image2.size

    result_width = max(width1,width2)
    result_height = height1 + height2

    result = Image.new('RGB', (result_width, result_height),color = 'white')
    result.paste(im=image1, box=(0, 0))
    result.paste(im=image2, box=(0, height1))
    return result

def side_merge_image2(image1, image2,open_img=True,align = "top"):
    #Merge two images into one, displayed side by side
    #:param file1: path to first image file
    #:param file2: path to second image file
    #:return: the merged Image object
 
    if open_img == True:
        image1 = Image.open(image1)
        image2 = Image.open(image2)

    (width1, height1) = image1.size
    (width2, height2) = image2.size

    result_width = width1 + width2
    result_height = max(height1,height2)

    if align == "top":
        result = Image.new('RGB', (result_width, result_height),color = 'white')
        result.paste(im=image1, box=(0, 0))
        result.paste(im=image2, box=(width1, 0))
    elif align == "bottom":
        result = Image.new('RGB', (result_width, result_height),color = 'white')
        result.paste(im=image1, box=(0, result_height - height1))
        result.paste(im=image2, box=(width1, result_height - height2))
    return result

def merge_image3(image1, image2,image3,open_img = True):
    if open_img:
        image1 = Image.open(image1)
        image2 = Image.open(image2)
        image3 = Image.open(image3)
	
    (width1, height1) = image1.size
    (width2, height2) = image2.size
    (width3, height3) = image3.size

    result_width = max(width1,width2,width3)
    result_height = height1 + height2 + height3

    result = Image.new('RGB', (result_width, result_height),color = 'white')
    result.paste(im=image1, box=(0, 0))
    result.paste(im=image2, box=(0, height1))
    result.paste(im=image3, box=(0, height1 + height2))
    return result

merge_image3(
    path + "\\GPR_BG.jpeg",
    path + "\\GPR_EA.jpeg",
    path + "\\GPR_IP.jpeg", True).save(path + "\\GPR.jpeg")

merge_image3(
    path + "\\RR_BG.jpeg",
    path + "\\RR_EA.jpeg",
    path + "\\RR_IP.jpeg", True).save(path + "\\RR.jpeg")
