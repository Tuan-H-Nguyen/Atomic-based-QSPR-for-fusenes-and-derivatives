#%%
from PIL import Image

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

def side_merge_image2(file1, file2):
    #Merge two images into one, displayed side by side
    #:param file1: path to first image file
    #:param file2: path to second image file
    #:return: the merged Image object
 
    image1 = Image.open(file1)
    image2 = Image.open(file2)

    (width1, height1) = image1.size
    (width2, height2) = image2.size

    result_width = width1 + width2
    result_height = max(height1,height2)

    result = Image.new('RGB', (result_width, result_height),color = 'white')
    result.paste(im=image1, box=(0, 0))
    result.paste(im=image2, box=(width1, 0))
    return result

def merge_image3(file1, file2,file3):
	image1 = Image.open(file1)
	image2 = Image.open(file2)
	image3 = Image.open(file3)
	
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
	

def side_merge_any(files_list,open_img=True):

    if open_img:
        files_list = [Image.open(file) for file in files_list]

    widths = [width for width,_ in [file.size for file in files_list]]
    widths.insert(0,0)
    result_width = sum(widths)
    result_height = max([height for _,height in [file.size for file in files_list]])

    result = Image.new('RGB', (result_width, result_height),color = 'white')

    for f,file in enumerate(files_list):
        result.paste(im=file, box=(sum(widths[:f+1]), 0))

    return result
