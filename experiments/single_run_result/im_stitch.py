import sys, os
path = os.path.dirname(os.path.realpath(__file__))

#%%
from PIL import Image
import sys, os
path = os.path.dirname(os.path.realpath(__file__)) + "\\plot"

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

merge_image3(
    path + "\\GPR_BG.jpeg",
    path + "\\GPR_EA.jpeg",
    path + "\\GPR_IP.jpeg", True).save(path + "\\GPR.jpeg")

merge_image3(
    path + "\\GPR_r2_BG.jpeg",
    path + "\\GPR_r2_EA.jpeg",
    path + "\\GPR_r2_IP.jpeg", True).save(path + "\\GPR_r2.jpeg")

merge_image3(
    path + "\\RR_BG.jpeg",
    path + "\\RR_EA.jpeg",
    path + "\\RR_IP.jpeg", True).save(path + "\\RR.jpeg")

merge_image3(
    path + "\\RR_r2_BG.jpeg",
    path + "\\RR_r2_EA.jpeg",
    path + "\\RR_r2_IP.jpeg", True).save(path + "\\RR_r2.jpeg")

side_merge_any(
    [path + "\\parity_plot_"+data_type + "_ecfp_gpr.jpeg" 
    for data_type in ["mixed", "pah", "subst"]],
    True
).save(path + "\\parity_plot_ecfp_gpr.jpeg")

side_merge_any(
    [path + "\\parity_plot_"+data_type + "_subtree_gpr.jpeg" 
    for data_type in ["mixed", "pah", "subst"]],
    True
).save(path + "\\parity_plot_subtree_gpr.jpeg")

side_merge_any(
    [path + "\\parity_plot_"+data_type + "_edge_gpr.jpeg" 
    for data_type in ["mixed", "pah", "subst"]],
    True
).save(path + "\\parity_plot_edge_gpr.jpeg")

side_merge_any(
    [path + "\\parity_plot_"+data_type + "_shortest_path_gpr_.jpeg" 
    for data_type in ["mixed", "pah", "subst"]],
    True
).save(path + "\\parity_plot_shortest_path_gpr_.jpeg")

side_merge_any(
    [path + "\\parity_plot_"+data_type + "_subtree_rr.jpeg" 
    for data_type in ["mixed", "pah", "subst"]],
    True
).save(path + "\\parity_plot_subtree_rr.jpeg")

side_merge_any(
    [path + "\\parity_plot_"+data_type + "_edge_rr.jpeg" 
    for data_type in ["mixed", "pah", "subst"]],
    True
).save(path + "\\parity_plot_edge_rr.jpeg")

side_merge_any(
    [path + "\\parity_plot_"+data_type + "_shortest_path_rr.jpeg" 
    for data_type in ["mixed", "pah", "subst"]],
    True
).save(path + "\\parity_plot_shortest_path_rr.jpeg")

side_merge_any(
    [path + "\\parity_plot_"+data_type + "_ecfp_rr.jpeg" 
    for data_type in ["mixed", "pah", "subst"]],
    True
).save(path + "\\parity_plot_ecfp_rr.jpeg")

