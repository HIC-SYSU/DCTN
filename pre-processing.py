import nibabel as nib
import os
import numpy as np
import SimpleITK as sitk
from matplotlib import pylab as plt


# ResampleSize
def ResampleSize(sitkImage, width, height,deepth):
    euler3d = sitk.Euler3DTransform()

    xsize, ysize, zsize = sitkImage.GetSize()
    xspacing, yspacing, zspacing = sitkImage.GetSpacing()

    new_spacing_x = xspacing / (width / float(xsize))
    new_spacing_y = yspacing / (height / float(ysize))
    new_spacing_z = zspacing/(deepth/float(zsize))

    origin = sitkImage.GetOrigin()
    direction = sitkImage.GetDirection()

    newsize = (int(xsize*xspacing/new_spacing_x),int(ysize*yspacing/new_spacing_y),int(zsize*zspacing/new_spacing_z))
    newspace = (new_spacing_x, new_spacing_y, new_spacing_z)
    sitkImage = sitk.Resample(sitkImage,newsize,euler3d,sitk.sitkNearestNeighbor,origin,newspace,direction)
    return sitkImage

#nifit ，resize as：(256, 256, 512)
def Resize (img_path,save_path, WIDTH,HEIGHT,DEEPTH):
    filelist = os.listdir(img_path)
    filelist.sort()
    num = 0
    for filename in filelist:
        sitkImage = sitk.ReadImage(img_path + filename)

        # WIDTH,HEIGHT,DEEPTH = 256,256,512
        newsitkImage = ResampleSize(sitkImage, width=WIDTH,height=HEIGHT ,deepth=DEEPTH)

        img_array = sitk.GetArrayFromImage(newsitkImage)     ## （H,W,D) ->（D，H，W）
        np.save(save_path + str(filename).split('.')[0] + '.npy', img_array)  # 保存为npy

        num += 1
        # print('num = ',num)
    # print('Resize over')
    return 0

# adjust window_width and window_level

def saved_preprocessed(savedImg, origin, direction, xyz_thickness, saved_name):
    newImg = sitk.GetImageFromArray(savedImg)
    newImg.SetOrigin(origin)
    newImg.SetDirection(direction)
    newImg.SetSpacing((xyz_thickness[0], xyz_thickness[1], xyz_thickness[2]))
    sitk.WriteImage(newImg, saved_name)

def window_transform(ct_array, windowWidth, windowCenter, normal=False):
    """
    return: trucated image according to window center and window width and normalized to [0,1]
    """
    minWindow = float(windowCenter) - 0.5 * float(windowWidth)
    newing = (ct_array - minWindow) / float(windowWidth)
    newing[newing < 0] = 0
    newing[newing > 1] = 1

    # [0,1]
    if not normal:
        newing = (newing * 255).astype('uint8')
    return newing

def CT_trans(ct_path,label_path,saved_path):
    name_list = os.listdir(ct_path)
    name_list.sort()
    num = 0
    for name in name_list:
        ct = sitk.ReadImage(os.path.join(ct_path, name))
        origin = ct.GetOrigin()
        direction = ct.GetDirection()
        xyz_thickness = ct.GetSpacing()
        ct_array = sitk.GetArrayFromImage(ct)
        # seg_array = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(label_path, name.replace('image', 'GT'))))
        seg_array = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(label_path, name.replace('a2', 'm'))))

        seg_bg = seg_array == 0
        seg_1 = seg_array == 1
        # seg_2 = seg_array == 2

        ct_bg = ct_array * seg_bg
        ct_1 = ct_array * seg_1
        # ct_2 = ct_array * seg_2

        aorta1_min = ct_1.min()
        aorta1_max = ct_1.max()
        # aorta2_min = ct_2.min()
        # aorta2_max = ct_2.max()

        # by aorta1
        aorta1_wide = aorta1_max - aorta1_min
        aorta1_center = (aorta1_max + aorta1_min) / 2
        print('level:',aorta1_center,';','windowwidth:',aorta1_wide)
        aorta1_wl = window_transform(ct_array, aorta1_wide, aorta1_center, normal=False)
        saved_name = os.path.join(saved_path, name)
        saved_preprocessed(aorta1_wl, origin, direction, xyz_thickness, saved_name)

        # num += 1
        # print('num =',num,'\n','--------------------------')
    # print('post-process over')
