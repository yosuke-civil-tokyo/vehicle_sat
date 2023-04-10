import subprocess
import os
import argparse
# from osgeo import gdal, gdalconst, gdal_array
# import shutil

from utils import *

parser = argparse.ArgumentParser()
parser.add_argument('-data', type=str, default='data')
parser.add_argument('-ulx', type=float, default=139.775)
parser.add_argument('-uly', type=float, default=35.683333)
parser.add_argument('-lrx', type=float, default=139.8375)
parser.add_argument('-lry', type=float, default=35.641667)
parser.add_argument('-place', type=str, default='toyosu')
parser.add_argument('-disaster', type=int, default=0)
parser.add_argument('-mul2', type=str, default='False')
args = parser.parse_args()

region = [args.ulx, args.uly, args.lrx, args.lry]
place = args.place
disaster = args.disaster
if args.mul2=='True':
    mul2 = True
else:
    mul2 = False

data_path = os.path.join(args.data, '{}/satellite/{}_{}/'.format(place, place, disaster))
raw_path = data_path + 'rawimg/'
make_path = data_path + 'preprocessed/panmul/'
make_path2 = data_path + 'preprocessed/panmul2/'
folder_list = os.listdir(data_path)

os.makedirs(make_path, exist_ok=True
os.makedirs(make_path2, exist_ok=True)


# radiometric correction
# also convert 11-bit, 9-bit, 8-bit
def radio_cor(input_path, output_path, scale_list, offset):
    # use if radiometric correction is needed

    scale_str = ''
    #for i in range(len(scale_list)):
    #    scale_str += ' -scale_{} 0 2046 {} {}'.format(i+1, int(offset[i]), int((scale_list[i]*2046)+offset[i])//2)
    #subprocess.call('gdal_translate -ot Byte{} {} {}'.format(scale_str, input_path, output_path).split())

    subprocess.call('gdal_translate -ot Byte -scale 0 511 0 255 {} {}'.format(input_path, output_path).split())


# convert coordinate (do interpolation if needed)
def convert_cord(input_path, output_path, cord, w, h, delete=False):
    source_projection = 'EPSG:32654'
    ulx, uly, lrx, lry = cord
    subprocess.call('gdalwarp -s_srs {} -t_srs EPSG:4326 -te {} {} {} {} -ts {} {} -r cubicspline {} {}'.format(
        source_projection, ulx, lry, lrx, uly, w, h, input_path, output_path).split())

    if delete:
        os.remove(input_path)


# clip the image with given range & output with given size
def clip(input_path, output_path, cord, w, h, delete=False):
    ulx, uly, lrx, lry = cord
    subprocess.call('gdal_translate -projwin {} {} {} {} -outsize {} {} {} {}'.format(
        ulx, uly, lrx, lry, w, h, input_path, output_path).split())

    if delete:
        os.remove(input_path)


# get the configs of satellite imagery (assume WV-2/WV-3)
def get_scale(imd_path, pan_or_mul, mul2):
    fac_list = []
    width_list = []
    with open(imd_path) as f:
        l = f.readlines()

    for line in l:
        if line.startswith('\tabsCalFactor'):
            fac_list.append(float(line[line.find('=')+2:-2]))
        elif line.startswith('\teffectiveBandwidth'):
            width_list.append(float(line[line.find('=') + 2:-2]))

    if pan_or_mul=='PAN':
        gain_list = [0.923]
        offset = [-1.7]
        scale_list = [0.923 * fac_list[0] / width_list[0]]
    elif (pan_or_mul=='MUL')&(not mul2):
        gain_list = [0.905, 0.907, 0.945, 0.982]
        offset = [-4.189, -3.287, -1.350, -3.752]
        scale_list = [gain_list[i]*fac_list[i]/width_list[i] for i in range(len(gain_list))]
    elif mul2:
        gain_list = [0.863, 0.905, 0.907, 0.938, 0.945, 0.980, 0.982, 0.954]
        offset = [-7.154, -4.189, -3.287, -1.816, -1.350, -2.617, -3.752, -1.507]
        scale_list = [gain_list[i] * fac_list[i] / width_list[i] for i in range(len(gain_list))]
    else:
        print('specify PAN or MUL')
        return None

    return scale_list, offset


# get raw satellite imagery & output necessary range, size, band of pre-processed imagery
def get_tif(root, pan_or_mul, copy_to_path, cord, w, h, pan_img_path=None, mul2=False, radio=True):
    w_pix = int(w / 0.3)
    h_pix = int(h / 0.3)

    data_dir = [os.path.join(root, fol) for fol in os.listdir(root) if fol.endswith(pan_or_mul)][0]
    img_path = [os.path.join(data_dir, img) for img in os.listdir(data_dir) if img.upper().endswith('.TIF')][0]
    imd_path = [os.path.join(data_dir, img) for img in os.listdir(data_dir) if img.upper().endswith('.IMD')][0]
    output_path1 = os.path.join(copy_to_path, place+'_'+pan_or_mul+'_meter.tif')
    output_path2 = os.path.join(copy_to_path, place + '_' + pan_or_mul + '_noclip.tif')
    output_path3 = os.path.join(copy_to_path, place + '_' + pan_or_mul + '.tif')
    mul_sharpen_path = os.path.join(copy_to_path, place + '_MULPAN_noclip' + '.tif')
    mul_sharpen_path_clip = os.path.join(copy_to_path, place + '_MULPAN' + '.tif')
    nir_path = os.path.join(copy_to_path, place + '_NIR_noclip' + '.tif')
    nir_path_clip = os.path.join(copy_to_path, place + '_NIR' + '.tif')
    nir_sharpen_path = os.path.join(copy_to_path, place + '_NIRPAN_noclip' + '.tif')
    nir_sharpen_path_clip = os.path.join(copy_to_path, place + '_NIRPAN' + '.tif')
    scale_list, offset = get_scale(imd_path, pan_or_mul, mul2)

    if radio:
        radio_cor(img_path, output_path1, scale_list, offset)
    else:
        output_path1 = img_path

    if pan_or_mul=='PAN':
        # subprocess.call('gdalwarp -t_srs EPSG:4326 -r cubicspline {} {}'.format(
        #     output_path1, output_path2).split())
        convert_cord(output_path1, output_path3, cord, w_pix, h_pix)
        # clip(output_path2, output_path3, cord, w_pix, h_pix, delete=False)

    elif (pan_or_mul=='MUL')&(not mul2):
        """# pansharpening mul1
        subprocess.call(
            'gdal_pansharpen.py -b 3 -b 2 -b 1 {} {} {}'.format(pan_img_path, output_path1, mul_sharpen_path).split())
        convert_cord(mul_sharpen_path, mul_sharpen_path_clip, cord, w_pix, h_pix, delete=False)"""
        subprocess.call(
            'gdal_pansharpen.py {} {} {}'.format(pan_img_path, output_path1, output_path3).split())
        convert_cord(output_path3, mul_sharpen_path, cord, w_pix, h_pix, delete=False)

        subprocess.call(
            'gdal_translate -b 3 -b 2 -b 1 {} {}'.format(mul_sharpen_path, mul_sharpen_path_clip).split())

        """subprocess.call(
            'gdal_pansharpen.py -b 4 {} {} {}'.format(pan_img_path, output_path1, nir_sharpen_path).split())"""
        #convert_cord(output_path3, nir_sharpen_path, cord, w_pix, h_pix, delete=True)
        subprocess.call(
            'gdal_translate -b 4 {} {}'.format(mul_sharpen_path, nir_sharpen_path_clip).split())

        subprocess.call(
            'gdal_translate -b 4 {} {}'.format(output_path1, nir_path).split())
        convert_cord(nir_path, nir_path_clip, cord, w_pix, h_pix, delete=True)

        # resize mul to pan size
        # clip(mul_sharpen_path, mul_sharpen_path_clip, cord, w_pix, h_pix, delete=True)
        # clip(nir_sharpen_path, nir_sharpen_path_clip, cord, w_pix, h_pix, delete=True)
        # clip(nir_path, nir_path_clip, cord, w_pix, h_pix, delete=True)

    elif mul2:
        subprocess.call(
            'gdal_pansharpen.py {} {} {}'.format(pan_img_path, output_path1, output_path3).split())
        convert_cord(output_path3, mul_sharpen_path, cord, w_pix, h_pix, delete=False)

        subprocess.call(
            'gdal_translate -b 5 -b 3 -b 2 {} {}'.format(mul_sharpen_path, mul_sharpen_path_clip).split())

        #convert_cord(output_path3, nir_sharpen_path, cord, w_pix, h_pix, delete=True)
        subprocess.call(
            'gdal_translate -b 7 {} {}'.format(mul_sharpen_path, nir_sharpen_path_clip).split())

        subprocess.call(
            'gdal_translate -b 7 {} {}'.format(output_path1, nir_path).split())
        convert_cord(nir_path, nir_path_clip, cord, w_pix, h_pix, delete=True)

        # resize mul to pan size
        # clip(mul_sharpen_path, mul_sharpen_path_clip, cord, w_pix, h_pix, delete=True)
        # clip(nir_sharpen_path, nir_sharpen_path_clip, cord, w_pix, h_pix, delete=True)
        # clip(nir_path, nir_path_clip, cord, w_pix, h_pix, delete=True)

        # process mul2
        nir_path = os.path.join(copy_to_path, place + '_NIR2_noclip' + '.tif')
        nir_path_clip = os.path.join(copy_to_path, place + '_NIR2' + '.tif')
        nir_sharpen_path = os.path.join(copy_to_path, place + '_NIR2PAN_noclip' + '.tif')
        nir_sharpen_path_clip = os.path.join(copy_to_path, place + '_NIR2PAN' + '.tif')

        # pansharpening mul2
        subprocess.call(
            'gdal_translate -b 8 {} {}'.format(mul_sharpen_path, nir_sharpen_path_clip).split())
        subprocess.call(
            'gdal_translate -b 8 {} {}'.format(output_path1, nir_path).split())
        convert_cord(nir_path, nir_path_clip, cord, w_pix, h_pix, delete=True)
        # resize mul2 to pan size
        # clip(nir_sharpen_path, nir_sharpen_path_clip, cord, w_pix, h_pix, delete=True)
        # clip(nir_path, nir_path_clip, cord, w_pix, h_pix, delete=True)


    # shutil.copy2(img_path, os.path.join(copy_to_path, place+'_'+pan_or_mul+'.tif'))

    return output_path1


if __name__ == '__main__':
    meshlist = obtain_mesh(region)
    df_mesh = pd.DataFrame({'mesh': meshlist})
    df_mesh['ulx'], df_mesh['uly'] = ju.to_meshpoint(df_mesh.mesh, 1, 0)[::-1]
    df_mesh['lrx'], df_mesh['lry'] = ju.to_meshpoint(df_mesh.mesh, 0, 1)[::-1]
    ulx = min(df_mesh['ulx'])
    uly = max(df_mesh['uly'])
    lrx = max(df_mesh['lrx'])
    lry = min(df_mesh['lry'])
    cord = (ulx, uly, lrx, lry)
    
    w_meter = cal_meter_from_latlon([ulx,(uly+lry)/2], [lrx,(uly+lry)/2])
    h_meter = cal_meter_from_latlon([(ulx+lrx)/2,uly], [(ulx+lrx)/2,lry])
    
    pan_img_path = get_tif(raw_path, 'PAN', make_path, cord, w=w_meter, h=h_meter)
    mul_img_path = get_tif(raw_path, 'MUL', make_path, cord, w=w_meter, h=h_meter, pan_img_path=pan_img_path, mul2=mul2)
    
    os.remove(pan_img_path)
    os.remove(mul_img_path)
    # os.remove(pan_img_path2)
    # os.remove(mul_img_path2)
    
    #pan_img_path = get_tif(raw_path, 'PAN', make_path2, cord, w=w_meter, h=h_meter, radio=False)
    #mul_img_path = get_tif(raw_path, 'MUL', make_path2, cord, w=w_meter, h=h_meter, pan_img_path=pan_img_path, mul2=mul2, radio=False)
    
    #os.remove(pan_img_path)
    #os.remove(mul_img_path)
    