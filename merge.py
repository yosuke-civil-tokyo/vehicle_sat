import argparse
import subprocess
from utils import *

parser = argparse.ArgumentParser()
parser.add_argument('-ulx', type=float, default=139.775)
parser.add_argument('-uly', type=float, default=35.683333)
parser.add_argument('-lrx', type=float, default=139.8375)
parser.add_argument('-lry', type=float, default=35.641667)
parser.add_argument('-place', type=str, default='toyosu')
parser.add_argument('-disaster', type=int, default=0)
args = parser.parse_args()

region = [args.ulx, args.uly, args.lrx, args.lry]
place = args.place
disaster = args.disaster

data_dir = '/data/{}/satellite/{}_{}/'.format(place, place, disaster)
data_path = data_dir + 'preprocessed/'
clip_dir = data_path + '/panmul/clipped/'

def main():
    meshlist = obtain_mesh(region)
    txt = [clip_dir + str(code) + '/det.tif' for code in meshlist]
    print(txt)
    cmd = 'gdal_merge.py -o output_all.tif'
    for img in txt:
        cmd = cmd + ' ' + img
    subprocess.call(cmd.split())

if __name__ == '__main__':
    main()