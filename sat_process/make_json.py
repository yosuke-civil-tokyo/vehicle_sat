import pandas as pd
import numpy as np
import os
import argparse
import datetime

from utils import *

parser = argparse.ArgumentParser()
parser.add_argument('-data', type=str, default='data')
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
data_dir = os.path.join(args.data, '{}/satellite/{}_{}/'.format(place, place, disaster))
output_dir = os.path.join(args.data, '{}/satellite/{}_0/'.format(place, place), 'output/')
output_path = os.path.join(args.data, '{}/satellite/{}_{}/'.format(place, place, disaster), 'output/')
mask_dir = os.path.join(args.data, '{}/satellite/{}_0/'.format(place, place, disaster), 'mask/')
raw_dir = [os.path.join(data_dir+'rawimg/', fo) for fo in os.listdir(data_dir+'rawimg/') if fo.endswith('MUL')][0]
imd_path = [os.path.join(data_dir, img) for img in os.listdir(raw_dir) if img.upper().endswith('.IMD')][0]
json_dir = output_path + 'json/'

os.makedirs(json_dir, exist_ok=True)

def main():
    #time_data = get_time(imd_path)
    #with open(output_dir / "train.txt", "w") as f:
    #    f.write("time : {}".format(time_data))
    node_path = output_dir + 'node_for_sim.csv'
    edge_names = [os.path.splitext(files)[0] for files in os.listdir(output_dir) if files.startswith('edge')]
    edge_csvs = [os.path.join(output_dir, files+'.csv') for files in edge_names]
    edge_jsons = [os.path.join(json_dir, files+'.geojson') for files in edge_names]

    for edge, output in zip(edge_csvs, edge_jsons):
        edge_to_geojson(node_path, edge, output)


    vehicle_names = [os.path.splitext(files)[0] for files in os.listdir(output_dir) if files.startswith('vehicle')]
    vehicle_csvs = [os.path.join(output_dir, files+'.csv') for files in vehicle_names]
    vehicle_jsons = [os.path.join(json_dir, files+'.geojson') for files in vehicle_names]

    for vehicle, output in zip(vehicle_csvs, vehicle_jsons):
        vehicle_to_geojson(vehicle, output)

    return None


if __name__ == "__main__":
    main()