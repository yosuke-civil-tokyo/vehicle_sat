import os
import subprocess
import cv2
import time
import argparse

from numbers_vehicle import *
from utils import *

def parse_args():
    """Parse command line arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-data', type=str, default='satellite')
    parser.add_argument('-ulx', type=float, default=139.775)
    parser.add_argument('-uly', type=float, default=35.683333)
    parser.add_argument('-lrx', type=float, default=139.8375)
    parser.add_argument('-lry', type=float, default=35.641667)
    parser.add_argument('-place', type=str, default='toyosu')
    parser.add_argument('-disaster', type=int, default=0)
    parser.add_argument('-mul2', type=str, default='False')
    args = parser.parse_args()

    return args

# function for end-to-end speed calculation
def add_speed_from_road(df_mesh, df_vehicle, road_unit_vec, data_path, mul2):
    clip_dir = os.path.join(data_path, '/panmul/clipped/')
    table = []
    for index, row in df_mesh.iterrows():
        code = int(row['mesh'])
        active_dir = clip_dir + str(code) + '/'
        ulx, uly, lrx, lry = row['ulx'], row['uly'], row['lrx'], row['lry']
        mesh_range = [ulx, uly, lrx, lry]

        vehicles = df_vehicle.loc[df_vehicle['mesh']==code]
        road_unit_vecs = road_unit_vec[df_vehicle['mesh']==code]

        # read
        img_raw_mesh = Image.open(active_dir + 'Pansharpen.tif')
        src_img = Image.open(active_dir+'src.tif')
        # use NIR as target image, when calculate speed of vehicles
        dst_img = Image.open(active_dir+'dst.tif')
        width, height = img_raw_mesh.size
        if mul2:
            dst_img2 = Image.open(active_dir+'dst2.tif')
            for vehicle, road_vec in zip(vehicles.values, road_unit_vecs):
                mesh = vehicle[0]
                x1, y1 = cord_to_pix([vehicle[1],vehicle[2]], mesh_range, height, width)
                x2, y2 = cord_to_pix([vehicle[3],vehicle[4]], mesh_range, height, width)

                speed, azimuth, reliability = pred_speed(src_img, dst_img, x1, y1, x2, y2, road_vec, window_size=31, dst_img2=dst_img2)
                table.append([mesh, vehicle[1], vehicle[2], vehicle[3], vehicle[4], 1, speed, azimuth, reliability, vehicle[-3], vehicle[-2], vehicle[-1]])

        else:
            for vehicle, road_vec in zip(vehicles.values, road_unit_vecs):
                mesh = vehicle[0]
                x1, y1 = cord_to_pix([vehicle[1],vehicle[2]], mesh_range, height, width)
                x2, y2 = cord_to_pix([vehicle[3],vehicle[4]], mesh_range, height, width)

                speed, azimuth, reliability = pred_speed(src_img, dst_img, x1, y1, x2, y2, road_vec, window_size=31, dst_img2=None)
                table.append([mesh, vehicle[1], vehicle[2], vehicle[3], vehicle[4], 1, speed, azimuth, reliability, vehicle[-3], vehicle[-2], vehicle[-1]])

    df = pd.DataFrame(table,
                      columns=['mesh', 'ulx', 'uly', 'lrx', 'lry', 'bool_on_road', 'speed', 'azimuth', 'reliability', 'edge_id', 'lane', 'inter_dist'])

    return df


# end-to-end vehicle detection from RGB image
def detect_vehicle(df_mesh, data_dir, mask_dir, place):
    yolov3_path = '../vehicle/'
    data_path = data_dir + 'preprocessed/'
    mask_path = mask_dir
    config_path = yolov3_path + 'config/yolov3_custom.yaml'  # yolov3_custom.yaml のパスを指定してください
    weights_path = yolov3_path + 'train_output/yolov3_final.pth'  # 重みのパスを指定してください
    src_img_path = data_path + 'panmul/{}_PAN.tif'.format(place)
    dst_img_path = data_path + 'panmul/{}_NIR.tif'.format(place)
    raw_img_file = data_path + 'panmul/{}_MULPAN.tif'.format(place)
    mask_img_file = mask_path + 'mask_f.tif'
    mask_img_file_use = mask_path + 'mask_for_veh.tif'
    clip_dir = data_path + 'panmul/clipped/'

    # raw_img_file = data_path + 'panmul/Toyosu_Pan_clip_8bit.tif'

    if place == 'ibaraki':
        mul2 = True
        dst_img_path2 = os.path.join(data_path, 'panmul/{}_NIR2.tif'.format(place))
    else:
        mul2 = False

    try:
        img_raw = Image.open(raw_img_file)
    except:
        print('No satellite image is found')
        print(raw_img_file)

    w1, h1 = img_raw.size
    exif_r = img_raw.getexif()
    for k, v in exif_r.items():
        if k==34853:
            print(v)
    
    try:
        img_mask= Image.open(mask_img_file)
    except:
        print('No road mask image is found')
        print(img_mask)

    w2, h2 = img_mask.size
    img_mask = img_mask.crop(((w2-w1)//2, (h2-h1)//2, w1+(w2-w1)//2, h1+(h2-h1)//2))
    img_mask.save(mask_img_file_use, exif=exif_r)
    table = []
    detector = Detector(config_path, weights_path)
    for index, row in df_mesh.iterrows():
        code = int(row['mesh'])
        active_dir = clip_dir + str(code) + '/'
        os.makedirs(active_dir, exist_ok=True)
        ulx, uly, lrx, lry = row['ulx'], row['uly'], row['lrx'], row['lry']

        # clip images according to the mesh range
        subprocess.call('gdal_translate -projwin {} {} {} {} {} {}'.format(ulx, uly, lrx, lry, raw_img_file, active_dir+'Pansharpen.tif').split())
        subprocess.call('gdal_translate -projwin {} {} {} {} {} {}'.format(ulx, uly, lrx, lry, mask_img_file_use, active_dir+'mask.tif').split())
        subprocess.call('gdal_translate -projwin {} {} {} {} {} {}'.format(ulx, uly, lrx, lry, src_img_path, active_dir + 'src.tif').split())
        subprocess.call('gdal_translate -projwin {} {} {} {} {} {}'.format(ulx, uly, lrx, lry, dst_img_path, active_dir + 'dst.tif').split())
        if mul2:
            subprocess.call('gdal_translate -projwin {} {} {} {} {} {}'.format(ulx, uly, lrx, lry, dst_img_path2,
                                                                               active_dir + 'dst2.tif').split())

        img_raw_mesh = Image.open(active_dir+'Pansharpen.tif')
        exif_r = img_raw_mesh.getexif()
        img_mask_mesh = Image.open(active_dir+'mask.tif')
        src_img = Image.open(active_dir+'src.tif')
        # use NIR as target image, when calculate speed of vehicles
        dst_img = Image.open(active_dir+'dst.tif')
        position_img = Image.new(mode=img_raw_mesh.mode, size=(img_raw_mesh.width, img_raw_mesh.height), color=(0,0,0))

        #basically width&height nearly equal to 500[m] / 0.3[m/pix] = 1667[pix], so h_num&v_num=4
        width, height = img_raw_mesh.size
        h_num = width // 500 + 1
        v_num = height // 500 + 1
        detect_img_list = [[img_raw_mesh.crop((i*500, j*500, i*500+512, j*500+512)) for i in range(h_num)] for j in range(v_num)]

        detection = [[detector.detect_from_imgs(np.asarray(detect_img_list[j][i])) for i in range(h_num)] for j in range(v_num)]

        #plot vehicle points, draw rectangles
        mask_np = np.asarray(img_mask_mesh)
        draw_lect = ImageDraw.Draw(img_raw_mesh)
        draw_point = ImageDraw.Draw(position_img)

        # for each clipped image
        for i in range(h_num):
            for j in range(v_num):
                detected_cord = detection[j][i][0]
                # for each vehicle
                for box in detected_cord:
                    x1 = int(np.clip(box["x1"]+i*500, 0, img_raw_mesh.size[0] - 1))
                    y1 = int(np.clip(box["y1"]+j*500, 0, img_raw_mesh.size[1] - 1))
                    x2 = int(np.clip(box["x2"]+i*500, 0, img_raw_mesh.size[0] - 1))
                    y2 = int(np.clip(box["y2"]+j*500, 0, img_raw_mesh.size[1] - 1))
                    x1_c, y1_c, x2_c, y2_c = calc_veh_cord([ulx, uly, lrx, lry], [x1, y1, x2, y2], h=height, w=width)

                    bool_on_road = (mask_np[(y1 + y2) // 2, (x1 + x2) // 2, 0] >= 50)*1
                    if bool_on_road==1:
                        speed, azimuth, reliability, vec = get_speed(src_img, dst_img, x1, y1, x2, y2, window_size=15)
                        draw_lect.line((((x1 + x2) // 2, (y1 + y2) // 2),
                                       (((x1 + x2) // 2) - vec[1], ((y1 + y2) // 2) - vec[0])), fill=(255,0,0))
                    else:
                        speed = 0
                        azimuth = 0
                        reliability = 1
                        vec = [0,0]

                    draw_lect.rectangle((x1, y1, x2, y2), outline=(0, 250, 0), width=1)
                    draw_point.point(((x1+x2)//2, (y1+y2)//2), fill=(1, 0, 0))
                    #speed = 0
                    #azimuth = 0

                    #edge_belong = 0
                    #lane = 0

                    table.append([code, x1_c, y1_c, x2_c, y2_c, bool_on_road, speed, azimuth, reliability])

        img_raw_mesh.save(active_dir + 'det.tif', exif=exif_r)
        position_img.save(active_dir + 'pos.tif')

        position_np = np.asarray(position_img)
        masked_vehicles = position_np * mask_np
        print('num of total vehicles is {}'.format(np.sum(position_np[:,:,0]!=0)))
        print('num of vehicles on road is {}'.format(np.sum(masked_vehicles[:,:,0]!=0)))
        print(str(code), ' : end')
    df = pd.DataFrame(table, columns=['mesh', 'ulx', 'uly', 'lrx', 'lry', 'bool_on_road', 'speed', 'azimuth', 'reliability'])

    # output is a dataframe, contains vehicle information (each row corresponds to a vehicle)
    return df


def main():
    # setting
    args = parse_args()
    region = [args.ulx, args.uly, args.lrx, args.lry]
    place = args.place
    disaster = args.disaster
    data = args.data
    data_dir = os.path.join(data, '{}/satellite/{}_{}/'.format(place, place, disaster))
    output_read_dir = os.path.join(data, '{}/satellite/{}_0/'.format(place, place), 'output1/')
    output_dir = os.path.join(data, '{}/satellite/{}_{}/'.format(place, place, disaster), 'output1/')
    mask_dir = os.path.join(data, '{}/satellite/{}_0/'.format(place, place), 'mask1/')
    yolov3_path = '../vehicle/'
    data_path = os.path.join(data_dir, 'preprocessed/')

    if args.mul2=='True':
        dst_img_path2 = os.path.join(data_path, 'panmul/{}_NIR2.tif'.format(place))

    # set meshes for computation
    meshlist = obtain_mesh(region)
    df_mesh = pd.DataFrame({'mesh': meshlist})
    df_mesh['ulx'], df_mesh['uly'] = ju.to_meshpoint(df_mesh.mesh, 1, 0)[::-1]
    df_mesh['lrx'], df_mesh['lry'] = ju.to_meshpoint(df_mesh.mesh, 0, 1)[::-1]

    # detect vehicles on image
    start_time = time.time()
    df_vehicle = detect_vehicle(df_mesh=df_mesh, data_dir=data_dir, mask_dir=mask_dir, place=place)
    print('Time for Detection : {:.2f}s'.format(time.time()-start_time))
    df_vehicle.to_csv(output_dir + 'vehicle_all.csv', index=False)

    # split vehicles into 'on road' or 'not on road'
    df_vehicle_on_road = df_vehicle[df_vehicle['bool_on_road']==1]
    df_vehicle_out_road = df_vehicle[df_vehicle['bool_on_road']==0]
    print('Num of all vehicles : {}'.format(len(df_vehicle_on_road) + len(df_vehicle_out_road)))
    del df_vehicle

    # read road mask
    try:
        road_mask_img = cv2.imread(mask_dir + 'mask_qua.tif')
    except:
        print('No road mask image is found')
        print(mask_dir + 'mask_qua.tif')
    # prepare distributing vehicles onto each edge
    try:
        road_mask_table = pd.read_csv(mask_dir+'pix_to_edge.csv')
        edge_data = pd.read_csv(output_read_dir+'edge_for_sim.csv')
        node_data = pd.read_csv(output_read_dir+'node_for_sim.csv')
    except:
        print('No road network data is found')
        print('check {}'.format(output_dir)) 
    print('Num of road mask pix : ', len(road_mask_table))
    h, w, c = road_mask_img.shape
    ulx = min(df_mesh['ulx'])
    uly = max(df_mesh['uly'])
    lrx = max(df_mesh['lrx'])
    lry = min(df_mesh['lry'])
    img_cord = [ulx, uly, lrx, lry]
    df_vehicle['edge_id'] = 0
    df_vehicle['lane'] = 0

    # execute distribution
    start_time = time.time()
    df_vehicle_on_road = distribute_to_road(road_mask_table.values, df_vehicle_on_road, img_cord, h, w)
    df_vehicle_on_road, vec_road = calculate_inter_dist(df_vehicle_on_road, edge_data, node_data)
    print('Time for Distribute to Road : {:.2f}s'.format(time.time() - start_time))
    print('Num of vehicles on road : {}'.format(sum(df_vehicle_on_road['bool_on_road'].values!=0)))

    # aggregate the num of vehicles & average speed on edge level
    num_vehicle_each_link = df_vehicle_on_road.groupby('edge_id', as_index=False).count()[['edge_id', 'mesh']]
    speed_each_link = df_vehicle_on_road.groupby('edge_id', as_index=False).mean()[['edge_id', 'speed']]
    print('Num of all links : {}'.format(len(edge_data)))
    print('Num of links with vehicles : {}'.format(len(num_vehicle_each_link)))

    # predict speed using road information
    start_time = time.time()
    speed_with_edge_info = add_speed_from_road(df_mesh, df_vehicle_on_road, vec_road, data_path, args.mul2)
    print('Time for Speed Calculation : {:.2f}s'.format(time.time() - start_time))

    # merge the num of vehicles & average speed info on edge data
    edge_data = edge_data.merge(num_vehicle_each_link, on='edge_id', how='left').fillna(0).rename(columns={'mesh':'num_cars'})
    edge_data = edge_data.merge(speed_each_link, on='edge_id', how='left').fillna(0)
    # save
    edge_data.to_csv(output_dir+'edge_with_traffic.csv', index=False)
    df_vehicle_on_road.to_csv(output_dir+'vehicle_edgeinfo.csv', index=False)
    speed_with_edge_info.to_csv(output_dir+'vehicle_simspeed.csv', index=False)


if __name__ == "__main__":
    main()
    