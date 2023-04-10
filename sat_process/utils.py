import os
import numpy as np
import pandas as pd
import heapq
import cv2
import jismesh.utils as ju
import json
import math
import datetime

# 隣のメッシュを取得する, hw=1:東隣, hw=-1:南隣
def neibour_mesh(mesh_code, hw):
    y, x = ju.to_meshpoint(mesh_code, (hw*0.5), 1+(hw*0.5))
    return ju.to_meshcode(y, x, 4)


# 指定範囲内のメッシュidのリストを作成
def obtain_mesh(cord):
    ulx = cord[0] + 0.0001
    uly = cord[1] - 0.0001
    lrx = cord[2] - 0.0001
    lry = cord[3] + 0.0001
    ulm = ju.to_meshcode(uly, ulx, 4)
    llm = ju.to_meshcode(lry, ulx, 4)
    urm = ju.to_meshcode(uly, lrx, 4)
    lrm = ju.to_meshcode(lry, lrx, 4)

    h, w = (0, 0)
    tmp_m = ulm
    while(tmp_m!=llm):
        tmp_m = neibour_mesh(tmp_m, -1)
        h += 1

    tmp_m = ulm
    while (tmp_m!=urm):
        tmp_m = neibour_mesh(tmp_m, 1)
        w += 1

    mesh_list = [ulm]
    parent_m = ulm
    tmp_m = parent_m
    for j in range(w):
        tmp_m = neibour_mesh(tmp_m, 1)
        mesh_list.append(tmp_m)
    for i in range(h):
        parent_m = neibour_mesh(parent_m, -1)
        tmp_m = parent_m
        mesh_list.append(tmp_m)
        for j in range(w):
            tmp_m = neibour_mesh(tmp_m, 1)
            mesh_list.append(tmp_m)

    return mesh_list


# 2地点間(緯度軽度)の距離を計算
def cal_meter_from_latlon(start, end):
    pole_radius = 6356752.314245  # 極半径
    equator_radius = 6378137.0  # 赤道半径

    # 緯度経度をラジアンに変換
    lat_start = math.radians(start[1])
    lon_start = math.radians(start[0])
    lat_end = math.radians(end[1])
    lon_end = math.radians(end[0])

    lat_difference = lat_start - lat_end       # 緯度差
    lon_difference = lon_start - lon_end       # 経度差
    lat_average = (lat_start + lat_end) / 2    # 平均緯度

    e2 = (math.pow(equator_radius, 2) - math.pow(pole_radius, 2)) \
            / math.pow(equator_radius, 2)  # 第一離心率^2

    w = math.sqrt(1- e2 * math.pow(math.sin(lat_average), 2))

    m = equator_radius * (1 - e2) / math.pow(w, 3) # 子午線曲率半径

    n = equator_radius / w                         # 卯酉線曲半径

    distance = math.sqrt(math.pow(m * lat_difference, 2) \
                   + math.pow(n * lon_difference * math.cos(lat_average), 2)) # 距離計測

    return distance


# geojsonを作成 -> アプリやGISで可視化
def create_geojson(csv_table, geotype):
    table = {"type" : "FeatureCollection",
             "crs" : {"type": "name",
                     "properties": {
                         "name": "urn:ogc:def:crs:OGC:1.3:CRS84"
                     }
                     },
             "features" : [
             ]
             }

    if geotype=="LineString":
        cord_name_set = {"s_lon", "s_lat", "e_lon", "e_lat"}
        for i, row in csv_table.iterrows():
            feat_dict = {"type":"Feature",
                         "properties":{},
                         "geometry":{
                             "type":geotype,
                             "coordinates":[[row["s_lon"], row["s_lat"]], [row["e_lon"], row["e_lat"]]]
                         }}
            for index in row.index:
                if index not in cord_name_set:
                    feat_dict["properties"][index] = row[index]
            table["features"].append(feat_dict)

    elif geotype=="Polygon":
        cord_name_set = {"ulx", "uly", "lrx", "lry"}
        for i, row in csv_table.iterrows():
            feat_dict = {"type":"Feature",
                         "properties":{},
                         "geometry":{
                             "type":geotype,
                             "coordinates":[]
                         }}

            feat_dict["geometry"]["coordinates"].append([row["ulx"], row["uly"]])
            feat_dict["geometry"]["coordinates"].append([row["ulx"], row["lry"]])
            feat_dict["geometry"]["coordinates"].append([row["lrx"], row["lry"]])
            feat_dict["geometry"]["coordinates"].append([row["lrx"], row["uly"]])
            feat_dict["geometry"]["coordinates"].append([row["ulx"], row["uly"]])

            for index in row.index:
                if index not in cord_name_set:
                    feat_dict["properties"][index] = row[index]
            table["features"].append(feat_dict)

    return table


# edgeデータ(linestring)のgeojson化 追加関数
def edge_to_geojson(node_dir, edge_dir, output_dir):
    node_data = pd.read_csv(node_dir)
    edge_data = pd.read_csv(edge_dir)

    s_dict = dict(zip(edge_data['edge_id'], edge_data['from']))
    e_dict = dict(zip(edge_data['edge_id'], edge_data['to']))
    lon_dict = dict(zip(node_data['node_id'], node_data['lon']))
    lat_dict = dict(zip(node_data['node_id'], node_data['lat']))

    s_lon = edge_data['edge_id'].replace(s_dict).replace(lon_dict).values
    s_lat = edge_data['edge_id'].replace(s_dict).replace(lat_dict).values
    e_lon = edge_data['edge_id'].replace(e_dict).replace(lon_dict).values
    e_lat = edge_data['edge_id'].replace(e_dict).replace(lat_dict).values

    edge_data['s_lon'] = s_lon; edge_data['s_lat'] = s_lat
    edge_data['e_lon'] = e_lon; edge_data['e_lat'] = e_lat

    json_data = create_geojson(edge_data, "LineString")
    with open(output_dir, 'w') as f:
        json.dump(json_data, f, indent=4)

    return None


# 車両データ(polygon)のgeojson化 追加関数
def vehicle_to_geojson(vehicle_dir, output_dir):
    vehicle_data = pd.read_csv(vehicle_dir)
    json_data = create_geojson(vehicle_data, "Polygon")

    with open(output_dir, 'w') as f:
        json.dump(json_data, f, indent=4)

    return None


# 衛星データの取得日時を確認(WV-2/WV-3)
# ただし、configファイルの日時には、バンド間の撮影ラグは反映されていない模様
def get_time(imd_path):
    fac_list = []
    width_list = []
    with open(imd_path) as f:
        l = f.readlines()

    for line in l:
        if line.startswith('\tearliestAcqTime'):
            time_data = line[line.find('=')+2:-2]
    time_data = datetime.datetime.fromisoformat(time_data[:-9])

    return time_data