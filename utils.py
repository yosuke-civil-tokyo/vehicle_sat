import os
import numpy as np
import pandas as pd
import heapq
import cv2
import jismesh.utils as ju


# 隣のメッシュを取得する, hw=1:東隣, hw=-1:南隣
def neibour_mesh(mesh_code, hw):
    y, x = ju.to_meshpoint(mesh_code, (hw*0.5), 1+(hw*0.5))
    return ju.to_meshcode(y, x, 4)


# 指定範囲内のメッシュリスト
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


def cord_to_pix(in_cord, img_cord, img_h, img_w):
    # in_cord = [x, y]
    # img_cord = [ulx, uly, lrx, lry]
    x_pix = img_w * ((in_cord[0] - img_cord[0]) / (img_cord[2] - img_cord[0]))
    y_pix = img_h * ((in_cord[1] - img_cord[1]) / (img_cord[3] - img_cord[1]))

    return int(x_pix), int(y_pix)


# 画像のピクセル座標を緯度経度に変換
def pix_to_cord(in_pix, img_cord, img_h, img_w):
    # in_pix = [x, y]
    # img_cord = [ulx, uly, lrx, lry]
    x_cord = img_cord[0] + (img_cord[2]-img_cord[0])*(in_pix[0]/img_w)
    y_cord = img_cord[1] + (img_cord[3]-img_cord[1])*(in_pix[1]/img_h)

    return x_cord, y_cord


# 内積を用いて、スタートからの距離
def calc_dist_from_start(road_vec, vehicle_vec):
    road_vec_unit = road_vec / np.linalg.norm(road_vec, axis=1, keepdims=True).clip(min=1e-10)
    dot_prod = road_vec_unit[:, 0]*vehicle_vec[:, 0] + road_vec_unit[:, 1]*vehicle_vec[:, 1]

    return dot_prod, road_vec_unit


# 道路に対して集計を行う関数
def distribute_to_road(mask_table, feat_table, img_cord, img_h, img_w):
    #mode = 'disaster' or 'vehicle'
    feat_table['lon'] = (feat_table['ulx'] + feat_table['lrx']) / 2
    feat_table['lat'] = (feat_table['uly'] + feat_table['lry']) / 2
    #x_pix, y_pix = feat_table[['lon', 'lat']].apply(cord_to_pix, img_cord=img_cord, img_h=img_h, img_w=img_w)

    for i, vehicle in feat_table.iterrows():
        x_cord, y_cord = (vehicle[-2], vehicle[-1])
        x_pix, y_pix = cord_to_pix([x_cord, y_cord], img_cord, img_h, img_w)
        pix = np.where((mask_table[:,0]==y_pix)&(mask_table[:,1]==x_pix))
        if len(pix[0])!=0:
            pix_data = mask_table[pix][0]
            road_id = pix_data[2]
            lane = pix_data[3]
            feat_table.loc[i, 'edge_id'] = road_id
            feat_table.loc[i, 'lane'] = lane
        else:
            feat_table.loc[i, 'bool_on_road'] = 0

    return feat_table[feat_table['bool_on_road']!=0]
