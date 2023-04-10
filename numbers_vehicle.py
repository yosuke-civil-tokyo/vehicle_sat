from yolov3.detector import Detector

import subprocess
import jismesh.utils as ju
from PIL import Image, ImageDraw
import pandas as pd
import numpy as np
import cv2
import os
from utils import *

Image.MAX_IMAGE_PIXELS = 400000000

# calculate pixel coordinate in original image from clipped coordinate
def cord_in_ori_img(x1, y1, x2, y2, h, w):
    return x1 + 500*h, y1 + 500*w, x2 + 500*h, y2 + 500*w


# convert pixel location to lat/lon rectangle
def calc_veh_cord(img_cord, veh_pix, h, w):
    # img_cord = [ulx, uly, lrx, lry]
    veh_ulx = img_cord[0] + (img_cord[2] - img_cord[0]) * (veh_pix[0] / w)
    veh_uly = img_cord[1] + (img_cord[3] - img_cord[1]) * (veh_pix[1] / h)
    veh_lrx = img_cord[0] + (img_cord[2] - img_cord[0]) * (veh_pix[2] / w)
    veh_lry = img_cord[1] + (img_cord[3] - img_cord[1]) * (veh_pix[3] / h)

    return veh_ulx, veh_uly, veh_lrx, veh_lry


def corr_of_img(src, dst):
    #print(src)
    #print(dst)
    mu_src = np.nanmean(src)
    mu_dst = np.nanmean(dst)

    src_ = src - mu_src
    dst_ = dst - mu_dst

    nume = np.nanmean(src_*dst_)
    deno = np.sqrt(np.nanmean(src_**2)) * np.sqrt(np.nanmean(dst_**2))

    if deno==0:
        return 0
    else:
        return nume / deno


def cross_cor(src, dst, h, w, window_size):
    table = np.zeros((window_size, window_size))
    for i in range(window_size):
        for j in range(window_size):
            dst_match = dst[i:i+h, j:j+w]
            table[i, j] = corr_of_img(src, dst_match)

    return table


# simulationによる速度ベクトルの補正項を計算する関数
# 公開repoでは「載っているリンクの方向ベクトルとの内積」をスコアに
def speed_sim(traffic_row, window_size):
    vec_x = traffic_row[0]
    vec_y = traffic_row[1]
    half_win = (window_size - 1) // 2
    dot_matrix = np.array([[vec_x*i+vec_y*j for j in range(-half_win, half_win+1)] for i in range(-half_win, half_win+1)])
    len_matrix = np.array([[i**2+j**2 for j in range(-half_win, half_win+1)] for i in range(-half_win, half_win+1)])

    dot_matrix = np.abs(np.where(dot_matrix==0, 1, dot_matrix))
    len_matrix = np.where(len_matrix==0, 1, len_matrix)

    return np.clip(dot_matrix / np.sqrt(len_matrix), a_min=0.1, a_max=1)


# 空間相関とsimulationに基づいて、速度計算を行う関数
def pred_speed(src_img, dst_img, ulx, uly, lrx, lry, traffic_row, window_size=31, pix_size=0.3, time_dif=0.2, dst_img2=None):
    half_win = (window_size-1) // 2
    w = lrx - ulx
    h = lry - uly

    src_box = np.asarray(src_img.crop((ulx, uly, lrx, lry)))
    dst_box = np.asarray(dst_img.crop((ulx-half_win, uly-half_win, lrx+half_win, lry+half_win)))
    cor_mat = cross_cor(src_box, dst_box, h, w, window_size)
    score_matrix = np.flip(cor_mat)

    # if use 8-band multispectrum　& panchromatic
    if dst_img2:
        dst_box2 = np.asarray(dst_img2.crop((ulx-half_win, uly-half_win, lrx+half_win, lry+half_win)))
        cor_mat2 = cross_cor(src_box, dst_box2, h, w, window_size)
        score_matrix = (np.flip(cor_mat) + cor_mat2) / 2

    # calculate speed vector probability distribution by simulation
    simulated_speed_matrix = speed_sim(traffic_row, window_size)
    score_matrix = score_matrix * simulated_speed_matrix

    max_cor = np.nanmax(score_matrix)
    id_max_cor = np.array(np.unravel_index(np.nanargmax(score_matrix, axis=None), cor_mat.shape))
    vec = id_max_cor - np.array([half_win, half_win])

    # 速度計算(km/h)
    speed = np.linalg.norm(vec) * pix_size / time_dif
    speed_km_h = speed * 3600 / 1000
    # 角度は北を0、東を90、西を-90、南を180とする
    azimuth = np.arctan2(vec[0], vec[1]) * 180 / np.pi

    return speed_km_h, azimuth, max_cor


# 車間距離を計算する関数
def calculate_inter_dist(vehicle_table, edge_table, node_table):
    s_dict = dict(zip(edge_table['edge_id'], edge_table['from']))
    e_dict = dict(zip(edge_table['edge_id'], edge_table['to']))
    lon_dict = dict(zip(node_table['node_id'], node_table['lon']))
    lat_dict = dict(zip(node_table['node_id'], node_table['lat']))

    s_lon = vehicle_table['edge_id'].replace(s_dict).replace(lon_dict).values
    s_lat = vehicle_table['edge_id'].replace(s_dict).replace(lat_dict).values
    e_lon = vehicle_table['edge_id'].replace(e_dict).replace(lon_dict).values
    e_lat = vehicle_table['edge_id'].replace(e_dict).replace(lat_dict).values

    vehicle_lon = vehicle_table['lon'].values
    vehicle_lat = vehicle_table['lat'].values

    vec_road = np.array([e_lon - s_lon, e_lat - s_lat]).T
    vec_from_s = np.array([vehicle_lon - s_lon, vehicle_lat - s_lat]).T

    dist_from_s, vec_road = calc_dist_from_start(vec_road, vec_from_s)
    vehicle_table['d_from_s'] = dist_from_s

    vehicle_table = vehicle_table.sort_values(['edge_id', 'lane', 'd_from_s']).reset_index(drop=True)
    inter_dist = vehicle_table['d_from_s'].diff(periods=-1).values
    same_lane = vehicle_table[['edge_id', 'lane']].diff(periods=-1)

    bool_same_lane = ((same_lane['edge_id'].values * same_lane['lane'].values) == 0) * 1

    vehicle_table['inter_dist'] = inter_dist * bool_same_lane
    vehicle_table['inter_dist'] = vehicle_table['inter_dist'].replace(0, 100)

    return vehicle_table.drop(['lon', 'lat', 'd_from_s'], axis=1), vec_road


# easily calculate speed using spatial correlation
def get_speed(src_img, dst_img, ulx, uly, lrx, lry, window_size=31, pix_size=0.3, time_dif=0.2):
    half_win = (window_size-1) // 2
    w = lrx - ulx
    h = lry - uly

    src_box = np.asarray(src_img.crop((ulx, uly, lrx, lry)))
    dst_box = np.asarray(dst_img.crop((ulx-half_win, uly-half_win, lrx+half_win, lry+half_win)))
    cor_mat = cross_cor(src_box, dst_box, h, w, window_size)
    max_cor = np.nanmax(cor_mat)
    id_max_cor = np.array(np.unravel_index(np.nanargmax(cor_mat, axis=None), cor_mat.shape))
    vec = id_max_cor - np.array([half_win, half_win])

    # 速度計算(km/h)
    speed = np.linalg.norm(vec, ord=2) * pix_size / time_dif
    speed_km_h = speed * 3600 / 1000
    # 角度は北を0、東を90、西を-90、南を180とする
    azimuth = np.arctan2(-vec[0], vec[1]) * 180 / np.pi

    return speed_km_h, azimuth, max_cor, vec
