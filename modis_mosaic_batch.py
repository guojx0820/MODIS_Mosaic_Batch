# -*- coding: utf-8 -*-
'''
@Time ： 2023/7/13 11:10
@Auth ： Guo Jiaxiang
@Blog : https://www.guojxblog.cn
@GitHub : https://github.com/guojx0820
@Email : guojx0820@gmail.com
'''
# -*- coding: utf-8 -*-

import numpy as np
from osgeo import gdal, osr
import os
from pyhdf.SD import SD
import cv2 as cv
import time
import math
import shutil as sh


class MODIS_Radiance_Lib_Create:
    def __init__(self, l1b_file, cloud_file, out_name):  #
        self.l1b_file = l1b_file
        self.cloud_file = cloud_file
        self.out_name = out_name
        self.geo_resolution = 0.01
        self.lon_min = 9999.0
        self.lon_max = -9999.0
        self.lat_min = 9999.0
        self.lat_max = -9999.0
        self.band_layer = 16

    def _read_modis_data_(self):
        modis_l1b = SD(self.l1b_file)
        modis_cloud = SD(self.cloud_file)
        qkm_rad = self._radical_calibration_(modis_l1b, 'EV_250_Aggr1km_RefSB', 'radiance_scales',
                                             'radiance_offsets')
        hkm_rad = self._radical_calibration_(modis_l1b, 'EV_500_Aggr1km_RefSB', 'radiance_scales',
                                             'radiance_offsets')
        km_rad = self._radical_calibration_(modis_l1b, 'EV_1KM_RefSB', 'radiance_scales',
                                            'radiance_offsets')
        cloud_data = modis_cloud.select('Cloud_Mask').get()
        lon = modis_l1b.select('Longitude').get()
        lat = modis_l1b.select('Latitude').get()
        del modis_l1b, modis_cloud
        return qkm_rad, hkm_rad, km_rad, cloud_data, lon, lat

    def _read_tiff_info_(self, file):
        dataset = gdal.Open(file)
        data = dataset.ReadAsArray()
        data[data > 500] = 0.0
        data_col = dataset.RasterXSize
        data_row = dataset.RasterYSize
        geo_transform = dataset.GetGeoTransform()
        projection = dataset.GetProjection()
        del dataset
        return data, data_row, data_col, geo_transform, projection

    def _radical_calibration_(self, modis_l1b, dataset_name, scales, offsets):
        object = modis_l1b.select(dataset_name)
        data = object.get()
        scales = object.attributes()[scales]
        offsets = object.attributes()[offsets]
        data_rad = np.zeros((data.shape[0], data.shape[1], data.shape[2]), dtype=np.float32)
        for i_layer in range(data.shape[0]):
            data_rad[i_layer, :, :] = scales[i_layer] * (data[i_layer, :, :] - offsets[i_layer])
        return data_rad

    def _cloud_mask_(self, cloud_data):
        cloud_0 = cloud_data[0, :, :]
        cloud_0 = (np.int64(cloud_0 < 0) * (256 + cloud_0)) + (np.int64(cloud_0 >= 0) * cloud_0)
        cloud_binary = np.zeros((cloud_0.shape[0], cloud_0.shape[1], 8), dtype=np.int64)
        for i_cloud in range(8):
            cloud_binary[:, :, i_cloud] = cloud_0 % 2
            cloud_0 //= 2
        clear_result = np.int64(cloud_binary[:, :, 0] == 1) & np.int64(cloud_binary[:, :, 1] == 1) \
                       & np.int64(cloud_binary[:, :, 2] == 1)
        ocean_result = np.int64(cloud_binary[:, :, 6] == 0) & np.int64(cloud_binary[:, :, 7] == 0)
        cloud_result = np.int64(clear_result == 0) | np.int64(ocean_result == 0)
        ocean_clear = np.int64(clear_result == 1) & np.int64(ocean_result == 1)
        del cloud_result
        return ocean_result

    def _band_extract_(self, lon, lat, qkm_rad, hkm_rad, km_rad, clear_result):
        rad_arr = np.array([km_rad[0], km_rad[1], hkm_rad[0], km_rad[2], km_rad[3], km_rad[4], hkm_rad[1], qkm_rad[0],
                            km_rad[5], km_rad[7], km_rad[9], qkm_rad[1], km_rad[10], km_rad[11], km_rad[13],
                            km_rad[12]], dtype=np.float32)
        del qkm_rad, hkm_rad, km_rad
        extraction_band = rad_arr[i_layer] * clear_result
        # hist_min_band = extraction_band - np.min(extraction_band[np.nonzero(extraction_band)])
        # hist_min_band[hist_min_band < 0] = 0.0
        # print(extraction_band, hist_min_band, sep='\n')
        del clear_result
        geo_band, lon_min, lon_max, lat_min, lat_max = self._georeference_(lon, lat, extraction_band)
        filter_band = self._average_filtering_(geo_band)
        return filter_band, lon_min, lon_max, lat_min, lat_max

    def _min_value_calculate_(self, temp_band_list, temp_lon_lat_list):
        for i_ll in temp_lon_lat_list:
            temp_lon_min = i_ll[0]
            temp_lon_max = i_ll[1]
            temp_lat_min = i_ll[2]
            temp_lat_max = i_ll[3]
            if temp_lon_min < self.lon_min: self.lon_min = temp_lon_min
            if temp_lon_max > self.lon_max: self.lon_max = temp_lon_max
            if temp_lat_min < self.lat_min: self.lat_min = temp_lat_min
            if temp_lat_max > self.lat_max: self.lat_max = temp_lat_max
        data_box_geo_col = math.ceil((self.lon_max - self.lon_min) / self.geo_resolution)
        data_box_geo_row = math.ceil((self.lat_max - self.lat_min) / self.geo_resolution)
        data_box_geo_min = np.array(np.zeros((data_box_geo_row, data_box_geo_col))) + 9999.0
        for i_band, j_ll in zip(temp_band_list, temp_lon_lat_list):
            temp_lon_min = j_ll[0]
            temp_lat_max = j_ll[3]

            for i_data_col in range(i_band.shape[1]):
                for i_data_row in range(i_band.shape[0]):
                    temp_lon = temp_lon_min + self.geo_resolution * i_data_col
                    temp_lat = temp_lat_max - self.geo_resolution * i_data_row
                    data_box_col_pos = math.floor((temp_lon - self.lon_min) / self.geo_resolution)
                    data_box_row_pos = math.floor((self.lat_max - temp_lat) / self.geo_resolution)
                    if i_band[i_data_row, i_data_col] == 0:
                        continue
                    elif i_band[i_data_row, i_data_col] <= data_box_geo_min[data_box_row_pos, data_box_col_pos]:
                        data_box_geo_min[data_box_row_pos, data_box_col_pos] = i_band[i_data_row, i_data_col]
                    else:
                        continue
        data_box_geo_min[data_box_geo_min == 9999.0] = 0.0
        del temp_band_list, temp_lon_lat_list
        return data_box_geo_min, self.lon_min, self.lat_max

    def _georeference_(self, lon, lat, data):
        lon_interp = cv.resize(lon, (data.shape[1], data.shape[0]), interpolation=cv.INTER_LINEAR)
        lat_interp = cv.resize(lat, (data.shape[1], data.shape[0]), interpolation=cv.INTER_LINEAR)
        lon_min = np.min(lon_interp)
        lon_max = np.max(lon_interp)
        lat_min = np.min(lat_interp)
        lat_max = np.max(lat_interp)
        geo_box_col = np.int64(np.ceil((lon_max - lon_min) / self.geo_resolution))
        geo_box_row = np.int64(np.ceil((lat_max - lat_min) / self.geo_resolution))
        geo_box = np.zeros((geo_box_row, geo_box_col), dtype=np.float32)
        del geo_box_row, geo_box_col
        geo_box_col_pos = np.int64(np.floor((lon_interp - lon_min) / self.geo_resolution))
        geo_box_row_pos = np.int64(np.floor((lat_max - lat_interp) / self.geo_resolution))
        geo_box[geo_box_row_pos, geo_box_col_pos] = data
        del lon_interp, lat_interp, data
        return geo_box, lon_min, lon_max, lat_min, lat_max

    def _average_filtering_(self, geo_box):
        geo_box_plus = np.zeros((geo_box.shape[0] + 2, geo_box.shape[1] + 2), dtype=np.float32) - 9999.0
        geo_box_plus[1:geo_box.shape[0] + 1, 1:geo_box.shape[1] + 1] = geo_box
        geo_box_out = np.zeros((geo_box.shape[0], geo_box.shape[1]), dtype=np.float32)
        for i_geo_box_row in range(1, geo_box.shape[0] + 1):
            for i_geo_box_col in range(1, geo_box.shape[1] + 1):
                if geo_box_plus[i_geo_box_row, i_geo_box_col] == 0.0:
                    temp_window = geo_box_plus[i_geo_box_row - 1:i_geo_box_row + 2,
                                  i_geo_box_col - 1:i_geo_box_col + 2]
                    temp_window = temp_window[temp_window > 0]  # 保留窗口中大于零的数
                    temp_window_sum = np.sum(temp_window)
                    temp_window_num = np.sum(np.int64(temp_window > 0.0))  # 计算大于零的值的个数
                    if temp_window_num >= 1:
                        geo_box_out[i_geo_box_row - 1, i_geo_box_col - 1] = temp_window_sum / temp_window_num
                    else:
                        geo_box_out[i_geo_box_row - 1, i_geo_box_col - 1] = 0.0
                else:
                    geo_box_out[i_geo_box_row - 1, i_geo_box_col - 1] = geo_box_plus[
                        i_geo_box_row, i_geo_box_col]
        del geo_box_plus
        return geo_box_out

    def _write_tiff_(self, data, lon_min, lat_max, out_name):
        if data.ndim == 3:
            band_count, rows, cols = data.shape
        else:
            band_count, (rows, cols) = 1, data.shape
        driver = gdal.GetDriverByName('GTiff')
        out_raster = driver.Create(out_name, cols, rows, band_count, gdal.GDT_Float32)
        out_raster.SetGeoTransform((lon_min, self.geo_resolution, 0, lat_max, 0, self.geo_resolution))
        out_raster_SRS = osr.SpatialReference()
        # 代码4326表示WGS84坐标
        out_raster_SRS.ImportFromEPSG(4326)
        out_raster.SetProjection(out_raster_SRS.ExportToWkt())
        if band_count == 1:
            out_raster.GetRasterBand(1).WriteArray(data)
        else:
            # 获取数据集第一个波段，是从1开始，不是从0开始
            for i_band_count in range(band_count):
                out_raster.GetRasterBand(i_band_count + 1).WriteArray(data[i_band_count])
        out_raster.FlushCache()
        del out_raster
        out_raster = None


if __name__ == '__main__':
    start_time = time.time()
    input_directory = '/mnt/e/Experiments/Aerosol_Classification/Data/MOD_MYD021KM2021_03_Demo/'
    output_directory = '/mnt/e/Experiments/Aerosol_Classification/Data/Results/MODIS_Mosaic/'
    band_layer = 16
    if os.path.exists(output_directory) == False:
        os.makedirs(output_directory)
    dirs_list = []
    for root, dirs, files in os.walk(input_directory):
        modis_file_list = [input_directory + i_modis for i_modis in files if
                           i_modis.endswith('.hdf') and i_modis.startswith('M', 0)]
        dirs_list.extend(dirs)
    # for i_file in modis_file_list:
    #     day_mosaic_name = input_directory + os.path.basename(i_file)[0:3] + os.path.basename(i_file)[10:17]
    #     print(i_file, day_mosaic_name)
    #     if not os.path.exists(day_mosaic_name):
    #         os.makedirs(day_mosaic_name)
    #     sh.move(i_file, day_mosaic_name)
    for i_dirs in range(len(dirs_list)):
        day_start_time = time.time()
        temp_input_directory = input_directory + dirs_list[i_dirs] + '/'
        for root, dirs, files in os.walk(temp_input_directory):
            l1b_file_list = [temp_input_directory + i_hdf for i_hdf in files if
                             i_hdf.endswith('.hdf') and i_hdf.startswith('D02', 2)]
            cloud_file_list = [temp_input_directory + i_hdf for i_hdf in files if
                               i_hdf.endswith('.hdf') and i_hdf.startswith('D35', 2)]
        temp_output_directory = output_directory + dirs_list[i_dirs] + '/'
        if os.path.exists(temp_output_directory) == False:
            os.makedirs(temp_output_directory)
        temp_out_name = temp_output_directory + str(
            dirs_list[i_dirs]) + '_16BandsMosaic_Western_Pacific_Ocean_Cloud.tiff'
        for i_layer in range(band_layer):
            if i_layer + 1 < 10:
                out_layer_name = temp_output_directory + str(dirs_list[i_dirs]) + \
                                 '_16BandsMosaic_Western_Pacific_Ocean_Cloud' + '_Layer0' + str(i_layer + 1) + '.tiff'
            else:
                out_layer_name = temp_output_directory + str(dirs_list[i_dirs]) + \
                                 '_16BandsMosaic_Western_Pacific_Ocean_Cloud' + '_Layer' + str(i_layer + 1) + '.tiff'
            start_time_layer = time.time()
            temp_band_list = []
            temp_lon_lat_list = []
            for i_l1b in l1b_file_list:
                for i_cloud in cloud_file_list:
                    if os.path.basename(i_l1b)[0:3] == os.path.basename(i_cloud)[0:3] and \
                            os.path.basename(i_l1b)[10:22] == os.path.basename(i_cloud)[10:22]:
                        start_time_each = time.time()
                        modis_rad_lib = MODIS_Radiance_Lib_Create(i_l1b, i_cloud, out_layer_name)
                        qkm_rad, hkm_rad, km_rad, cloud_data, lon, lat = modis_rad_lib._read_modis_data_()
                        clear_result = modis_rad_lib._cloud_mask_(cloud_data)
                        band, lon_min, lon_max, lat_min, lat_max = modis_rad_lib._band_extract_(lon, lat, qkm_rad,
                                                                                                hkm_rad,
                                                                                                km_rad, clear_result)
                        temp_band_list.append(band)
                        temp_lon_lat_list.append([lon_min, lon_max, lat_min, lat_max])
                        del qkm_rad, hkm_rad, km_rad, cloud_data, lon, lat, clear_result, band, lon_min, lon_max, lat_min, lat_max
                        end_time_each = time.time()
                        run_time_each = round(end_time_each - start_time_each, 3)
                        print('The image of ' + os.path.basename(i_l1b)[10:22] +
                              ' is finished! The time consuming is ' + str(run_time_each) + ' s.')
            data_box_geo_min, lon_min, lat_max = modis_rad_lib._min_value_calculate_(temp_band_list, temp_lon_lat_list)
            data_box_geo_out = modis_rad_lib._average_filtering_(data_box_geo_min)
            modis_rad_lib._write_tiff_(data_box_geo_out, lon_min, lat_max, out_layer_name)
            # del data_box_geo_min, lon_min, lat_max, data_box_geo_out
            end_time_layer = time.time()
            run_time_layer = round(end_time_layer - start_time_layer, 3)
            print('The layer ' + str(i_layer + 1) +
                  ' is finished! The time consuming is ' + str(run_time_layer) + ' s.')
        modis_rad_lib = MODIS_Radiance_Lib_Create(i_l1b, i_cloud, temp_out_name)  #
        for root, dirs, files in os.walk(temp_output_directory):
            file_list = [temp_output_directory + i_tiff for i_tiff in files if
                         i_tiff.endswith('.tiff') and i_tiff.startswith(
                             str(dirs_list[i_dirs]) + '_16BandsMosaic_Western_Pacific_Ocean_Cloud_Layer')]
        layer_band_list = []
        for i_file, j_num in zip(file_list, range(band_layer)):
            if j_num < 9:
                print(i_file[0:-7] + str(0) + str(j_num + 1) + '.tiff')
                data, data_row, data_col, geo_transform, projection = modis_rad_lib._read_tiff_info_(
                    i_file[0:-7] + str(0) + str(j_num + 1) + '.tiff')
            else:
                print(i_file[0:-7] + str(j_num + 1) + '.tiff')
                data, data_row, data_col, geo_transform, projection = modis_rad_lib._read_tiff_info_(
                    i_file[0:-7] + str(j_num + 1) + '.tiff')
            layer_band_list.append(data)
            # os.remove(i_file)
            del data
        layer_lon_min = geo_transform[0]
        layer_lat_max = geo_transform[3]
        layer_band_arr = np.array(layer_band_list, dtype=np.float32)
        print(layer_band_arr.shape)
        del layer_band_list, data_row, data_col, geo_transform, projection
        modis_rad_lib._write_tiff_(layer_band_arr, layer_lon_min, layer_lat_max, temp_out_name)
        del layer_band_arr, layer_lon_min, layer_lat_max
        day_end_time = time.time()
        day_run_time = round(day_end_time - day_start_time, 3)
        print('The image of ' + str(dirs_list[i_dirs]) + ' is processed! The total time consuming is '
              + str(day_run_time) + ' s.')
    end_time = time.time()
    run_time = round(end_time - start_time, 3)
    print('All the images were processed! The total time consuming is ' + str(run_time) + ' s.')
