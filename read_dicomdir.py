import pydicom
import numpy as np # linear algebra
import os
import open3d as o3d
import scipy
import matplotlib.pyplot as plt
import copy
import pandas as pd

from skimage import measure, morphology
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# dicomdir_path="/home/qiwen/Documents/CT_Daten/Case WM/2023-08-24-002/DICOMDIR"
# images_folder="/home/qiwen/Documents/CT_Daten/Case WM/2023-08-24-002/IMAGES"

df = pd.read_excel('/home/qiwen/Documents/CT_Daten/Points and definitions.xlsx', sheet_name='Case RS')
dicomdir_path='/home/qiwen/Documents/CT_Daten/Case RS/2023-08-24-001/DICOMDIR'
images_folder="/home/qiwen/Documents/CT_Daten/Case RS/2023-08-24-001/IMAGES"
target_study_id="8ac05771"
target_SeriesNumber="301"

class DICOMDIR:
    def __init__(self, dicomdir_path, images_folder, target_study_id, target_SeriesNumber, threshold=400, save=True):
        self.dicomdir_path = dicomdir_path
        self.images_folder = images_folder
        self.target_study_id = target_study_id
        self.target_SeriesNumber = target_SeriesNumber
        
        self.dicomdir = pydicom.dcmread(dicomdir_path)
        self.slices = self.load_scan_by_series()
        
        self.volume_image = self.get_pixels_hu()
        self.resample(new_spacing=[1,1,1])
        self.point_cloud = self.get_open3d_pc(threshold, save)
        
        self.image_orientation = self.slices[-1].ImageOrientationPatient
        self.image_position = self.slices[-1].ImagePositionPatient
        self.transformation_matrix, self.transformation_matrix_inv = self.get_transform_matrix()

    def get_study_by_id(self):
        # 遍历患者记录
        for patient_record in self.dicomdir.patient_records:
            # 遍历患者的检查记录
            for study in patient_record.children:
                # 获取当前 study 的 StudyID
                study_id = study.StudyID
                
                # 检查是否与目标 Study ID 匹配
                if study_id == self.target_study_id:
                    return study
        
        # 如果未找到匹配的 Study ID
        return None

    def get_series_by_SeriesNumber(self):
        for series in self.study.children:
            series_number = series.SeriesNumber
            if series_number == self.target_SeriesNumber:
                return series
        return None

    def load_scan_by_series(self):
        """
        根据 Study ID 和 Series Description 加载特定序列的 DICOM 数据。
        
        参数:
            dicomdir_path: DICOMDIR 文件的路径
            images_folder: DICOM 图像文件所在的文件夹路径
            target_study_id: 要加载的 StudyInstanceUID (Study ID)
            target_SeriesNumber: 要加载的 SeriesNumber
        
        返回:
            slices: 排序后的 DICOM 切片数据
        """
        # 存储 DICOM 文件的路径和信息
        slices = []

        # 获取指定 Study ID 的研究
        self.study = self.get_study_by_id()
        if self.study is None:
            raise ValueError(f"未找到 Study ID 为 {self.target_study_id} 的数据。")

        # 获取指定 Series Description 的序列
        series = self.get_series_by_SeriesNumber()
        if series is None:
            raise ValueError(f"未找到 target_SeriesNumber 为 {self.target_SeriesNumber} 的数据。")

        # 遍历序列中的每个实例（影像文件）
        for instance in series.children:
            # 构建影像文件路径
            instance_path = os.path.join(self.images_folder, instance.ReferencedFileID[1])
            
            # 读取 DICOM 文件并添加到列表
            slices.append(pydicom.dcmread(instance_path))

        # 按 InstanceNumber 排序切片
        slices.sort(key=lambda x: int(x.InstanceNumber))
        return slices

    def get_pixels_hu(self):
        image = np.stack([s.pixel_array for s in self.slices[::-1]])
        # Convert to int16 (from sometimes int16), 
        # should be possible as values should always be low enough (<32k)
        image = image.astype(np.int16)

        # Set outside-of-scan pixels to 0
        # The intercept is usually -1024, so air is approximately 0
        image[image == -2000] = 0
        
        # Convert to Hounsfield units (HU)
        for slice_number in range(len(self.slices)):
            
            intercept = self.slices[slice_number].RescaleIntercept
            slope = self.slices[slice_number].RescaleSlope
            
            if slope != 1:
                image[slice_number] = slope * image[slice_number].astype(np.float64)
                image[slice_number] = image[slice_number].astype(np.int16)
                
            image[slice_number] += np.int16(intercept)
        
        return np.array(image, dtype=np.int16)

    def resample(self, new_spacing=[1,1,1]):
        # Determine current pixel spacing
        slice_thickness = float(self.slices[0].SliceThickness)
        pixel_spacing = list(self.slices[0].PixelSpacing)
        spacing = np.array([slice_thickness] + pixel_spacing, dtype=np.float32)

        resize_factor = spacing / new_spacing
        new_real_shape = self.volume_image.shape * resize_factor
        new_shape = np.round(new_real_shape)
        real_resize_factor = new_shape / self.volume_image.shape
        new_spacing = spacing / real_resize_factor
        
        self.volume_image = scipy.ndimage.interpolation.zoom(self.volume_image, real_resize_factor, mode='nearest')
        self.new_spacing = new_spacing

    def get_open3d_pc(self, threshold=-300, save=False):
        # Position the scan upright,
        p = self.volume_image.transpose(2, 1, 0)

        verts, faces, _, _ = measure.marching_cubes(p, threshold)  # Notice the extra underscore for normals
        
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(verts)
        
        if save==True:
            file_name="CT_point_cloud.ply"
            
            o3d.io.write_point_cloud(file_name, point_cloud)
            print(f"Point cloud saved to {file_name}")
        
        o3d.visualization.draw_geometries([point_cloud])
        
        return point_cloud

    def get_transform_matrix(self):
        
        orientation = -np.array(self.image_orientation).reshape(2, 3) # x and y axes in DICOM are inverted to 3D Slicer

        # 创建一个旋转矩阵
        rotation_matrix = np.zeros((3, 3))
        rotation_matrix[:, 0] = orientation[0]  # 第一行
        rotation_matrix[:, 1] = orientation[1]  # 第二行
        rotation_matrix[:, 2] = np.cross(orientation[0], orientation[1])  # 第三行 (法向量)

        # 创建平移向量
        translation_vector = np.array(self.image_position)
        translation_vector[:2]=-translation_vector[:2] # x and y axes in DICOM are inverted to 3D Slicer

        # 构建变换矩阵
        transformation_matrix = np.eye(4)
        transformation_matrix[:3, :3] = rotation_matrix.T  # 旋转矩阵
        transformation_matrix[:3, 3] = translation_vector  # 平移向量
        
        print("TrasnsMatrix is: \n", transformation_matrix) # point cloud to 3D Slice coord
        transformation_matrix_inv = np.linalg.inv(transformation_matrix) # 3D Slice to point cloud coord

        return transformation_matrix, transformation_matrix_inv

    def vis_marked_point(self, point_name, save=False):
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=5.0) 
        sphere.paint_uniform_color([0.5, 0.5, 0.5])

        selected_point = df[df['label'] == point_name][['transversal', 'sagittal', 'vertikal']].values[0]
        sphere.translate(selected_point)
        sphere.transform(self.transformation_matrix_inv)

        if save == True:
            # red_color = np.tile([0.8, 0.2, 0.0], (len(self.point_cloud.points), 1))
            # self.point_cloud.colors = o3d.utility.Vector3dVector(red_color)

            sphere_points = sphere.sample_points_uniformly(number_of_points=1000)
            
            # sphere_colors = np.tile([0.5, 0.5, 0.5], (len(sphere_points.points), 1))
            # sphere_points.colors = o3d.utility.Vector3dVector(sphere_colors)

            combined_point_cloud = self.point_cloud + sphere_points
            o3d.io.write_point_cloud("marked_point_cloud.ply", combined_point_cloud)

        o3d.visualization.draw_geometries([self.point_cloud, sphere])

if __name__ == "__main__":

    dicom_data=DICOMDIR(dicomdir_path, images_folder, target_study_id, target_SeriesNumber)
    
    dicom_data.vis_marked_point("N", True)
