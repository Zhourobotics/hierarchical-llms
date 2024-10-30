import open3d as o3d
import numpy as np
import os
import random
import itertools


def load_pcd(file_path):
    cloud = o3d.io.read_point_cloud(file_path)
    return cloud

def save_pcd_file(cloud, file_path):
    # cloud = o3d.geometry.PointCloud()
    # cloud.points = o3d.utility.Vector3dVector(data)
    o3d.io.write_point_cloud(file_path, cloud)


def get_bounding_box(data):
    min_x, min_y, min_z = np.min(data, axis=0)
    max_x, max_y, max_z = np.max(data, axis=0)
    return min_x, min_y, min_z, max_x, max_y, max_z


def sample_segments(pcd, segment_size, num_samples, num_filter, output_dir):
    print("Sampling segments...")

    print("Number of points in the map:", len(pcd.points))
    print("pcd.colors:", pcd.colors[0])
    colors = np.asarray(pcd.colors)
    data = np.asarray(pcd.points)
    print("Shape of data:", data.shape)
    min_x, min_y, min_z, max_x, max_y, max_z = get_bounding_box(data)
    print(f"Map bounding box: ({min_x}, {min_y}, {min_z}) - ({max_x}, {max_y}, {max_z})")
    segment_x, segment_y, segment_z = segment_size


    i = 0
    while i < num_samples:
        print(f"------ Sample {i + 1}/{num_samples} ------")
        # x_start = random.uniform(min_x, max_x - segment_x)
        # y_start = random.uniform(min_y, max_y - segment_y)
        # z_start = random.uniform(min_z, max_z - segment_z)

        # x_start = -22
        # y_start = -22
        # z_start = -70
        x_start = -150
        y_start = -145
        z_start = -70
        x_end, y_end, z_end = x_start + segment_x, y_start + segment_y, z_start + segment_z

        print(f"Segment bounding box: ({x_start}, {y_start}, {z_start}) - ({x_end}, {y_end}, {z_end})")

        # segment_data = data[np.logical_and.reduce([
        #     data[:, 0] >= x_start,
        #     data[:, 0] <= x_end,
        #     data[:, 1] >= y_start,
        #     data[:, 1] <= y_end,
        #     data[:, 2] >= z_start,
        #     data[:, 2] <= z_end])]

        bounds = [[x_start, x_end], [y_start, y_end], [z_start, z_end]]  # set the bounds
        bounding_box_points = list(itertools.product(*bounds))  # create limit points
        bounding_box = o3d.geometry.AxisAlignedBoundingBox.create_from_points(
            o3d.utility.Vector3dVector(bounding_box_points))  # create bounding box object
        print(bounding_box)
        pcd_crop = pcd.crop(bounding_box)

        #move all the points to the origin
        pcd_crop.translate(-pcd_crop.get_center())
        #print(pcd_crop.colors[0])
        #print(segment_data.shape)
        pcd_crop.translate([0, 0, -5])

        shape = np.asarray(pcd_crop.points).shape

        print(f"Segment data size : ({shape})")

        #if shape[0] > num_filter:

        print(os.path.join(output_dir, f'{i}.pcd'))
        save_pcd_file(pcd_crop, os.path.join(output_dir, f'{i}.pcd'))
        i += 1

if __name__ == "__main__":
    #input_pcd_dir = "../datasets/PCDs_Fernando/falcon_forest_3_road_2.pcd"
    #input_pcd_dir = "../datasets/single_map_dataset/tmp.pcd"
    input_pcd_dir = "../../data/source/WMSC_points.pcd"
    segment_size = (50.0, 50.0, 50.0)  # Size of each segment in meters (x, y, z)
    num_samples = 1
    num_filter = 40000
    output_dir = "../../data/"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    pcd = load_pcd(input_pcd_dir)

    # print the color 
    sample_segments(pcd, segment_size, num_samples, num_filter, output_dir)