/**
 * 处理传入的动态点云数据
 * 剪切点云：使用 PCL 的 CropBox 滤波器。

降采样：使用 PCL 的 VoxelGrid 滤波器。

Octree 处理：使用 pcl::octree::OctreePointCloudChangeDetector 检测变化并标记。

参数配置：从 YAML 文件读取参数，通过 ROS 参数服务器加载。

跨类共享：提供 Octree 和处理后点云的访问接口。

输入输出：输入和输出均为 pcl::PointCloud<pcl::PointXYZI>::Ptr 类型。
 */

#ifndef POINT_CLOUD_PREPROCESSOR_H
#define POINT_CLOUD_PREPROCESSOR_H

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/octree/octree_pointcloud_changedetector.h>
#include <pcl/filters/crop_box.h>
#include <pcl/filters/voxel_grid.h>
#include <ros/ros.h>

class PointCloudPreprocessor {
public:
    // 构造函数，初始化参数和成员
    PointCloudPreprocessor(ros::NodeHandle& nh);

    // 处理点云的接口
    void processPointCloud(const pcl::PointCloud<pcl::PointXYZI>::ConstPtr& input_cloud);

    // 获取处理后的点云
    pcl::PointCloud<pcl::PointXYZI>::Ptr getProcessedCloud() const;

    // 获取 Octree（const 引用，供其他模块使用）
    const pcl::octree::OctreePointCloudChangeDetector<pcl::PointXYZI>& getOctree() const;

private:
    // 加载参数
    void loadParameters(ros::NodeHandle& nh);

    // 剪切点云
    void cropPointCloud(const pcl::PointCloud<pcl::PointXYZI>::ConstPtr& input_cloud,
                        pcl::PointCloud<pcl::PointXYZI>::Ptr& cropped_cloud);

    // 降采样点云
    void downsamplePointCloud(const pcl::PointCloud<pcl::PointXYZI>::Ptr& cropped_cloud,
                              pcl::PointCloud<pcl::PointXYZI>::Ptr& downsampled_cloud);

    // 参数
    double cube_size_;          // 立方体边长
    double octree_resolution_;  // Octree 分辨率
    double voxel_leaf_size_;    // 降采样体素大小

    // PCL 滤波器和数据结构
    pcl::CropBox<pcl::PointXYZI> crop_box_filter_;
    pcl::octree::OctreePointCloudChangeDetector<pcl::PointXYZI> octree_;
    pcl::VoxelGrid<pcl::PointXYZI> voxel_grid_filter_;

    // 处理后的点云
    pcl::PointCloud<pcl::PointXYZI>::Ptr processed_cloud_;
};

#endif // POINT_CLOUD_PREPROCESSOR_H
