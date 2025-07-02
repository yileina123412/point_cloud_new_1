#include "point_cloud_preprocessor.h"

PointCloudPreprocessor::PointCloudPreprocessor(ros::NodeHandle& nh)
    : octree_(0.1), processed_cloud_(new pcl::PointCloud<pcl::PointXYZI>) {
    loadParameters(nh);
}

void PointCloudPreprocessor::loadParameters(ros::NodeHandle& nh) {
    
    nh.param("preprocessor/cube_size", cube_size_, 70.0);
    nh.param("preprocessor/octree_resolution", octree_resolution_, 0.1);
    nh.param("preprocessor/voxel_leaf_size", voxel_leaf_size_, 0.1);

    ROS_INFO("点云处理的参数:cube_size = %.2f,octree_resolution = %.2f,voxel_leaf_size = %.2f",cube_size_,octree_resolution_,voxel_leaf_size_);

    // 设置 Octree 分辨率
    octree_.setResolution(octree_resolution_);
    // 设置体素网格滤波器参数
    voxel_grid_filter_.setLeafSize(voxel_leaf_size_, voxel_leaf_size_, voxel_leaf_size_);
}

void PointCloudPreprocessor::cropPointCloud(const pcl::PointCloud<pcl::PointXYZI>::ConstPtr& input_cloud,
                                            pcl::PointCloud<pcl::PointXYZI>::Ptr& cropped_cloud) {
    crop_box_filter_.setInputCloud(input_cloud);
    crop_box_filter_.setMin(Eigen::Vector4f(-cube_size_ / 2, -cube_size_ / 2, -cube_size_ / 2, 1.0));
    crop_box_filter_.setMax(Eigen::Vector4f(cube_size_ / 2, cube_size_ / 2, cube_size_ / 2, 1.0));
    crop_box_filter_.filter(*cropped_cloud);
}

void PointCloudPreprocessor::downsamplePointCloud(const pcl::PointCloud<pcl::PointXYZI>::Ptr& cropped_cloud,
                                                  pcl::PointCloud<pcl::PointXYZI>::Ptr& downsampled_cloud) {
    voxel_grid_filter_.setInputCloud(cropped_cloud);
    voxel_grid_filter_.filter(*downsampled_cloud);
}

void PointCloudPreprocessor::processPointCloud(const pcl::PointCloud<pcl::PointXYZI>::ConstPtr& input_cloud) {
    // 1. 剪切点云
    pcl::PointCloud<pcl::PointXYZI>::Ptr cropped_cloud(new pcl::PointCloud<pcl::PointXYZI>);
    cropPointCloud(input_cloud, cropped_cloud);

    // 2. 降采样
    pcl::PointCloud<pcl::PointXYZI>::Ptr downsampled_cloud(new pcl::PointCloud<pcl::PointXYZI>);
    downsamplePointCloud(cropped_cloud, downsampled_cloud);

    octree_.deleteTree();
    // 3. 更新 Octree
    octree_.setInputCloud(downsampled_cloud);
    octree_.addPointsFromInputCloud();

    // 4. 检测变化并标记
    std::vector<int> new_point_indices;
    octree_.getPointIndicesFromNewVoxels(new_point_indices);
    // 注意：这里仅获取变化点的索引，后续模块可根据需要处理这些区域

    // 5. 保存处理后的点云
    *processed_cloud_ = *downsampled_cloud;
}

pcl::PointCloud<pcl::PointXYZI>::Ptr PointCloudPreprocessor::getProcessedCloud() const {
    return processed_cloud_;
}

const pcl::octree::OctreePointCloudChangeDetector<pcl::PointXYZI>& PointCloudPreprocessor::getOctree() const {
    return octree_;
}
