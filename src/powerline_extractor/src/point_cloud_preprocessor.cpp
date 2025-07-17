#include "point_cloud_preprocessor.h"

PointCloudPreprocessor::PointCloudPreprocessor(ros::NodeHandle& nh)
    : octree_(0.1), processed_cloud_(new pcl::PointCloud<pcl::PointXYZI>){
    loadParameters(nh);

    // 在构造函数的最后添加
    colored_cloud_pub_ = nh.advertise<sensor_msgs::PointCloud2>("colored_clusters", 1);



}

void PointCloudPreprocessor::loadParameters(ros::NodeHandle& nh) {
    
    nh.param("preprocessor/cube_size", cube_size_, 70.0);
    nh.param("preprocessor/octree_resolution", octree_resolution_, 0.1);
    nh.param("preprocessor/voxel_leaf_size", voxel_leaf_size_, 0.1);

    // 在现有参数加载后添加
    nh.param("preprocessor/cluster_tolerance", cluster_tolerance_, 0.5);
    nh.param("preprocessor/min_cluster_size", min_cluster_size_, 10);
    nh.param("preprocessor/max_cluster_size", max_cluster_size_, 25000);



    ROS_INFO("点云处理的参数:cube_size = %.2f,octree_resolution = %.2f,voxel_leaf_size = %.2f",cube_size_,octree_resolution_,voxel_leaf_size_);
    ROS_INFO("聚类参数: cluster_tolerance = %.2f, min_cluster_size = %d, max_cluster_size = %d", 
         cluster_tolerance_, min_cluster_size_, max_cluster_size_);


    // 设置 Octree 分辨率
    octree_.setResolution(octree_resolution_);
    // 设置体素网格滤波器参数
    voxel_grid_filter_.setLeafSize(voxel_leaf_size_, voxel_leaf_size_, voxel_leaf_size_);
}

void PointCloudPreprocessor::cropPointCloud(const pcl::PointCloud<pcl::PointXYZI>::ConstPtr& input_cloud,
                                            pcl::PointCloud<pcl::PointXYZI>::Ptr& cropped_cloud) {
    crop_box_filter_.setInputCloud(input_cloud);
    crop_box_filter_.setMin(Eigen::Vector4f(-cube_size_ / 2, -cube_size_ / 2, 0, 1.0));
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

    // 5. 保存处理后的点云
    *processed_cloud_ = *downsampled_cloud;
    // 3. 基于密度聚类
    performClustering(downsampled_cloud);
}

pcl::PointCloud<pcl::PointXYZI>::Ptr PointCloudPreprocessor::getProcessedCloud() const {
    return processed_cloud_;
}

void PointCloudPreprocessor::performClustering(const pcl::PointCloud<pcl::PointXYZI>::Ptr& input_cloud) {
    cluster_results_.clear();
    
    // 创建KdTree对象用于搜索
    pcl::search::KdTree<pcl::PointXYZI>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZI>);
    tree->setInputCloud(input_cloud);
    
    // 创建聚类提取器
    std::vector<pcl::PointIndices> cluster_indices;
    pcl::EuclideanClusterExtraction<pcl::PointXYZI> ec;
    ec.setClusterTolerance(cluster_tolerance_);
    ec.setMinClusterSize(min_cluster_size_);
    ec.setMaxClusterSize(max_cluster_size_);
    ec.setSearchMethod(tree);
    ec.setInputCloud(input_cloud);
    ec.extract(cluster_indices);
    
    // 处理每个聚类
    int cluster_id = 0;
    for (const auto& indices : cluster_indices) {
        ClusterInfo cluster_info;
        cluster_info.cluster_id = cluster_id++;
        cluster_info.cluster_cloud.reset(new pcl::PointCloud<pcl::PointXYZI>);
        cluster_info.point_count = indices.indices.size();
        
        // 提取聚类点云
        pcl::ExtractIndices<pcl::PointXYZI> extract;
        extract.setInputCloud(input_cloud);
        pcl::PointIndices::Ptr indices_ptr(new pcl::PointIndices(indices));
        extract.setIndices(indices_ptr);
        extract.filter(*cluster_info.cluster_cloud);
        
        // 计算聚类中心
        Eigen::Vector4f centroid;
        pcl::compute3DCentroid(*cluster_info.cluster_cloud, centroid);
        cluster_info.centroid = Eigen::Vector3f(centroid[0], centroid[1], centroid[2]);
        
        cluster_results_.push_back(cluster_info);
    }
    
    ROS_INFO("检测到 %zu 个聚类", cluster_results_.size());
}

pcl::PointCloud<pcl::PointXYZRGB>::Ptr PointCloudPreprocessor::createColoredClusterCloud() {
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr colored_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
    
    // 预定义颜色 (RGB)
    std::vector<std::vector<int>> colors = {
        {255, 0, 0},     // 红色
        {0, 255, 0},     // 绿色
        {0, 0, 255},     // 蓝色
        {255, 255, 0},   // 黄色
        {255, 0, 255},   // 紫色
        {0, 255, 255},   // 青色
        {255, 128, 0},   // 橙色
        {128, 0, 255},   // 紫罗兰
        {255, 192, 203}, // 粉色
        {128, 128, 128}  // 灰色
    };
    
    for (const auto& cluster : cluster_results_) {
        // 选择颜色
        auto color = colors[cluster.cluster_id % colors.size()];
        
        // 为聚类中的每个点添加颜色
        for (const auto& point : cluster.cluster_cloud->points) {
            pcl::PointXYZRGB colored_point;
            colored_point.x = point.x;
            colored_point.y = point.y;
            colored_point.z = point.z;
            colored_point.r = color[0];
            colored_point.g = color[1];
            colored_point.b = color[2];
            
            colored_cloud->points.push_back(colored_point);
        }
    }
    
    colored_cloud->width = colored_cloud->points.size();
    colored_cloud->height = 1;
    colored_cloud->is_dense = true;
    
    return colored_cloud;
}

void PointCloudPreprocessor::publishColoredClusters() {
    if (cluster_results_.empty()) {
        return;
    }
    
    // 创建彩色点云
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr colored_cloud = createColoredClusterCloud();
    
    // 转换为ROS消息并发布
    sensor_msgs::PointCloud2 cloud_msg;
    pcl::toROSMsg(*colored_cloud, cloud_msg);
    cloud_msg.header.frame_id = "map";  // 根据你的坐标系修改
    cloud_msg.header.stamp = ros::Time::now();
    
    colored_cloud_pub_.publish(cloud_msg);
    
    ROS_INFO("发布了 %zu 个聚类的彩色点云，总共 %zu 个点", 
             cluster_results_.size(), colored_cloud->points.size());
}

std::vector<ClusterInfo> PointCloudPreprocessor::getClusterResults() const {
    return cluster_results_;
}