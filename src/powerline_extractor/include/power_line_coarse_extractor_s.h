#ifndef POWER_LINE_EXTRACTOR_H_S
#define POWER_LINE_EXTRACTOR_H_S

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/octree/octree_pointcloud_changedetector.h>
#include <pcl/features/normal_3d.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <sensor_msgs/PointCloud2.h>
#include <pcl_conversions/pcl_conversions.h>
#include <memory>
#include <ros/ros.h>
#include "point_cloud_preprocessor.h"

class PowerLineCoarseExtractor {
public:
    PowerLineCoarseExtractor(ros::NodeHandle& nh);

    void extractPowerLinesByPoints(const pcl::PointCloud<pcl::PointXYZI>::Ptr& input_cloud);
    pcl::PointCloud<pcl::PointXYZI>::Ptr getExtractedCloud() const;
    pcl::PointCloud<pcl::PointXYZI>::Ptr getEnvWithoutPowerCloud() const;

    // 新增可视化函数
    void visualizeParameters(const std::unique_ptr<PointCloudPreprocessor>& preprocessor_ptr);


private:
    void loadParameters(ros::NodeHandle& nh);
    void manualClustering(const pcl::PointCloud<pcl::PointXYZI>::Ptr& cloud,
                          std::vector<pcl::PointIndices>& cluster_indices,
                          double tolerance, int min_size);

    // 新增辅助函数声明
    bool isPowerLinePoint(const pcl::PointCloud<pcl::PointXYZI>::Ptr& cloud,
        const pcl::PointCloud<pcl::Normal>::Ptr& normals,
        int index);
    // 新增函数：滤除较短的簇
    void filterShortClusters(const pcl::PointCloud<pcl::PointXYZI>::Ptr& cloud,
        std::vector<pcl::PointIndices>& cluster_indices,
        double min_length);

    // 新增函数：边沿过滤
    void filterEdgePoints(const pcl::PointCloud<pcl::PointXYZI>::Ptr& input_cloud,
                         const pcl::PointCloud<pcl::PointXYZI>::Ptr& original_cloud,
                         const pcl::PointCloud<pcl::Normal>::Ptr& normals);

    
    //pub
    ros::Publisher pub_linearity_;
    ros::Publisher pub_curvature_;
    ros::Publisher pub_variance_;

    // 在现有pub后面添加 建筑物边沿检测
    ros::Publisher pub_qualified_;
    ros::Publisher pub_unqualified_;
    // 参数 
    double linearity_threshold_;
    double curvature_threshold_;
    double planarity_threshold_;
    bool use_planarity_;
    double cluster_tolerance_;
    int min_cluster_size_;

    // 新增参数
    double variance_threshold_;
    double search_radius_;

    // 新增参数
    double min_cluster_length_;

   
    // 在现有参数后面添加  建筑物边沿检测
    double edge_check_radius_;        // 边沿检查半径
    int max_unqualified_neighbors_;   // 最大不合格邻居点数量

   




    // PCL 对象
    pcl::NormalEstimation<pcl::PointXYZI, pcl::Normal> normal_estimation_;
    pcl::search::KdTree<pcl::PointXYZI>::Ptr kdtree_;

    // 提取后的点云
    pcl::PointCloud<pcl::PointXYZI>::Ptr extracted_cloud_;
    pcl::PointCloud<pcl::PointXYZI>::Ptr env_without_powerline_cloud_;

    // ROS NodeHandle 用于发布
    ros::NodeHandle& nh_;
};

#endif // POWER_LINE_EXTRACTOR_H