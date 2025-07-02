#ifndef OBSTACLE_ANALYZER_H_
#define OBSTACLE_ANALYZER_H_

#include <ros/ros.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/PointIndices.h>
#include <vector>
#include <Eigen/Dense>
#include <visualization_msgs/MarkerArray.h>
#include <visualization_msgs/Marker.h>
#include <Eigen/Geometry> // Eigen::Quaternionf
#include <iomanip>

// 障碍物包围盒数据结构
struct OrientedBoundingBox {
    Eigen::Vector3f position;           // 中心
    Eigen::Vector3f size;               // 长宽高
    Eigen::Matrix3f rotation;           // 姿态（每列为主方向）
    float min_distance_to_powerline;    // 最近距离
};

class ObstacleAnalyzer {
public:
    ObstacleAnalyzer(ros::NodeHandle& nh);

    // 主处理接口
    void analyzeObstacles(
        const pcl::PointCloud<pcl::PointXYZI>::Ptr& raw_cloud,
        const pcl::PointCloud<pcl::PointXYZI>::Ptr& powerline_cloud,
        std::vector<OrientedBoundingBox>& obb_results
    );
    void publishObbMarkers(
        const std::vector<OrientedBoundingBox>& obb_vec,
        ros::Publisher& marker_pub,
        const std::string& frame_id = "map"
    );
    //显示电力线的距离
    void publishPowerlineDistanceMarkers(
        const pcl::PointCloud<pcl::PointXYZI>::Ptr& powerline_cloud,
        ros::Publisher& marker_pub,
        const std::string& frame_id = "map"
        );

private:
    // 参数
    double cluster_tolerance_;     // 欧式聚类距离(米)
    int cluster_min_size_;         // 聚类最小点数
    int cluster_max_size_;         // 聚类最大点数
    double distance_search_step_;  // 距离KD树搜索步长（冗余，暴力法可不用）
    size_t g_last_num_obstacles = 0;  // 建议static全局变量，或者作为节点成员

    ros::NodeHandle nh_;

    

    void segmentObstacles(
        const pcl::PointCloud<pcl::PointXYZI>::Ptr& raw_cloud,
        const pcl::PointCloud<pcl::PointXYZI>::Ptr& powerline_cloud,
        pcl::PointCloud<pcl::PointXYZI>::Ptr& obstacles_cloud);

    void euclideanClustering(
        const pcl::PointCloud<pcl::PointXYZI>::Ptr& obstacles_cloud,
        std::vector<pcl::PointIndices>& cluster_indices);

    void computeOrientedBoundingBox(
        const pcl::PointCloud<pcl::PointXYZI>::Ptr& cluster_cloud,
        OrientedBoundingBox& obb);

    float computeMinDistance(
        const pcl::PointCloud<pcl::PointXYZI>::Ptr& cluster_cloud,
        const pcl::PointCloud<pcl::PointXYZI>::Ptr& powerline_cloud);



};



#endif
