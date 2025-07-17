#ifndef ADVANCED_OBSTACLE_ANALYZER_H_
#define ADVANCED_OBSTACLE_ANALYZER_H_

#include <ros/ros.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/PointIndices.h>
#include <vector>
#include <Eigen/Dense>
#include <visualization_msgs/MarkerArray.h>
#include <visualization_msgs/Marker.h>
#include <sensor_msgs/PointCloud2.h>
#include <pcl_conversions/pcl_conversions.h>
#include <Eigen/Geometry>
#include <iomanip>
#include "power_line_reconstruction.h"

// 障碍物包围盒数据结构
struct AdvancedObstacleBox {
    Eigen::Vector3f position;           // 中心
    Eigen::Vector3f size;               // 长宽高
    Eigen::Matrix3f rotation;           // 姿态（每列为主方向）
    float min_distance_to_powerline;    // 最近距离到电力线
    int warning_level;                  // 预警级别 (1,2,3)
    int closest_powerline_id;           // 最近的电力线ID
    bool is_end_obstacle;               // 是否为电力线端点障碍物
};

// 分层预警点云
struct LayeredWarningCloud {
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr merged_warning_cloud;  // 合并的分层预警点云(带颜色)
    pcl::PointCloud<pcl::PointXYZI>::Ptr end_obstacle_cloud;      // 端点障碍物点云(电线杆等)
    
    LayeredWarningCloud() {
        merged_warning_cloud.reset(new pcl::PointCloud<pcl::PointXYZRGB>);
        end_obstacle_cloud.reset(new pcl::PointCloud<pcl::PointXYZI>);
    }
};

class AdvancedObstacleAnalyzer {
public:
    AdvancedObstacleAnalyzer(ros::NodeHandle& nh, const std::string& frame_id = "map");
    
    // 主处理接口
    void analyzeObstacles(
        const std::vector<ReconstructedPowerLine>& power_lines,
        const pcl::PointCloud<pcl::PointXYZI>::Ptr& env_cloud,
        std::vector<AdvancedObstacleBox>& obstacle_results,
        LayeredWarningCloud& warning_cloud
    );
    
    // 可视化发布接口
    void publishObstacleMarkers(
        const std::vector<AdvancedObstacleBox>& obstacles,
        ros::Publisher& marker_pub,
        const std::string& frame_id = "map"
    );
    
    void publishWarningClouds(
        const LayeredWarningCloud& warning_cloud,
        ros::Publisher& cloud_pub_level1,
        ros::Publisher& cloud_pub_level2, 
        ros::Publisher& cloud_pub_level3,
        const std::string& frame_id = "map"
    );
    
    void publishPowerlineInfo(
        const std::vector<ReconstructedPowerLine>& power_lines,
        ros::Publisher& marker_pub,
        const std::string& frame_id = "map"
    );

    

private:
    ros::NodeHandle nh_;
    std::string frame_id_;

    //发布器
    ros::Publisher obstacle_markers_pub_;
    ros::Publisher powerline_info_pub_;
    ros::Publisher merged_warning_pub_;
    ros::Publisher end_obstacle_pub_;
    ros::Publisher warning_radius_pub_;  // 预警半径框架发布器
    
    // 聚类参数
    double cluster_tolerance_;           // 欧式聚类距离
    int cluster_min_size_;              // 最小聚类点数
    int cluster_max_size_;              // 最大聚类点数
    
    // 电力线清理参数
    double powerline_clean_radius_;     // 清理电力线的半径
    double powerline_merge_distance_;   // 电力线合并距离阈值(2m)
    
    // 分层预警参数
    double warning_level1_radius_;      // 第一层预警半径(3m)
    double warning_level2_radius_;      // 第二层预警半径(8m)
    
    // 端点过滤参数
    double end_filter_distance_;        // 端点过滤距离
    double end_filter_ratio_;           // 端点过滤比例
    
    // 可视化参数
    double obstacle_box_alpha_;         // 障碍物盒子透明度
    bool show_powerline_distance_;      // 是否显示电力线距离
    bool show_obstacle_distance_;       // 是否显示障碍物距离
    // 在现有参数后添加
    int min_warning_cloud_size_;        // 预警点云最小点数
    double warning_corridor_width_;     // 预警通道宽度（无限长矩形的宽度）

    
    // 内部状态
    size_t last_num_obstacles_;         // 上次障碍物数量
    size_t last_num_powerlines_;        // 上次电力线数量
    
    // 核心处理函数
    void cleanEnvironmentCloud(
        const std::vector<ReconstructedPowerLine>& power_lines,
        const pcl::PointCloud<pcl::PointXYZI>::Ptr& env_cloud,
        pcl::PointCloud<pcl::PointXYZI>::Ptr& cleaned_cloud
    );
    
    void euclideanClustering(
        const pcl::PointCloud<pcl::PointXYZI>::Ptr& input_cloud,
        std::vector<pcl::PointIndices>& cluster_indices
    );
    
    void computeAdvancedOBB(
        const pcl::PointCloud<pcl::PointXYZI>::Ptr& cluster_cloud,
        const std::vector<ReconstructedPowerLine>& power_lines,
        AdvancedObstacleBox& obb
    );
    
    bool isEndObstacle(
        const pcl::PointCloud<pcl::PointXYZI>::Ptr& cluster_cloud,
        const std::vector<ReconstructedPowerLine>& power_lines
    );
    
    void computeLayeredWarning(
        const pcl::PointCloud<pcl::PointXYZI>::Ptr& cleaned_cloud,
        const std::vector<ReconstructedPowerLine>& power_lines,
        LayeredWarningCloud& warning_cloud
    );
    
    float computeMinDistanceToPowerlines(
        const pcl::PointCloud<pcl::PointXYZI>::Ptr& cluster_cloud,
        const std::vector<ReconstructedPowerLine>& power_lines,
        int& closest_line_id
    );
    
    float pointToSplineDistance(
        const Eigen::Vector3f& point,
        const std::vector<Eigen::Vector3f>& spline_points
    );
    
    std::vector<std::vector<ReconstructedPowerLine>> groupPowerlinesByDistance(
        const std::vector<ReconstructedPowerLine>& power_lines
    );
    //可视化发布
    void publishAllVisualization(
        const std::vector<ReconstructedPowerLine>& power_lines,
        const std::vector<AdvancedObstacleBox>& obstacles,
        const LayeredWarningCloud& warning_cloud);
    
    // 辅助函数
    void loadParameters();
    Eigen::Vector3f getColorByWarningLevel(int level);
    std::string getWarningLevelText(int level);

    void publishWarningRadius(
        const std::vector<ReconstructedPowerLine>& power_lines,
        ros::Publisher& marker_pub,
        const std::string& frame_id = "map"
    );
    
    bool isNearPowerlineEnd(
        const Eigen::Vector3f& point,
        const std::vector<ReconstructedPowerLine>& power_lines,
        double end_threshold = 10.0  // 端点阈值距离
    );
    // 在现有私有函数后添加
    bool isPointInPowerlineCorridor(
        const Eigen::Vector3f& point,
        const std::vector<ReconstructedPowerLine>& power_lines,
        float& min_distance_to_axis
    );
    
    float pointToLineSegmentDistance(
        const Eigen::Vector3f& point,
        const Eigen::Vector3f& line_start,
        const Eigen::Vector3f& line_end
    );
};

#endif // ADVANCED_OBSTACLE_ANALYZER_H_