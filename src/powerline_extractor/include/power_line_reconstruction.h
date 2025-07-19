#ifndef POWER_LINE_RECONSTRUCTION_H
#define POWER_LINE_RECONSTRUCTION_H

#include <ros/ros.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/common/pca.h>
#include <pcl/common/common.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/search/kdtree.h>
#include <pcl/filters/extract_indices.h>
#include <pcl_conversions/pcl_conversions.h>
#include <sensor_msgs/PointCloud2.h>
#include <visualization_msgs/MarkerArray.h>
#include <visualization_msgs/Marker.h>
#include <std_msgs/ColorRGBA.h>
#include <Eigen/Dense>
#include <vector>
#include <algorithm>

// 电力线片段结构体
struct PowerLineSegment {
    pcl::PointCloud<pcl::PointXYZI>::Ptr points;
    Eigen::Vector3f center;
    Eigen::Vector3f start_point;
    Eigen::Vector3f end_point;
    Eigen::Vector3f local_direction;   // 中心到端点方向
    Eigen::Vector3f overall_direction; // 整体主方向（考虑弯曲）
    int cluster_id;   //聚类ID
    double length;
    
    PowerLineSegment() : points(new pcl::PointCloud<pcl::PointXYZI>), cluster_id(-1), length(0.0) {}
};

// 重构后的电力线结构体
struct ReconstructedPowerLine {
    std::vector<int> segment_indices;  // 包含的片段索引
    pcl::PointCloud<pcl::PointXYZI>::Ptr points;
    std::vector<Eigen::Vector3f> fitted_curve_points;  // 拟合的曲线点
    double total_length;
    int line_id;
    Eigen::Vector3f main_direction;  // 整条线的主方向
    
    ReconstructedPowerLine() : points(new pcl::PointCloud<pcl::PointXYZI>), total_length(0.0), line_id(-1) {}
};

class PowerLineReconstructor {
public:
    // 构造函数：通过ROS节点句柄读取参数
    PowerLineReconstructor(ros::NodeHandle& nh);

    // 主要处理函数
    void reconstructPowerLines(const pcl::PointCloud<pcl::PointXYZI>::Ptr& input_cloud,
                              pcl::PointCloud<pcl::PointXYZI>::Ptr& output_cloud,
                              std::vector<ReconstructedPowerLine>& power_lines);

private:
    // ROS相关
    ros::NodeHandle nh_;
    ros::Publisher original_cloud_pub_;           // 原始点云发布器
    ros::Publisher separated_segments_pub_;       // 分离片段发布器
    ros::Publisher reconstructed_lines_pub_;      // 重构电力线发布器
    ros::Publisher curve_markers_pub_;            // 拟合曲线标记发布器
    ros::Publisher segment_info_pub_;             // 片段信息标记发布器
    ros::Publisher segment_endpoints_pub_;        // 片段端点发布器
    ros::Publisher segment_endpoint_lines_pub_;   // 片段端点连线发布器

    // 参数变量
    double separation_distance_;        // 片段分离距离阈值
    double min_segment_points_;        // 片段最小点数
    double max_connection_distance_;   // 连接最大距离
    double max_perpendicular_distance_; // 垂直方向最大距离
    double direction_angle_threshold_; // 方向角度阈值
    double min_line_length_;          // 最小线长度
    double spline_resolution_;        // 样条拟合分辨率
    double parallel_threshold_;       // 平行线判断阈值
    double connection_weight_distance_; // 连接判断距离权重
    double connection_weight_angle_;   // 连接判断角度权重

    
    // 可视化参数
    bool enable_visualization_;       // 是否启用最终结果可视化
    bool enable_separation_visualization_; // 是否启用分离可视化
    double visualization_duration_;   // 可视化持续时间（秒）
    std::string frame_id_;           // 坐标系ID

    // 主要处理步骤
    void separateSegments(const pcl::PointCloud<pcl::PointXYZI>::Ptr& input_cloud,
                         std::vector<PowerLineSegment>& segments);
    
    void createSegmentStructures(std::vector<PowerLineSegment>& segments);
    
    void judgeConnectivity(const std::vector<PowerLineSegment>& segments,
                          std::vector<std::vector<int>>& connected_groups);
    
    void reconstructLines(const std::vector<PowerLineSegment>& segments,
                         const std::vector<std::vector<int>>& connected_groups,
                         std::vector<ReconstructedPowerLine>& power_lines);
    
    void fitSplineCurve(ReconstructedPowerLine& power_line);
    
    void visualizeResults(const std::vector<PowerLineSegment>& segments,
                         const std::vector<ReconstructedPowerLine>& power_lines);
    
    void visualizeSeparationResults(const pcl::PointCloud<pcl::PointXYZI>::Ptr& input_cloud,
                                   const std::vector<PowerLineSegment>& segments);

    // 辅助函数
    void computeSegmentProperties(PowerLineSegment& segment);
    
    double calculateConnectionScore(const PowerLineSegment& seg1, 
                                  const PowerLineSegment& seg2);
    
    bool isConnectable(const PowerLineSegment& seg1, 
                      const PowerLineSegment& seg2);
    
    void mergeSegmentPoints(const std::vector<PowerLineSegment>& segments,
                           const std::vector<int>& indices,
                           pcl::PointCloud<pcl::PointXYZI>::Ptr& merged_cloud);
    
    // ROS可视化辅助函数
    void publishColoredPointCloud(const pcl::PointCloud<pcl::PointXYZI>::Ptr& cloud,
                                 const std::vector<Eigen::Vector3f>& colors,
                                 const std::vector<int>& color_indices,
                                 const ros::Publisher& publisher,
                                 const std::string& frame_id);
    
    void publishCurveMarkers(const std::vector<ReconstructedPowerLine>& power_lines);
    
    void publishSegmentInfoMarkers(const std::vector<PowerLineSegment>& segments);
    
    std::vector<Eigen::Vector3f> generateColorPalette(int num_colors);
    
    std_msgs::ColorRGBA eigenToColorRGBA(const Eigen::Vector3f& color, float alpha = 1.0f);
    
    // DFS用于连通性分析
    void dfsConnectivity(int segment_idx,
                        const std::vector<PowerLineSegment>& segments,
                        const std::vector<std::vector<bool>>& connectivity_matrix,
                        std::vector<bool>& visited,
                        std::vector<int>& current_group);
    // 计算两个点之间的欧式距离
    float pointDistance(const Eigen::Vector3f &p1, const Eigen::Vector3f &p2) {
        return (p1 - p2).norm();
    }
    //发布端点的函数
    void publishSegmentEndpoints(const std::vector<PowerLineSegment>& segments);
};

#endif // POWER_LINE_RECONSTRUCTION_H