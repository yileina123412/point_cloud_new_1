#ifndef BUILDING_EDGE_FILTER_H
#define BUILDING_EDGE_FILTER_H

#include <ros/ros.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/search/kdtree.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/common/common.h>
#include <pcl/common/pca.h>
#include <Eigen/Dense>
#include <vector>
#include <algorithm>
#include <std_msgs/ColorRGBA.h>
#include "power_line_reconstruction.h"


// // 引入电力线重构模块的结构体
// struct ReconstructedPowerLine {
//     std::vector<int> segment_indices;
//     pcl::PointCloud<pcl::PointXYZI>::Ptr points;
//     std::vector<Eigen::Vector3f> fitted_curve_points;
//     double total_length;
//     int line_id;
//     Eigen::Vector3f main_direction;
    
//     ReconstructedPowerLine() : points(new pcl::PointCloud<pcl::PointXYZI>), total_length(0.0), line_id(-1) {}
// };

class BuildingEdgeFilter {
public:
    // 构造函数：通过ROS节点句柄读取参数
    BuildingEdgeFilter(ros::NodeHandle& nh);

    // 主要过滤函数
    bool filterBuildingEdges(
        const pcl::PointCloud<pcl::PointXYZI>::Ptr& environment_cloud,
        std::vector<ReconstructedPowerLine>& power_lines,
        pcl::PointCloud<pcl::PointXYZI>::Ptr& output_cloud
    );

private:
    // ROS相关
    ros::NodeHandle nh_;
    ros::Publisher filtered_lines_pub_;        // 过滤后的电力线发布器
    ros::Publisher filtered_info_pub_;         // 过滤信息标记发布器

    // 添加这一行：
    ros::Publisher cylinder_debug_pub_;        // 圆柱区域调试点云发布器

    // 参数变量
    float cylinder_radius_;           // 圆柱半径
    float end_exclusion_distance_;    // 头尾排除距离
    float min_plane_area_;           // 最小平面面积阈值
    float min_plane_density_;        // 最小平面密度阈值
    float plane_distance_threshold_; // 平面拟合距离阈值
    int min_plane_points_;           // 平面最小点数
    float coordinate_tolerance_;     // 坐标匹配容差
    bool enable_debug_output_;       // 是否启用调试输出
    bool enable_visualization_;      // 是否启用可视化
    std::string frame_id_;          // 坐标系ID
    double visualization_duration_;  // 可视化持续时间

    float powerline_expansion_radius_;   // 电力线扩展搜索半径

    // 创建带颜色的点云用于可视化
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr debug_cloud;

    // 核心处理函数
    bool checkSinglePowerLine(
        const pcl::PointCloud<pcl::PointXYZI>::Ptr& environment_cloud,
        const ReconstructedPowerLine& power_line
    );

    // 创建圆柱形区域点云
    pcl::PointCloud<pcl::PointXYZI>::Ptr createCylinderRegion(
        const pcl::PointCloud<pcl::PointXYZI>::Ptr& environment_cloud,
        const ReconstructedPowerLine& power_line
    );

    // 移除电力线自身的点
    void removePowerLinePoints(
        pcl::PointCloud<pcl::PointXYZI>::Ptr& cylinder_cloud,
        const pcl::PointCloud<pcl::PointXYZI>::Ptr& powerline_cloud
    );

    // 平面检测和面积计算
    bool detectLargePlane(
        const pcl::PointCloud<pcl::PointXYZI>::Ptr& neighborhood_cloud,
        float& plane_area,
        float& plane_density
    );

    // 计算点到轴线的距离
    float calculateDistanceToAxis(
        const pcl::PointXYZI& point,
        const ReconstructedPowerLine& power_line
    );

    // 检查点是否在电力线的中间部分
    bool isInMiddleSection(
        const pcl::PointXYZI& point,
        const ReconstructedPowerLine& power_line
    );

    // 计算平面面积
    float calculatePlaneArea(
        const pcl::PointCloud<pcl::PointXYZI>::Ptr& plane_points,
        const Eigen::Vector4f& plane_coefficients
    );

    // 辅助函数：判断两个点是否相近
    bool arePointsClose(
        const pcl::PointXYZI& p1,
        const pcl::PointXYZI& p2,
        float tolerance
    );

    // 可视化函数
    void visualizeFilterResults(
        const std::vector<ReconstructedPowerLine>& valid_lines,
        const std::vector<ReconstructedPowerLine>& filtered_lines
    );

    // 生成颜色调色板
    std::vector<Eigen::Vector3f> generateColorPalette(int num_colors);

    // 颜色转换函数
    std_msgs::ColorRGBA eigenToColorRGBA(const Eigen::Vector3f& color, float alpha = 1.0f);

    // 合并点云
    void mergePointClouds(
        const std::vector<ReconstructedPowerLine>& power_lines,
        pcl::PointCloud<pcl::PointXYZI>::Ptr& output_cloud
    );

    // 验证电力线完整性
    void validatePowerLinesIntegrity(
        const std::vector<ReconstructedPowerLine>& power_lines,
        const std::string& stage_name
    );

    // 扩展电力线点云（在其他辅助函数附近添加）
    pcl::PointCloud<pcl::PointXYZI>::Ptr expandPowerLinePoints(
        const pcl::PointCloud<pcl::PointXYZI>::Ptr& environment_cloud,
        const ReconstructedPowerLine& power_line,
        float search_radius = 0.05f
    );
};

#endif // BUILDING_EDGE_FILTER_H