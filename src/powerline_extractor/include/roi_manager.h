#ifndef ROI_MANAGER_H
#define ROI_MANAGER_H

#include <ros/ros.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/crop_box.h>
#include <visualization_msgs/MarkerArray.h>
#include <visualization_msgs/Marker.h>
#include <std_msgs/ColorRGBA.h>
#include <Eigen/Dense>
#include <vector>
#include <unordered_map>
#include <algorithm>
#include "power_line_probability_map.h"  // 为了使用LineROIInfo

// ROI长方体结构体
struct ROIBox {
    int primary_line_id;                    // 主电力线ID
    std::vector<int> merged_line_ids;       // 合并的所有电力线ID
    Eigen::Vector3f min_point;             // 长方体最小点
    Eigen::Vector3f max_point;             // 长方体最大点
    float confidence;                       // 置信度
    bool is_merged;                         // 是否合并结果
    ros::Time creation_time;                // 创建时间
    
    ROIBox() {
        primary_line_id = -1;
        confidence = 0.0f;
        is_merged = false;
        creation_time = ros::Time::now();
        min_point = Eigen::Vector3f::Zero();
        max_point = Eigen::Vector3f::Zero();
    }
    
    // 计算长方体体积
    float getVolume() const {
        Eigen::Vector3f size = max_point - min_point;
        return size.x() * size.y() * size.z();
    }
    
    // 计算长方体中心
    Eigen::Vector3f getCenter() const {
        return (min_point + max_point) * 0.5f;
    }
    
    // 计算长方体尺寸
    Eigen::Vector3f getSize() const {
        return max_point - min_point;
    }
    
    // 检查点是否在长方体内
    bool containsPoint(const Eigen::Vector3f& point) const {
        return (point.x() >= min_point.x() && point.x() <= max_point.x() &&
                point.y() >= min_point.y() && point.y() <= max_point.y() &&
                point.z() >= min_point.z() && point.z() <= max_point.z());
    }
};

// ROI结果结构体
struct ROIResult {
    ROIBox roi_box;                                          // ROI长方体信息
    pcl::PointCloud<pcl::PointXYZI>::Ptr cropped_cloud;     // 裁剪的点云
    int point_count;                                         // 点云数量
    float density;                                           // 点云密度 (点数/体积)
    
    ROIResult() {
        cropped_cloud = pcl::PointCloud<pcl::PointXYZI>::Ptr(new pcl::PointCloud<pcl::PointXYZI>);
        point_count = 0;
        density = 0.0f;
    }
    
    // 计算点云密度
    void calculateDensity() {
        float volume = roi_box.getVolume();
        density = (volume > 0.001f) ? static_cast<float>(point_count) / volume : 0.0f;
    }
};

// ROI统计信息
struct ROIStatistics {
    int total_roi_count;                    // 总ROI数量
    int merged_roi_count;                   // 合并后ROI数量
    int total_input_points;                 // 输入点云总数
    int total_output_points;                // 输出点云总数
    float compression_ratio;                // 压缩比
    float processing_time_ms;               // 处理时间（毫秒）
    
    ROIStatistics() {
        total_roi_count = 0;
        merged_roi_count = 0;
        total_input_points = 0;
        total_output_points = 0;
        compression_ratio = 0.0f;
        processing_time_ms = 0.0f;
    }
};

class ROIManager {
public:
    /**
     * @brief 构造函数：通过ROS节点句柄读取参数
     * @param nh ROS节点句柄
     */
    ROIManager(ros::NodeHandle& nh);
    
    /**
     * @brief 析构函数
     */
    ~ROIManager();
    
    // ==================== 主要接口函数 ====================
    
    /**
     * @brief 从完整点云中提取ROI区域
     * @param full_cloud 完整环境点云
     * @param line_rois 电力线ROI信息列表
     * @param roi_results 输出：按电力线ID组织的裁剪点云结果
     * @param total_cropped_cloud 输出：总体裁剪点云
     * @return 成功返回true，失败返回false
     */
    bool extractROI(const pcl::PointCloud<pcl::PointXYZI>::Ptr& full_cloud,
                   const std::vector<LineROIInfo>& line_rois,
                   std::vector<ROIResult>& roi_results,
                   pcl::PointCloud<pcl::PointXYZI>::Ptr& total_cropped_cloud);
    
    /**
     * @brief 为特定电力线提取ROI区域
     * @param full_cloud 完整环境点云
     * @param line_roi 单条电力线ROI信息
     * @return 裁剪后的点云
     */
    pcl::PointCloud<pcl::PointXYZI>::Ptr extractROIForLine(
        const pcl::PointCloud<pcl::PointXYZI>::Ptr& full_cloud,
        const LineROIInfo& line_roi);
    
    /**
     * @brief 获取ROI处理统计信息
     * @return ROI统计信息
     */
    ROIStatistics getROIStatistics() const { return last_statistics_; }
    
    /**
     * @brief 清空历史数据
     */
    void clearHistory();
    
    // ==================== 可视化接口 ====================
    
    /**
     * @brief 可视化ROI长方体
     * @param roi_boxes ROI长方体列表
     */
    void visualizeROIBoxes(const std::vector<ROIBox>& roi_boxes);
    
    /**
     * @brief 启用/禁用可视化
     * @param enable 是否启用可视化
     */
    void setVisualizationEnabled(bool enable) { enable_visualization_ = enable; }
    
    // ==================== 参数设置接口 ====================
    
    /**
     * @brief 设置扩展参数
     * @param expansion_factor 扩展系数
     * @param min_expansion 最小扩展距离
     */
    void setExpansionParameters(float expansion_factor, float min_expansion);
    
    /**
     * @brief 设置置信度阈值
     * @param threshold 置信度阈值
     */
    void setConfidenceThreshold(float threshold) { confidence_threshold_ = threshold; }

private:
    // ==================== ROS相关 ====================
    
    ros::NodeHandle nh_;                    // ROS节点句柄
    ros::Publisher roi_boxes_pub_;          // ROI长方体可视化发布器
    ros::Publisher roi_stats_pub_;          // ROI统计信息发布器
    
    // ==================== 参数变量 ====================
    
    // 扩展参数
    float expansion_factor_;                // 扩展系数 (相对于原尺寸)
    float min_expansion_;                   // 最小扩展距离 (m)
    float max_expansion_;                   // 最大扩展距离 (m)
    
    // 过滤参数
    float confidence_threshold_;            // 置信度阈值
    int min_points_per_roi_;               // 每个ROI最小点数
    float min_roi_volume_;                 // 最小ROI体积 (m³)
    float max_roi_volume_;                 // 最大ROI体积 (m³)
    
    // 合并参数
    bool enable_merge_;                     // 是否启用合并
    float merge_overlap_threshold_;         // 合并重叠阈值
    int max_roi_count_;                    // 最大ROI数量
    
    // 空间边界参数
    float x_min_, x_max_;                  // X轴边界
    float y_min_, y_max_;                  // Y轴边界
    float z_min_, z_max_;                  // Z轴边界
    
    // 可视化参数
    bool enable_visualization_;             // 是否启用可视化
    float visualization_duration_;          // 可视化持续时间
    std::string frame_id_;                 // 坐标系ID
    
    // 性能参数
    bool enable_parallel_processing_;      // 是否启用并行处理
    int max_processing_threads_;           // 最大处理线程数
    
    // ==================== 内部数据 ====================
    
    ROIStatistics last_statistics_;         // 上次处理统计信息
    
    // ==================== 内部函数 ====================
    
    // 参数读取函数
    void loadParameters();
    
    // ROI生成函数
    std::vector<ROIBox> generateInitialROIBoxes(const std::vector<LineROIInfo>& line_rois);
    ROIBox createROIBoxFromRegions(const LineROIInfo& line_roi);
    void expandROIBox(ROIBox& roi_box);
    float calculateExpansionFactor(const std::vector<Eigen::Vector3f>& regions);
    
    // 重叠检测与合并函数
    std::vector<ROIBox> mergeOverlappingBoxes(std::vector<ROIBox>& initial_boxes);
    bool isOverlapping(const ROIBox& box1, const ROIBox& box2);
    ROIBox mergeBoxes(const ROIBox& box1, const ROIBox& box2);
    float calculateOverlapRatio(const ROIBox& box1, const ROIBox& box2);
    
    // 点云裁剪函数
    pcl::PointCloud<pcl::PointXYZI>::Ptr cropPointCloud(
        const pcl::PointCloud<pcl::PointXYZI>::Ptr& full_cloud,
        const ROIBox& roi_box);
    
    pcl::PointCloud<pcl::PointXYZI>::Ptr combinePointClouds(
        const std::vector<pcl::PointCloud<pcl::PointXYZI>::Ptr>& clouds);
    
    // 验证与过滤函数
    bool validateROIBox(const ROIBox& roi_box);
    bool isInBounds(const Eigen::Vector3f& point);
    std::vector<ROIBox> filterROIBoxes(const std::vector<ROIBox>& roi_boxes);
    
    // 统计计算函数
    void calculateStatistics(const pcl::PointCloud<pcl::PointXYZI>::Ptr& input_cloud,
                           const std::vector<ROIResult>& roi_results,
                           float processing_time_ms);
    
    // 可视化辅助函数
    void publishROIBoxesVisualization(const std::vector<ROIBox>& roi_boxes);
    void publishROIStatistics();
    
    visualization_msgs::Marker createBoxMarker(const ROIBox& roi_box, int marker_id);
    std_msgs::ColorRGBA getROIBoxColor(const ROIBox& roi_box);
    
    // 数据验证函数
    bool validateInputData(const pcl::PointCloud<pcl::PointXYZI>::Ptr& full_cloud,
                          const std::vector<LineROIInfo>& line_rois);
};

#endif // ROI_MANAGER_H