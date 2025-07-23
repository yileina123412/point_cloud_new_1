#ifndef ENHANCED_POWER_LINE_TRACKER_H
#define ENHANCED_POWER_LINE_TRACKER_H

#include <ros/ros.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <Eigen/Dense>
#include <vector>
#include <map>
#include <memory>
#include <visualization_msgs/MarkerArray.h>
#include <visualization_msgs/Marker.h>
#include <std_msgs/ColorRGBA.h>
#include "power_line_reconstruction.h"
#include "power_line_probability_map.h"

// 轨迹状态枚举
enum LineTrackStatus {
    TRACK_INITIALIZING = 0,    // 初始化中
    TRACK_STABLE = 1,          // 稳定跟踪
    TRACK_PARTIAL = 2,         // 部分观测（需要补全）
    TRACK_PREDICTED = 3,       // 纯预测（无观测）
    TRACK_LOST = 4             // 轨迹丢失
};

// 检测片段信息
struct DetectionSegment {
    int detection_index;                           // 原始检测索引
    std::vector<Eigen::Vector3f> segment_points;   // 片段样条点
    Eigen::Vector3f start_point;                   // 片段起点
    Eigen::Vector3f end_point;                     // 片段终点
    float segment_length;                          // 片段长度
    float confidence;                              // 片段置信度
    
    DetectionSegment() : detection_index(-1), segment_length(0.0f), confidence(0.0f) {}
};

// 融合后的观测信息
struct FusedObservation {
    std::vector<DetectionSegment> segments;        // 组成片段
    std::vector<Eigen::Vector3f> observed_points; // 观测到的样条点
    std::vector<bool> point_validity;             // 每个点是否有效观测
    float total_observed_length;                   // 总观测长度
    float completeness_ratio;                      // 完整度比例 [0,1]
    
    FusedObservation() : total_observed_length(0.0f), completeness_ratio(0.0f) {}
};

// 单条电力线轨迹器
class LineTracker {
public:
    int line_id_;                                  // 对应概率地图的line_id
    LineTrackStatus status_;                       // 轨迹状态
    
    // 卡尔曼滤波状态 (控制点 + 速度 + 形状参数)
    static const int NUM_CONTROL_POINTS = 10;     // 控制点数量
    static const int STATE_DIM = NUM_CONTROL_POINTS * 3 + 6; // 30个位置 + 6个形状参数
    
    Eigen::VectorXf state_vector_;                 // 状态向量 [36维]
    Eigen::MatrixXf covariance_matrix_;            // 协方差矩阵 [36x36]
    Eigen::MatrixXf process_noise_;                // 过程噪声矩阵
    
    // 轨迹属性
    float confidence_;                             // 轨迹置信度 [0,1]
    int consecutive_updates_;                      // 连续更新次数
    int consecutive_misses_;                       // 连续未匹配次数
    float estimated_total_length_;                 // 估计总长度
    
    // 时间信息
    ros::Time creation_time_;                      // 创建时间
    ros::Time last_update_time_;                   // 最后更新时间
    
    // 历史信息
    std::vector<FusedObservation> observation_history_; // 观测历史
    
public:
    LineTracker(int line_id, const PowerLineProbabilityMap& prob_map);
    
    // 预测步骤
    void predict(float dt);
    
    // 更新步骤（融合多个片段）
    void update(const FusedObservation& observation, const PowerLineProbabilityMap& prob_map);
    
    // 生成完整电力线（包含补全）
    ReconstructedPowerLine generateCompleteLine(const PowerLineProbabilityMap& prob_map) const;
    
    // 状态管理
    void markAsLost();
    bool shouldBeRemoved() const;
    bool isStable() const;

    std::vector<Eigen::Vector3f> getControlPointsFromState() const;
    
private:
    void markDetectedParts(ReconstructedPowerLine& complete_line) const {
        if (observation_history_.empty() || complete_line.fitted_curve_points.empty()) {
            return;
        }
        // 获取最近的观测信息
        const auto& recent_obs = observation_history_.back();
        
        // 为每个完整曲线点检查是否接近观测点
        for (size_t i = 0; i < complete_line.fitted_curve_points.size(); ++i) {
            const auto& curve_point = complete_line.fitted_curve_points[i];
            bool is_detected = false;
            
            // 检查是否接近任何观测点
            for (const auto& obs_point : recent_obs.observed_points) {
                float distance = (curve_point - obs_point).norm();
                if (distance < 0.5f) { // 0.5米阈值
                    is_detected = true;
                    break;
                }
            }
            
            complete_line.point_types[i] = is_detected ? 1 : 0; // 1=检测, 0=补全
        }
    }
    void initializeFromProbabilityMap(const PowerLineProbabilityMap& prob_map);
    void updateStateFromObservation(const FusedObservation& observation);
    std::vector<Eigen::Vector3f> interpolateCompleteCurve(const PowerLineProbabilityMap& prob_map) const;
    
    void setControlPointsToState(const std::vector<Eigen::Vector3f>& control_points);
    bool isPointInProbabilityMap(const Eigen::Vector3f& point, const PowerLineProbabilityMap& prob_map) const;
};

// 主跟踪器类
class EnhancedPowerLineTracker {
public:
    // 构造函数
    EnhancedPowerLineTracker(ros::NodeHandle& nh);
    
    // 析构函数
    ~EnhancedPowerLineTracker();
    
    // ==================== 主要接口 ====================
    
    /**
     * @brief 初始化跟踪器
     * @param prob_map 概率地图引用
     * @return 成功返回true
     */
    bool initializeTracker(const PowerLineProbabilityMap& prob_map);

    /**
     * @brief 初始化跟踪器（结合检测结果）
     * @param prob_map 概率地图引用
     * @param initial_detections 初始检测结果（可选，用于更好的初始化）
     * @return 成功返回true
     */
    bool initializeTracker(PowerLineProbabilityMap& prob_map,
                        const std::vector<ReconstructedPowerLine>& initial_detections);
    
    /**
     * @brief 更新跟踪器（主要接口）
     * @param current_detections 当前帧检测结果
     * @param prob_map 概率地图引用
     * @return 完整的电力线列表（检测+补全）
     */
    std::vector<ReconstructedPowerLine> updateTracker(
        const std::vector<ReconstructedPowerLine>& current_detections,
        PowerLineProbabilityMap& prob_map);
    
    /**
     * @brief 获取轨迹统计信息
     */
    void getTrackerStatistics(int& total_tracks, int& stable_tracks, 
                             int& partial_tracks, float& avg_confidence) const;
    
    /**
     * @brief 清空所有轨迹
     */
    void clearAllTracks();
    
private:
    // ==================== 核心数据 ====================
    
    ros::NodeHandle nh_;                           // ROS节点句柄
    std::map<int, std::unique_ptr<LineTracker>> line_trackers_; // 按line_id管理的轨迹
    ros::Time last_update_time_;                   // 上次更新时间
    bool is_initialized_;                          // 是否已初始化
    
    // ==================== ROS发布器 ====================
    
    ros::Publisher track_visualization_pub_;       // 轨迹可视化发布器
    ros::Publisher detected_parts_pub_;            // 检测部分发布器  
    ros::Publisher completed_lines_pub_;           // 补全线段发布器
    ros::Publisher track_statistics_pub_;          // 统计信息发布器
    
    // ==================== 参数变量 ====================
    
    // 卡尔曼滤波参数
    float process_noise_position_;                 // 位置过程噪声
    float process_noise_velocity_;                 // 速度过程噪声
    float process_noise_shape_;                    // 形状过程噪声
    float measurement_noise_;                      // 测量噪声
    
    // 片段融合参数
    float segment_merge_distance_;                 // 片段合并距离阈值
    float segment_overlap_threshold_;              // 片段重叠阈值
    float min_segment_length_;                     // 最小片段长度
    float max_gap_distance_;                       // 最大间隙距离
    
    // 补全参数
    float completion_confidence_threshold_;        // 补全置信度阈值
    float max_extrapolation_distance_;            // 最大外推距离
    float interpolation_resolution_;               // 插值分辨率
    bool enable_completion_;                       // 是否启用补全
    
    // 轨迹管理参数
    int max_consecutive_misses_;                   // 最大连续未匹配次数
    int min_updates_for_stable_;                   // 稳定轨迹最小更新次数
    float min_confidence_threshold_;               // 最小置信度阈值
    int max_track_count_;                          // 最大轨迹数量
    float track_timeout_duration_;                 // 轨迹超时时间
    
    // 可视化参数
    bool enable_visualization_;                    // 是否启用可视化
    float visualization_duration_;                 // 可视化持续时间
    std::string frame_id_;                         // 坐标系ID
    
    // ==================== 内部函数 ====================
        // 初始化辅助函数
    bool initializeFromDetections(const std::vector<ReconstructedPowerLine>& detections,
                                 PowerLineProbabilityMap& prob_map);
    // 参数读取
    void loadParameters();
    
    // 核心处理流程
    std::map<int, std::vector<DetectionSegment>> groupDetectionsByLineID(
        const std::vector<ReconstructedPowerLine>& detections,
        PowerLineProbabilityMap& prob_map);
    
    FusedObservation fuseSegments(const std::vector<DetectionSegment>& segments);
    
    void predictAllTracks(float dt);
    void updateTracksWithObservations(const std::map<int, FusedObservation>& observations,
                                     const PowerLineProbabilityMap& prob_map);
    void createNewTracks(const std::map<int, FusedObservation>& observations,
                        const PowerLineProbabilityMap& prob_map);
    void removeInactiveTracks();
    
    std::vector<ReconstructedPowerLine> generateOutputLines(const PowerLineProbabilityMap& prob_map);
    
    // 辅助函数
    DetectionSegment createSegmentFromDetection(const ReconstructedPowerLine& detection, int index);
    std::vector<Eigen::Vector3f> sortPointsSpatially(const std::vector<Eigen::Vector3f>& points);
    std::vector<Eigen::Vector3f> fillGapsInPoints(const std::vector<Eigen::Vector3f>& points,
                                                  const std::vector<bool>& validity);
    bool areSegmentsConnectable(const DetectionSegment& seg1, const DetectionSegment& seg2);
    float calculateSegmentConfidence(const DetectionSegment& segment, const PowerLineProbabilityMap& prob_map);
    
    // 可视化函数
    void publishTrackVisualization();
    void publishCompletedLinesVisualization(const std::vector<ReconstructedPowerLine>& completed_lines);
    void publishTrackStatistics();
    std_msgs::ColorRGBA getTrackStatusColor(LineTrackStatus status) const;
    visualization_msgs::Marker createTrackMarker(const LineTracker& tracker, int marker_id) const;
};

#endif // ENHANCED_POWER_LINE_TRACKER_H