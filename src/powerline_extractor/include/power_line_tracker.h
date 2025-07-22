#ifndef POWER_LINE_TRACKER_H
#define POWER_LINE_TRACKER_H

#include <ros/ros.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <Eigen/Dense>
#include <vector>
#include <unordered_map>
#include <memory>
#include "power_line_reconstruction.h"  // 使用ReconstructedPowerLine结构体
#include "power_line_probability_map.h" // 概率地图接口

// 轨迹状态枚举
enum TrackStatus {
    TRACK_NEW = 0,           // 新建轨迹
    TRACK_ACTIVE = 1,        // 活跃跟踪
    TRACK_PREDICTED = 2,     // 预测状态（当前帧无检测）
    TRACK_LOST = 3,          // 丢失轨迹
    TRACK_CONFIRMED = 4      // 确认稳定轨迹
};

// 电力线状态向量
struct PowerLineState {
    // 几何特征
    Eigen::Vector3f start_point;      // 起点
    Eigen::Vector3f end_point;        // 终点
    Eigen::Vector3f main_direction;   // 主方向
    float total_length;               // 总长度
    float average_curvature;          // 平均曲率
    
    // 关键控制点（用于形状匹配）
    std::vector<Eigen::Vector3f> control_points;  // 5个控制点
    
    // 运动参数（主要用于小幅预测）
    Eigen::Vector3f position_drift;   // 位置漂移速度   最后没有实际用到
    float length_change_rate;         // 长度变化率
    
    PowerLineState() {
        total_length = 0.0f;
        average_curvature = 0.0f;
        length_change_rate = 0.0f;
        position_drift = Eigen::Vector3f::Zero();
        control_points.resize(5);
    }
};

// 单条电力线轨迹
class TrackedPowerLine {
public:
    int track_id_;                    // 轨迹ID
    TrackStatus status_;              // 轨迹状态
    PowerLineState state_;            // 当前状态
    PowerLineState predicted_state_;  // 预测状态
    
    float confidence_;                // 轨迹置信度 [0,1]
    int consecutive_misses_;          // 连续未匹配次数
    int consecutive_hits_;            // 连续匹配次数
    int total_updates_;               // 总更新次数
    
    ros::Time creation_time_;         // 创建时间
    ros::Time last_update_time_;      // 最后更新时间
    
    // 卡尔曼滤波相关
    Eigen::VectorXf state_vector_;    // 状态向量 [start(3) + end(3) + direction(3) + length(1) + curvature(1) = 11维]
    Eigen::MatrixXf covariance_;      // 协方差矩阵 11x11
    Eigen::MatrixXf process_noise_;   // 过程噪声
    
    // 历史信息（用于补全）
    std::vector<ReconstructedPowerLine> history_;  // 最近N帧的检测历史

    // 添加分线概率地图
    std::unordered_map<VoxelKey, PowerLineVoxel> line_voxel_map_;
    
public:
    TrackedPowerLine(int id, const ReconstructedPowerLine& initial_detection);
    
    // 预测下一帧状态
    void predict();
    
    // 更新状态（卡尔曼滤波更新）
    void update(const ReconstructedPowerLine& detection);
    
    // 基于历史信息生成补全的电力线
    ReconstructedPowerLine generateCompletedLine() const;
    
    // 计算与检测结果的相似度
    float calculateSimilarity(const ReconstructedPowerLine& detection) const;

    Eigen::Vector3f interpolateControlPoint(float t) const;



    
    // 状态管理
    void markAsLost();
    void markAsConfirmed();
    bool isStable() const;
    bool shouldBeRemoved() const;
    
private:
    void initializeKalmanFilter(const ReconstructedPowerLine& initial_detection);
    void stateToVector();
    void vectorToState();
    
    void updateHistory(const ReconstructedPowerLine& detection);
    std::vector<Eigen::Vector3f> extractControlPoints(const ReconstructedPowerLine& line) const;
};

// 数据关联结果
struct AssociationResult {
    std::vector<std::pair<int, int>> matched_pairs;  // (track_id, detection_id)
    std::vector<int> unmatched_tracks;               // 未匹配的轨迹
    std::vector<int> unmatched_detections;           // 未匹配的检测
};

// 扩展的数据关联结果（支持1对多匹配）
struct EnhancedAssociationResult {
    std::vector<std::pair<int, std::vector<int>>> track_to_fragments;  // 轨迹ID -> 多个检测片段索引
    std::vector<int> unmatched_tracks;                                 // 未匹配的轨迹
    std::vector<int> unmatched_detections;                            // 未匹配的检测
    std::vector<std::vector<int>> fragment_clusters;                  // 预聚类的片段组
};

// 片段连续性评估结果
struct ContinuityScore {
    float spatial_continuity;    // 空间连续性 [0,1]
    float direction_continuity;  // 方向连续性 [0,1]
    float overall_score;         // 综合评分 [0,1]
    
    ContinuityScore() : spatial_continuity(0.0f), direction_continuity(0.0f), overall_score(0.0f) {}
};

// 主跟踪器类
class PowerLineTracker {
public:
    // 构造函数
    PowerLineTracker(ros::NodeHandle& nh);
    
    // 析构函数
    ~PowerLineTracker();
    
    // ==================== 主要接口 ====================
    
    /**
     * @brief 初始化跟踪器（第一帧后调用）
     * @param initial_detections 第一帧检测到的电力线
     * @return 成功返回true
     */
    bool initializeTracker(const std::vector<ReconstructedPowerLine>& initial_detections);
    
    /**
     * @brief 更新跟踪器（第二帧及后续调用）
     * @param current_detections 当前帧检测到的电力线
     * @param prob_map 概率地图引用（用于验证）
     * @return 更新后的完整电力线列表（检测+补全）
     */
    std::vector<ReconstructedPowerLine> updateTracker(
        const std::vector<ReconstructedPowerLine>& current_detections,
        const PowerLineProbabilityMap& prob_map);
    
    /**
     * @brief 获取当前活跃轨迹数量
     */
    int getActiveTrackCount() const;
    
    /**
     * @brief 获取所有轨迹状态
     */
    std::vector<TrackedPowerLine> getActiveTracksStatus() const;
    
    /**
     * @brief 清空所有轨迹
     */
    void clearAllTracks();
    
    /**
     * @brief 设置概率地图引用（可选，用于更好的验证）
     */
    void setProbabilityMapReference(std::shared_ptr<PowerLineProbabilityMap> prob_map);
    
private:
    // ==================== 核心数据 ====================
    
    ros::NodeHandle nh_;                              // ROS节点句柄
    std::vector<TrackedPowerLine> active_tracks_;     // 活跃轨迹列表
    int next_track_id_;                               // 下一个轨迹ID
    bool is_initialized_;                             // 是否已初始化
    
    // 概率地图引用（可选）
    std::shared_ptr<PowerLineProbabilityMap> prob_map_ptr_;
    
    // ==================== 参数变量 ====================
    
    // 相似度计算参数
    float spatial_overlap_weight_;        // 空间重叠权重
    float direction_similarity_weight_;   // 方向相似度权重
    float length_similarity_weight_;      // 长度相似度权重
    float shape_similarity_weight_;       // 形状相似度权重
    
    // 关联阈值
    float association_threshold_;         // 关联相似度阈值
    float max_association_distance_;      // 最大关联距离
    
    // 轨迹管理参数
    int max_consecutive_misses_;          // 最大连续未匹配次数
    int min_hits_for_confirmation_;       // 确认轨迹所需最小匹配次数
    float min_confidence_threshold_;      // 最小置信度阈值
    int max_track_count_;                 // 最大轨迹数量
    int history_buffer_size_;             // 历史缓存大小
    
    // 卡尔曼滤波参数
    float process_noise_position_;        // 位置过程噪声
    float process_noise_direction_;       // 方向过程噪声
    float process_noise_length_;          // 长度过程噪声
    float measurement_noise_;             // 测量噪声
    
    // 预测和补全参数
    float prediction_uncertainty_;        // 预测不确定性
    bool enable_completion_;              // 是否启用补全
    float completion_confidence_threshold_; // 补全置信度阈值


    // 可视化相关
    ros::Publisher track_markers_pub_;         // 轨迹可视化发布器
    ros::Publisher prediction_markers_pub_;    // 预测轨迹发布器
    ros::Publisher completed_lines_pub_;       // 补全线段发布器
    ros::Publisher track_info_pub_;           // 轨迹信息发布器


    // 可视化参数
    bool enable_track_visualization_;
    float track_visualization_duration_;
    std::string frame_id_;

    // 片段处理参数
    float fragment_continuity_threshold_;     // 片段连续性阈值
    float max_fragment_gap_distance_;         // 最大片段间隙距离
    float direction_consistency_threshold_;   // 方向一致性阈值
    float fragment_merge_confidence_;         // 片段合并置信度阈值
    // ==================== 内部函数 ====================
    
    // 参数读取
    void loadParameters();
    
    // 预测步骤
    void predictAllTracks();
    
    // 数据关联
    AssociationResult associateDetections(
        const std::vector<ReconstructedPowerLine>& detections,
        const PowerLineProbabilityMap* prob_map = nullptr) const;
    
    // 相似度计算
    float calculateSimilarity(const TrackedPowerLine& track,
                             const ReconstructedPowerLine& detection,
                             const PowerLineProbabilityMap* prob_map = nullptr) const;
    
    float calculateSpatialOverlap(const TrackedPowerLine& track,
                                 const ReconstructedPowerLine& detection) const;
    
    float calculateDirectionSimilarity(const TrackedPowerLine& track,
                                      const ReconstructedPowerLine& detection) const;
    
    float calculateLengthSimilarity(const TrackedPowerLine& track,
                                   const ReconstructedPowerLine& detection) const;
    
    float calculateShapeSimilarity(const TrackedPowerLine& track,
                                  const ReconstructedPowerLine& detection) const;
    
    // 轨迹生命周期管理
    void processAssociationResults(const AssociationResult& results,
                                  const std::vector<ReconstructedPowerLine>& detections);

    
    void createNewTracks(const std::vector<int>& unmatched_detections,
                        const std::vector<ReconstructedPowerLine>& detections,
                        const PowerLineProbabilityMap* prob_map = nullptr);
    
    void updateMatchedTracks(const std::vector<std::pair<int, int>>& matched_pairs,
                            const std::vector<ReconstructedPowerLine>& detections);
    
    void handleUnmatchedTracks(const std::vector<int>& unmatched_tracks);
    
    void removeInactiveTracks();
    
    // 补全功能
    std::vector<ReconstructedPowerLine> generateCompletedLines() const;
    
    // 验证功能
    bool validateNewDetection(const ReconstructedPowerLine& detection,
                             const PowerLineProbabilityMap* prob_map = nullptr) const;
    
    // 匈牙利算法（简化版本用于关联）
    std::vector<std::pair<int, int>> hungarianAssignment(const Eigen::MatrixXf& cost_matrix) const;
    
    // 辅助函数
    int findTrackById(int track_id) const;
    std::vector<Eigen::Vector3f> extractControlPoints(const ReconstructedPowerLine& line) const;
    void limitTrackCount();


    // 可视化函数
    void publishTrackVisualization();
    void publishPredictionVisualization();
    void publishCompletedLinesVisualization(const std::vector<ReconstructedPowerLine>& completed_lines);
    void publishTrackStatistics();
    visualization_msgs::Marker createTrackMarker(const TrackedPowerLine& track, int marker_id) const;
    std_msgs::ColorRGBA getTrackStatusColor(TrackStatus status) const;



};

#endif // POWER_LINE_TRACKER_H