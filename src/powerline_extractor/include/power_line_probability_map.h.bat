#ifndef POWER_LINE_PROBABILITY_MAP_H
#define POWER_LINE_PROBABILITY_MAP_H

#include <ros/ros.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <sensor_msgs/PointCloud2.h>
#include <visualization_msgs/MarkerArray.h>
#include <visualization_msgs/Marker.h>
#include <std_msgs/ColorRGBA.h>
#include <Eigen/Dense>
#include <unordered_map>
#include <vector>
#include <chrono>
#include "power_line_reconstruction.h"


// 前向声明 - 需要包含你的电力线重构头文件
struct ReconstructedPowerLine;

// 体素索引结构体
struct VoxelKey {
    int x, y, z;
    
    VoxelKey() : x(0), y(0), z(0) {}
    VoxelKey(int x_, int y_, int z_) : x(x_), y(y_), z(z_) {}
    
    bool operator==(const VoxelKey& other) const {
        return x == other.x && y == other.y && z == other.z;
    }
};

// 体素键的哈希函数
struct VoxelKeyHash {
    std::size_t operator()(const VoxelKey& key) const {
        return std::hash<int>()(key.x) ^ 
               (std::hash<int>()(key.y) << 1) ^ 
               (std::hash<int>()(key.z) << 2);
    }
};

// 电力线体素信息
struct PowerLineVoxel {
    float line_probability = 0.5f;        // 电力线存在概率 [0,1]
    float confidence = 0.0f;              // 置信度 [0,1]
    Eigen::Vector3f local_direction;      // 局部电力线方向
    int observation_count = 0;            // 观测次数
    int frames_since_last_observation = 0; // 自上次观测的帧数
    ros::Time last_update_time;           // 最后更新时间
    
    PowerLineVoxel() {
        local_direction = Eigen::Vector3f::Zero();
        last_update_time = ros::Time::now();
    }
};

// ROI区域信息
struct ROIRegion {
    Eigen::Vector3f center;               // ROI中心点
    float radius;                        // ROI半径
    float average_probability;           // 平均概率
    float confidence;                    // 置信度
    
    ROIRegion(const Eigen::Vector3f& c, float r, float prob, float conf) 
        : center(c), radius(r), average_probability(prob), confidence(conf) {}
};

// 空间边界
struct SpaceBounds {
    float x_min, x_max;
    float y_min, y_max; 
    float z_min, z_max;
    
    SpaceBounds() : x_min(-35.0f), x_max(35.0f), 
                   y_min(-35.0f), y_max(35.0f),
                   z_min(0.0f), z_max(35.0f) {}
    
    bool isInBounds(const Eigen::Vector3f& point) const {
        return point.x() >= x_min && point.x() <= x_max &&
               point.y() >= y_min && point.y() <= y_max &&
               point.z() >= z_min && point.z() <= z_max;
    }
};

class PowerLineProbabilityMap {
public:
    // 构造函数：通过ROS节点句柄读取参数
    PowerLineProbabilityMap(ros::NodeHandle& nh);
    
    // 析构函数
    ~PowerLineProbabilityMap();
    
    // ==================== 核心接口函数 ====================
    
    /**
     * @brief 初始化概率地图（第一帧使用）
     * @param lines 检测到的电力线列表
     */
    void initializeProbabilityMap(const std::vector<ReconstructedPowerLine>& lines);
    
    /**
     * @brief 贝叶斯更新概率地图（后续帧使用）
     * @param lines 当前帧检测到的电力线列表
     */
    void updateProbabilityMap(const std::vector<ReconstructedPowerLine>& lines);
    
    /**
     * @brief 获取高概率区域作为ROI
     * @param probability_threshold 概率阈值 [0,1]
     * @param confidence_threshold 置信度阈值 [0,1]
     * @return ROI区域列表
     */
    std::vector<ROIRegion> getHighProbabilityRegions(
        float probability_threshold = 0.7f, 
        float confidence_threshold = 0.5f) const;
    
    /**
     * @brief 根据ROI区域裁剪点云
     * @param input_cloud 输入完整点云
     * @param roi_regions ROI区域列表
     * @param expansion_factor ROI扩展系数 (默认1.2倍)
     * @return 裁剪后的ROI点云
     */
    pcl::PointCloud<pcl::PointXYZI>::Ptr extractROIPointCloud(
        const pcl::PointCloud<pcl::PointXYZI>::Ptr& input_cloud,
        const std::vector<ROIRegion>& roi_regions,
        float expansion_factor = 1.2f) const;
    
    /**
     * @brief 获取历史电力线信息（用于补全）
     * @param min_confidence 最小置信度要求
     * @return 历史电力线的样条点列表
     */
    std::vector<std::vector<Eigen::Vector3f>> getHistoricalSplinePoints(
        float min_confidence = 0.6f) const;
    
    // ==================== 可视化接口 ====================
    
    /**
     * @brief 可视化概率地图
     */
    void visualizeProbabilityMap();
    
    /**
     * @brief 可视化ROI区域
     * @param roi_regions ROI区域列表
     */
    void visualizeROIRegions(const std::vector<ROIRegion>& roi_regions);
    
    // ==================== 查询接口 ====================
    
    /**
     * @brief 查询指定位置的电力线概率
     * @param position 世界坐标位置
     * @return 概率值 [0,1]，如果位置无数据返回0.5
     */
    float queryProbabilityAtPosition(const Eigen::Vector3f& position) const;
    
    /**
     * @brief 获取概率地图统计信息
     */
    void printMapStatistics() const;
    
    /**
     * @brief 清空概率地图
     */
    void clearMap();

private:
    // ==================== ROS相关 ====================
    ros::NodeHandle nh_;
    ros::Publisher prob_map_vis_pub_;         // 概率地图可视化发布器
    ros::Publisher roi_vis_pub_;              // ROI可视化发布器
    ros::Publisher roi_pointcloud_pub_;       // ROI点云发布器
    
    // ==================== 参数变量 ====================
    // 体素化参数
    float voxel_size_;                        // 体素大小 (m)
    float expansion_radius_;                  // 电力线扩展半径 (m)
    
    // 贝叶斯更新参数
    float prob_hit_;                          // 命中概率
    float prob_miss_;                         // 未命中概率
    float decay_rate_;                        // 历史信息衰减率
    int max_frames_without_observation_;      // 最大无观测帧数
    
    // 概率阈值参数
    float initial_line_probability_;          // 初始电力线概率
    float background_probability_;            // 背景概率
    float uncertainty_probability_;           // 不确定概率
    
    // 可视化参数
    bool enable_visualization_;               // 是否启用可视化
    float visualization_update_rate_;         // 可视化更新频率 (Hz)
    float marker_lifetime_;                   // 标记生命周期 (s)
    int max_markers_per_publish_;            // 每次发布的最大标记数
    
    // 空间参数
    SpaceBounds space_bounds_;               // 空间边界
    std::string frame_id_;                   // 坐标系ID
    
    // ==================== 核心数据结构 ====================
    std::unordered_map<VoxelKey, PowerLineVoxel, VoxelKeyHash> voxel_map_;
    
    // ==================== 私有辅助函数 ====================
    
    // 坐标转换
    VoxelKey worldToVoxel(const Eigen::Vector3f& world_pos) const;
    Eigen::Vector3f voxelToWorld(const VoxelKey& voxel_key) const;
    
    // 电力线区域标记
    void markLineRegion(const Eigen::Vector3f& spline_point, 
                       const Eigen::Vector3f& direction,
                       float initial_probability);
    
    // 贝叶斯更新
    float updateBayesian(float prior_prob, float likelihood, bool positive_evidence) const;
    
    // 概率计算
    float calculateInitialProbability(float distance_from_centerline) const;
    float calculateLikelihood(float distance_from_centerline) const;
    
    // 历史信息管理
    void decayUnobservedRegions();
    void updateConfidence(PowerLineVoxel& voxel);
    
    // 空间查询
    std::vector<VoxelKey> getNeighborVoxels(const VoxelKey& center, int radius = 1) const;
    bool isValidVoxel(const VoxelKey& key) const;
    
    // ROI处理
    std::vector<ROIRegion> clusterHighProbabilityRegions(
        const std::vector<VoxelKey>& high_prob_voxels) const;
    
    // 可视化辅助
    std_msgs::ColorRGBA probabilityToColor(float probability, float confidence) const;
    visualization_msgs::Marker createVoxelMarker(const VoxelKey& key, 
                                                const PowerLineVoxel& voxel, 
                                                int marker_id) const;
    visualization_msgs::Marker createROIMarker(const ROIRegion& roi, int marker_id) const;
    
    // 几何计算
    Eigen::Vector3f getPerpendicularVector(const Eigen::Vector3f& direction, 
                                          float radius, float angle) const;
    std::vector<Eigen::Vector3f> generateCircularPoints(const Eigen::Vector3f& center,
                                                       const Eigen::Vector3f& direction,
                                                       float radius, int num_points = 16) const;
    
    // 参数读取
    void loadParameters();
    
    // 统计信息
    int total_voxels_created_ = 0;
    int total_updates_performed_ = 0;
    ros::Time last_visualization_time_;
};

#endif // POWER_LINE_PROBABILITY_MAP_H