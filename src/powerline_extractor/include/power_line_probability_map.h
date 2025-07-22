#ifndef POWER_LINE_PROBABILITY_MAP_H // 头文件保护，防止重复包含
#define POWER_LINE_PROBABILITY_MAP_H // 头文件保护宏定义

#include <ros/ros.h> // ROS主头文件
#include <pcl/point_cloud.h> // PCL点云库头文件
#include <pcl/point_types.h> // PCL点类型定义
#include <visualization_msgs/MarkerArray.h> // ROS可视化Marker数组消息
#include <visualization_msgs/Marker.h> // ROS可视化Marker消息
#include <std_msgs/ColorRGBA.h> // ROS颜色消息
#include <Eigen/Dense> // Eigen线性代数库
#include <unordered_map> // C++标准库无序哈希表
#include <vector> // C++标准库向量容器
#include <cmath> // C++标准库数学函数
#include <bits/unordered_set.h> // C++标准库无序集合
#include "power_line_reconstruction.h"  // 为了使用ReconstructedPowerLine结构体

#include <pcl/filters/crop_box.h> // PCL包围盒裁剪 <-- 添加这行
#include <pcl_conversions/pcl_conversions.h> // PCL转换 <-- 添加这行
#include <sensor_msgs/PointCloud2.h> // ROS点云消息 <-- 添加这行

// 前向声明，避免循环依赖
struct ReconstructedPowerLine; // 电力线重构结构体前向声明   表明知道这个结构体的存在

// 体素键值结构体，用于哈希表索引
struct VoxelKey {
    int x, y, z; // 体素在三维空间的索引
    
    VoxelKey(int x_idx = 0, int y_idx = 0, int z_idx = 0) // 构造函数，初始化体素索引
        : x(x_idx), y(y_idx), z(z_idx) {}
    
    bool operator==(const VoxelKey& other) const { // 判断两个体素索引是否相等
        return x == other.x && y == other.y && z == other.z;
    }
};

// AABB包围盒结构体  轴对齐包围盒  主要是如何形成包围盒的最大点和最小点，以及合并
struct AABB {
    Eigen::Vector3f min_point;  // 最小点
    Eigen::Vector3f max_point;  // 最大点
    int line_id;  // 关联的电力线ID
    
    AABB() {
        min_point = Eigen::Vector3f(FLT_MAX, FLT_MAX, FLT_MAX);
        max_point = Eigen::Vector3f(-FLT_MAX, -FLT_MAX, -FLT_MAX);
        line_id = -1;
    }
    
    AABB(const Eigen::Vector3f& min_pt, const Eigen::Vector3f& max_pt, int id = -1) 
        : min_point(min_pt), max_point(max_pt), line_id(id) {}
    
    // 扩展包围盒以包含点  应该是为了遍历每个电力线的点云，据此找到最大点和最小点
    void expandToInclude(const Eigen::Vector3f& point) {
        min_point = min_point.cwiseMin(point);   //逐元素最小值比较 每个维度取最小
        max_point = max_point.cwiseMax(point);
    }

    // 检查是否与另一个AABB重叠  如果重叠 返回true
    bool overlaps(const AABB& other) const {
        return (min_point.x() <= other.max_point.x() && max_point.x() >= other.min_point.x() &&
                min_point.y() <= other.max_point.y() && max_point.y() >= other.min_point.y() &&
                min_point.z() <= other.max_point.z() && max_point.z() >= other.min_point.z());
    }
    
    // 合并两个AABB 
    AABB merge(const AABB& other) const {
        AABB result;
        result.min_point = min_point.cwiseMin(other.min_point);
        result.max_point = max_point.cwiseMax(other.max_point);
        result.line_id = -1; // 合并后的包围盒没有单独的line_id
        return result;
    }
    
    // 获取中心点
    Eigen::Vector3f getCenter() const {
        return (min_point + max_point) * 0.5f;
    }
    
    // 获取尺寸  三个方向的长度
    Eigen::Vector3f getSize() const {
        return max_point - min_point;
    }
};

// 电力线区域信息结构体 空间位置，生命周期，稳定性
struct LineRegionInfo {
    int line_id;
    Eigen::Vector3f region_center;     // 区域中心
    float region_radius;               // 区域半径  就是电力线的一半长度加扩展半径
    Eigen::Vector3f start_point;       // 起点
    Eigen::Vector3f end_point;         // 终点
    ros::Time creation_time;           // 创建时间
    ros::Time last_active_time;        // 最后活跃时间
    bool is_stable;                    // 是否稳定
    int frames_since_creation;         // 创建后经过的帧数
    
    LineRegionInfo() {
        line_id = -1;
        region_radius = 2.0f;
        creation_time = ros::Time::now();
        last_active_time = ros::Time::now();
        is_stable = false;
        frames_since_creation = 0;
    }
};

// ROI信息结构体  高概率区域
struct LineROIInfo {
    int line_id;
    std::vector<Eigen::Vector3f> high_prob_regions;
    float confidence;
    bool is_active;
    ros::Time last_update;
    
    LineROIInfo() {
        line_id = -1;
        confidence = 0.0f;
        is_active = false;
        last_update = ros::Time::now();
    }
};



// 为VoxelKey提供哈希函数  可以在unordered_map中使用  在std中添加一个模板特化
namespace std {
    template<>
    struct hash<VoxelKey> {
        size_t operator()(const VoxelKey& key) const { // 哈希函数实现  函数调用重载
            return hash<int>()(key.x) ^ (hash<int>()(key.y) << 1) ^ (hash<int>()(key.z) << 2);
        }
    };
}

// 电力线体素信息结构体  每一个体素电力线存在的统计信息
struct PowerLineVoxel {
    float line_probability = 0.5f;        // 电力线存在概率 [0,1]
    float confidence = 0.0f;              // 置信度 [0,1]  每观测一次增加0.1，直到上限为1
    int observation_count = 0;            // 观测次数
    int frames_since_last_observation = 0; // 距离上次观测的帧数
    ros::Time last_update_time;           // 最后更新时间
    
    PowerLineVoxel() { // 构造函数，初始化更新时间
        last_update_time = ros::Time::now();
    }
    
    // 更新置信度
    void updateConfidence() {
        confidence = std::min(1.0f, observation_count * 0.1f); // 置信度随观测次数增加
    }
};

// 空间边界结构体
struct SpaceBounds {
    float x_min = -35.0f, x_max = 35.0f;  // X轴边界
    float y_min = -35.0f, y_max = 35.0f;  // Y轴边界  
    float z_min = 0.0f,   z_max = 35.0f;  // Z轴边界
    
    bool isInBounds(const Eigen::Vector3f& point) const { // 判断点是否在边界内
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
    
    // ==================== 主要接口函数 ====================
    
    /**
     * @brief 初始化概率地图（第一帧后调用）
     * @param power_lines 第一帧检测到的电力线
     * @return 成功返回true，失败返回false
     */
    bool initializeProbabilityMap(const std::vector<ReconstructedPowerLine>& power_lines); // 初始化概率地图
    
    /**
     * @brief 更新概率地图（第二帧及后续调用）
     * @param power_lines 当前帧检测到的电力线
     * @return 成功返回true，失败返回false
     */
    bool updateProbabilityMap(const std::vector<ReconstructedPowerLine>& power_lines); // 更新概率地图
    
    /**
     * @brief 获取高概率区域的中心点列表
     * @param threshold 概率阈值，默认0.7
     * @param min_confidence 最小置信度阈值，默认0.5
     * @return 高概率区域中心点列表
     */
    std::vector<Eigen::Vector3f> getHighProbabilityRegions(float threshold = 0.7f, 
                                                          float min_confidence = 0.5f) const; // 获取高概率区域
    
    /**
     * @brief 查询指定位置的电力线存在概率
     * @param position 查询位置
     * @return 该位置的电力线存在概率
     */
    float queryProbabilityAtPosition(const Eigen::Vector3f& position) const; // 查询某点概率
    
    /**
     * @brief 获取概率地图统计信息
     * @param total_voxels 总体素数量
     * @param high_prob_voxels 高概率体素数量
     * @param avg_probability 平均概率
     */
    void getMapStatistics(int& total_voxels, int& high_prob_voxels, float& avg_probability) const; // 获取统计信息


    // ==================== 分线概率地图接口 ====================
    
    /**
     * @brief 获取所有电力线的ROI信息
     * @param threshold 概率阈值，默认0.7
     * @param min_confidence 最小置信度阈值，默认0.5
     * @return 所有活跃电力线的ROI信息列表
     */
    std::vector<LineROIInfo> getAllLineROIs(float threshold = 0.7f, 
                                           float min_confidence = 0.5f) const;
    
    /**
     * @brief 获取特定电力线的高概率区域
     * @param line_id 电力线ID
     * @param threshold 概率阈值
     * @param min_confidence 最小置信度阈值
     * @return 该电力线的高概率区域
     */
    std::vector<Eigen::Vector3f> getHighProbabilityRegionsForLine(int line_id,
                                                                 float threshold = 0.7f,
                                                                 float min_confidence = 0.5f) const;
    
    /**
     * @brief 获取活跃电力线数量
     */
    int getActiveLineCount() const { return line_regions_.size(); }
    
    /**
     * @brief 检查特定电力线是否活跃
     */
    bool isLineActive(int line_id) const;
    
    // ==================== 可视化接口 ====================
    
    /**
     * @brief 可视化概率地图
     */
    void visualizeProbabilityMap(); // 可视化概率地图
    
    /**
     * @brief 启用/禁用可视化
     * @param enable 是否启用可视化
     */
    void setVisualizationEnabled(bool enable) { enable_visualization_ = enable; } // 设置可视化开关
    
    // ==================== 辅助接口 ====================
    
    /**
     * @brief 清空概率地图
     */
    void clearMap(); // 清空地图
    
    /**
     * @brief 获取体素大小
     */
    float getVoxelSize() const { return voxel_size_; } // 获取体素大小
    
    /**
     * @brief 获取总体素数量
     */
    size_t getVoxelCount() const { return voxel_map_.size(); } // 获取体素数量

    // ==================== 包围盒和点云裁剪接口 ====================

    /**
     * @brief 获取合并后的包围盒信息
     * @return 包围盒列表
     */
    std::vector<AABB> getBoundingBoxes() const { return merged_bounding_boxes_; } // <-- 添加这行

    /**
     * @brief 处理环境点云，根据电力线包围盒进行裁剪
     * @param input_cloud 输入的环境点云
     * @return 裁剪后的ROI点云
     */
    pcl::PointCloud<pcl::PointXYZI>::Ptr processEnvironmentPointCloud(
        const pcl::PointCloud<pcl::PointXYZI>::Ptr& input_cloud); // <-- 添加这行

    /**
     * @brief 处理ROS点云消息
     * @param cloud_msg ROS点云消息
     * @return 裁剪后的ROI点云
     */
    pcl::PointCloud<pcl::PointXYZI>::Ptr processEnvironmentPointCloud(
        const sensor_msgs::PointCloud2::ConstPtr& cloud_msg); // <-- 添加这行

    // ==================== 跟踪器支持接口 ====================

    /**
     * @brief 验证检测结果的可信度（供跟踪器使用）
     * @param detection 检测到的电力线
     * @param min_probability 最小概率阈值
     * @return 验证分数 [0,1]，越高越可信
     */
    float validateDetectionCredibility(const ReconstructedPowerLine& detection,
                                    float min_probability = 0.3f) const;

    /**
     * @brief 获取预测区域的概率分布（供跟踪器预测使用）
     * @param center_point 预测中心点
     * @param search_radius 搜索半径
     * @return 区域内的平均概率
     */
    float getPredictionRegionProbability(const Eigen::Vector3f& center_point,
                                        float search_radius = 2.0f) const;

    /**
     * @brief 批量查询多个点的概率（性能优化版本）
     * @param points 查询点列表
     * @return 对应的概率列表
     */
    std::vector<float> batchQueryProbability(const std::vector<Eigen::Vector3f>& points) const;



    
private:
    // ==================== 核心数据结构 ====================
    
    // 稀疏体素存储
    std::unordered_map<VoxelKey, PowerLineVoxel> voxel_map_; // 体素哈希表，存储概率信息  每一个位置对应一个电力线的体素信息
    // ==================== 分线概率地图数据 ====================
    
    // 分线概率地图：每条电力线独立的体素地图
    std::unordered_map<int, std::unordered_map<VoxelKey, PowerLineVoxel>> line_specific_maps_;    
    
    // 电力线区域信息：ID到空间区域的映射
    std::unordered_map<int, LineRegionInfo> line_regions_;
    
    // 下一个可用的电力线ID
    int next_available_line_id_;




    
    // 空间边界
    SpaceBounds bounds_; // 空间边界
    
    // ==================== ROS相关 ====================
    
    ros::NodeHandle nh_; // ROS节点句柄
    ros::Publisher prob_map_pub_;          // 概率地图可视化发布器
    ros::Publisher map_info_pub_;          // 地图信息发布器

    ros::Publisher line_specific_pub_;     // 分线概率地图发布器


    ros::Publisher bounding_box_pub_;      // 包围盒发布器 <-- 添加这行
    ros::Publisher cropped_cloud_pub_;     // 裁剪点云发布器 <-- 添加这行


    
    // ==================== 参数变量 ====================
    
    // 体素化参数
    float voxel_size_;                     // 体素大小 (m)
    float expansion_radius_;               // 样条点扩展半径 (m)
    
    // 概率参数
    float initial_probability_center_;     // 中心线初始概率  样条中心线附近的初始概率
    float initial_probability_edge_;       // 边缘初始概率
    float background_probability_;         // 背景概率
    
    // 贝叶斯更新参数
    float hit_likelihood_;                 // 检测命中似然概率
    float miss_likelihood_;                // 检测丢失似然概率
    float decay_rate_;                     // 时间衰减率
    int max_frames_without_observation_;   // 最大未观测帧数
    
    // 查询参数
    float probability_threshold_;          // 默认概率阈值
    float confidence_threshold_;           // 默认置信度阈值
    float clustering_radius_;              // 聚类半径
    
    // 可视化参数
    bool enable_visualization_;            // 是否启用可视化
    float visualization_duration_;         // 可视化持续时间
    int max_visualization_markers_;        // 最大可视化标记数
    std::string frame_id_;                 // 坐标系ID

    // 分线管理参数
    float max_inactive_duration_;          // 最大非活跃时间 (秒)
    float spatial_overlap_threshold_;      // 空间重叠阈值
    int min_stable_frames_;               // 最小稳定帧数
    int max_line_count_;                  // 最大电力线数量
    // ==================== 包围盒相关 ====================
    std::vector<AABB> merged_bounding_boxes_; // 合并后的包围盒列表 <-- 添加这行



    
    // ==================== 内部函数 ====================
    
    // 坐标转换函数
    VoxelKey pointToVoxel(const Eigen::Vector3f& point) const; // 点转体素索引
    Eigen::Vector3f voxelToPoint(const VoxelKey& voxel_key) const; // 体素索引转点
    
    // 体素标记函数 voxel_map_
    void markLineRegion(const Eigen::Vector3f& spline_point, 
                       const Eigen::Vector3f& direction,
                       float initial_probability); // 标记电力线区域
    
    void markCylindricalRegion(const Eigen::Vector3f& center,
                              const Eigen::Vector3f& direction,
                              float length,
                              float radius,
                              float probability); // 标记圆柱体区域（未实现）
    
    // 贝叶斯更新函数
    float updateBayesian(float prior_probability, 
                        float likelihood, 
                        bool positive_evidence) const; // 贝叶斯概率更新
    
    void bayesianUpdate(const std::vector<ReconstructedPowerLine>& power_lines); // 贝叶斯批量更新
    void decayUnobservedRegions(); // 衰减未观测区域
    
    // 区域查询函数
    std::vector<Eigen::Vector3f> clusterAdjacentRegions(
        const std::vector<Eigen::Vector3f>& points) const; // 聚类相邻区域
    
    // 几何辅助函数
    Eigen::Vector3f getPerpendicularVector(const Eigen::Vector3f& direction, 
                                          float radius, 
                                          float angle) const; // 获取垂直向量
    
    float calculateInitialProbability(float distance_from_centerline) const; // 计算初始概率
    
    Eigen::Vector3f computeLocalDirection(const std::vector<Eigen::Vector3f>& curve_points,
                                         size_t point_index) const; // 计算局部方向  就是计算点之间的方向向量并标准化一下
    
    // 可视化辅助函数
    void publishProbabilityMarkers(); // 发布体素可视化
    void publishMapStatistics(); // 发布统计信息
    void publishLineSpecificMarkers(); // 发布分线可视化 <-- 添加这行
    std_msgs::ColorRGBA getLineColor(int line_id) const; // 获取电力线颜色 <-- 添加这行
    
    std_msgs::ColorRGBA probabilityToColor(float probability, float confidence) const; // 概率转颜色
    
    visualization_msgs::Marker createVoxelMarker(const VoxelKey& voxel_key,
                                                const PowerLineVoxel& voxel,
                                                int marker_id) const; // 创建体素Marker
    
    // 参数读取函数
    void loadParameters(); // 读取参数

    // 分线管理函数
    int assignLineID(const ReconstructedPowerLine& new_line); // 分配电力线ID
    void updateLineSpecificMaps(const std::vector<ReconstructedPowerLine>& power_lines); // 更新分线地图
    void manageLineLifecycles(); // 管理电力线生命周期
    void createNewLineRegion(int line_id, const ReconstructedPowerLine& line); // 根据三次样条曲线创建新区域 LineRegionInfo
    float calculateSpatialOverlap(const ReconstructedPowerLine& line, const LineRegionInfo& region) const; // 计算空间重叠
    std::vector<Eigen::Vector3f> extractHighProbRegions(const std::unordered_map<VoxelKey, PowerLineVoxel>& line_map, float threshold) const; // 提取高概率区域
    float calculateLineConfidence(int line_id) const; // 计算电力线置信度
    void markLineRegionForSpecificLine(int line_id, const Eigen::Vector3f& spline_point, 
                                      const Eigen::Vector3f& direction, float initial_probability); // 标记特定电力线区域
    
    // 数据验证函数 先是power_lines是否为空，再是是否里面有拟合曲线，然后是否在范围内
    bool validatePowerLines(const std::vector<ReconstructedPowerLine>& power_lines) const; // 验证输入数据
    bool isInitialized() const { return !voxel_map_.empty(); } // 判断是否初始化

    // 包围盒相关函数
    std::vector<AABB> calculateLineBoundingBoxes() const; // 计算电力线包围盒 <-- 添加这行
    std::vector<AABB> mergeBoundingBoxes(const std::vector<AABB>& boxes) const; // 合并包围盒 <-- 添加这行
    void publishBoundingBoxes(); // 发布包围盒可视化 <-- 添加这行
    pcl::PointCloud<pcl::PointXYZ>::Ptr cropPointCloudWithBoxes(
        const pcl::PointCloud<pcl::PointXYZ>::Ptr& input_cloud) const; // 裁剪点云 <-- 添加这行
    void publishCroppedPointCloud(const pcl::PointCloud<pcl::PointXYZI>::Ptr& cloud); // 发布裁剪点云 <-- 添加这行



    

    
};

#endif // POWER_LINE_PROBABILITY_MAP_H  头文件保护宏结束