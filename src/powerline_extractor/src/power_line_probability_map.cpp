#include "power_line_probability_map.h" // 包含头文件，声明类和相关结构体

#include <chrono> // 用于计时统计
#include <algorithm> // 用于算法函数如std::max
#include <pcl/common/common.h> // PCL通用函数

PowerLineProbabilityMap::PowerLineProbabilityMap(ros::NodeHandle& nh) : nh_(nh) { // 构造函数，初始化ROS句柄  初始化概率地图
    // 读取参数
    loadParameters();

    // 初始化分线管理变量
    next_available_line_id_ = 0;
    


    // 初始化ROS发布器
    prob_map_pub_ = nh_.advertise<visualization_msgs::MarkerArray>(
        "power_line_probability_map/probability_markers", 1); // 概率地图Marker发布器
    map_info_pub_ = nh_.advertise<visualization_msgs::Marker>(
        "power_line_probability_map/map_info", 1); // 地图统计信息发布器
    line_specific_pub_ = nh_.advertise<visualization_msgs::MarkerArray>(
        "power_line_probability_map/line_specific_markers", 1); // 分线概率地图发布器 <-- 添加这行



    bounding_box_pub_ = nh_.advertise<visualization_msgs::MarkerArray>(
    "power_line_probability_map/bounding_boxes", 1); // 包围盒发布器 <-- 添加这行
    cropped_cloud_pub_ = nh_.advertise<sensor_msgs::PointCloud2>(
    "power_line_probability_map/cropped_pointcloud", 1); // 裁剪点云发布器 <-- 添加这行


    
    ROS_INFO("PowerLineProbabilityMap 初始化完成"); // 输出初始化信息
    ROS_INFO("参数配置:");
    ROS_INFO("  体素大小: %.3f m", voxel_size_);
    ROS_INFO("  扩展半径: %.3f m", expansion_radius_);
    ROS_INFO("  初始概率(中心): %.2f", initial_probability_center_);
    ROS_INFO("  初始概率(边缘): %.2f", initial_probability_edge_);
    ROS_INFO("  命中似然: %.2f", hit_likelihood_);
    ROS_INFO("  丢失似然: %.2f", miss_likelihood_);
    ROS_INFO("  衰减率: %.3f", decay_rate_);
    ROS_INFO("  概率阈值: %.2f", probability_threshold_);
    ROS_INFO("  置信度阈值: %.2f", confidence_threshold_);
    ROS_INFO("  启用可视化: %s", enable_visualization_ ? "是" : "否");

}

PowerLineProbabilityMap::~PowerLineProbabilityMap() { // 析构函数
    clearMap(); // 清空地图
    ROS_INFO("PowerLineProbabilityMap 析构完成"); // 输出析构信息
}

void PowerLineProbabilityMap::loadParameters() { // 读取参数
    // 体素化参数
    nh_.param("probability_map/voxel_size", voxel_size_, 0.1f); // 体素大小
    nh_.param("probability_map/expansion_radius", expansion_radius_, 0.15f); // 扩展半径
    
    // 空间边界参数
    nh_.param("probability_map/x_min", bounds_.x_min, -35.0f); // X最小
    nh_.param("probability_map/x_max", bounds_.x_max, 35.0f); // X最大
    nh_.param("probability_map/y_min", bounds_.y_min, -35.0f); // Y最小
    nh_.param("probability_map/y_max", bounds_.y_max, 35.0f); // Y最大
    nh_.param("probability_map/z_min", bounds_.z_min, 0.0f); // Z最小
    nh_.param("probability_map/z_max", bounds_.z_max, 35.0f); // Z最大
    
    // 概率参数
    nh_.param("probability_map/initial_probability_center", initial_probability_center_, 0.9f); // 中心概率
    nh_.param("probability_map/initial_probability_edge", initial_probability_edge_, 0.6f); // 边缘概率
    nh_.param("probability_map/background_probability", background_probability_, 0.1f); // 背景概率
    
    // 贝叶斯更新参数
    nh_.param("probability_map/hit_likelihood", hit_likelihood_, 0.8f); // 命中似然
    nh_.param("probability_map/miss_likelihood", miss_likelihood_, 0.2f); // 丢失似然
    nh_.param("probability_map/decay_rate", decay_rate_, 0.98f); // 衰减率
    nh_.param("probability_map/max_frames_without_observation", max_frames_without_observation_, 10); // 最大未观测帧数
    
    // 查询参数
    nh_.param("probability_map/probability_threshold", probability_threshold_, 0.7f); // 概率阈值
    nh_.param("probability_map/confidence_threshold", confidence_threshold_, 0.5f); // 置信度阈值
    nh_.param("probability_map/clustering_radius", clustering_radius_, 1.0f); // 聚类半径
    
    // 可视化参数
    nh_.param("probability_map/enable_visualization", enable_visualization_, true); // 可视化开关
    nh_.param("probability_map/visualization_duration", visualization_duration_, 5.0f); // 可视化持续时间
    nh_.param("probability_map/max_visualization_markers", max_visualization_markers_, 5000); // 最大可视化标记数
    nh_.param("probability_map/frame_id", frame_id_, std::string("base_link")); // 坐标系ID

    // 分线管理参数
    nh_.param("probability_map/max_inactive_duration", max_inactive_duration_, 30.0f); // 最大非活跃时间
    nh_.param("probability_map/spatial_overlap_threshold", spatial_overlap_threshold_, 0.7f); // 空间重叠阈值
    nh_.param("probability_map/coincidence_rate_threshold", coincidence_rate_threshold_, 0.3f); // 时间重叠阈值
    nh_.param("probability_map/min_stable_frames", min_stable_frames_, 3); // 最小稳定帧数
    nh_.param("probability_map/max_line_count", max_line_count_, 20); // 最大电力线数量
}

bool PowerLineProbabilityMap::initializeProbabilityMap(
    const std::vector<ReconstructedPowerLine>& power_lines) { // 初始化概率地图
    
    auto start = std::chrono::high_resolution_clock::now(); // 记录开始时间
    
    ROS_INFO("开始初始化概率地图，电力线数量: %zu", power_lines.size()); // 输出电力线数量
    
    // 验证输入数据
    if (!validatePowerLines(power_lines)) { // 检查输入有效性
        ROS_ERROR("输入电力线数据无效"); // 输出错误
        return false; // 返回失败
    }
    
    // 清空现有地图
    clearMap(); // 清空体素哈希表
    
    // 为每条电力线建立概率区域
    for (const auto& line : power_lines) { // 遍历每条电力线
        if (line.fitted_curve_points.empty()) { // 如果拟合点为空
            ROS_WARN("电力线 %d 没有拟合曲线点，跳过", line.line_id); // 输出警告
            continue; // 跳过
        }
        
        ROS_DEBUG("处理电力线 %d，拟合点数: %zu", 
                 line.line_id, line.fitted_curve_points.size()); // 输出调试信息


        
        // 沿着三次样条点建立概率区域
        for (size_t i = 0; i < line.fitted_curve_points.size(); ++i) { // 遍历拟合点
            const auto& spline_point = line.fitted_curve_points[i]; // 当前样条点
            
            // 检查是否在有效范围内
            if (!bounds_.isInBounds(spline_point)) { // 超出边界跳过
                continue;
            }
            
            // 计算局部方向
            Eigen::Vector3f local_direction = computeLocalDirection(line.fitted_curve_points, i); // 计算方向

            // 分配电力线ID（第一帧）
            if (i == 0) { // 只在处理第一个点时分配ID
                if (line_regions_.find(line.line_id) == line_regions_.end()) {  //如果这个id还没有在其中
                    createNewLineRegion(line.line_id, line);
                }
            }
            
            // 同时更新全局地图和分线地图
            markLineRegion(spline_point, local_direction, initial_probability_center_);  //将样条点扩展到体素中，并加入voxel_map_
            markLineRegionForSpecificLine(line.line_id, spline_point, local_direction, initial_probability_center_); //将该样条加入到每条线的概率地图
        }
    }
    
    auto end = std::chrono::high_resolution_clock::now(); // 记录结束时间
    std::chrono::duration<double> duration = end - start; // 计算耗时
    
    ROS_INFO("概率地图初始化完成"); // 输出完成信息
    ROS_INFO("  耗时: %.3f 秒", duration.count()); // 输出耗时
    ROS_INFO("  创建体素数: %zu", voxel_map_.size()); // 输出体素数


    
    // 发布可视化
    if (enable_visualization_) { // 如果启用可视化
        visualizeProbabilityMap(); // 可视化地图

    }
    
    return true; // 返回成功
}

bool PowerLineProbabilityMap::updateProbabilityMap(
    const std::vector<ReconstructedPowerLine>& power_lines) { // 更新概率地图
    
    auto start = std::chrono::high_resolution_clock::now(); // 记录开始时间
    
    ROS_INFO("更新概率地图，新检测电力线数量: %zu", power_lines.size()); // 输出电力线数量
    
    if (!isInitialized()) { // 如果未初始化
        ROS_INFO("概率地图未初始化，请先调用initializeProbabilityMap"); // 输出提示
        return false; // 返回失败
    }
    
    // 验证输入数据
    if (!validatePowerLines(power_lines)) { // 检查输入有效性
        ROS_INFO("输入电力线数据无效，跳过更新"); // 输出提示
        return false; // 返回失败
    }
    
    // 增加所有体素的未观测帧数
    for (auto& [key, voxel] : voxel_map_) { // 遍历所有体素
        voxel.frames_since_last_observation++; // 未观测帧数加1
    }
    
    // // 贝叶斯更新
    // bayesianUpdate(power_lines); // 执行贝叶斯概率更新

    // 更新分线概率地图
    updateLineSpecificMaps(power_lines);

    
    // 管理电力线生命周期
    manageLineLifecycles();

    // // 衰减长期未观测区域
    // decayUnobservedRegions(); // 衰减未观测体素
    
    auto end = std::chrono::high_resolution_clock::now(); // 记录结束时间
    std::chrono::duration<double> duration = end - start; // 计算耗时
    
    ROS_INFO("概率地图更新完成，耗时: %.3f 秒", duration.count()); // 输出完成信息
    
    // 发布可视化
    if (enable_visualization_) { // 如果启用可视化
        visualizeProbabilityMap(); // 可视化地图

    }
    
    return true; // 返回成功
}

std::vector<Eigen::Vector3f> PowerLineProbabilityMap::getHighProbabilityRegions(
    float threshold, float min_confidence) const { // 获取高概率区域
    
    std::vector<Eigen::Vector3f> high_prob_points; // 存储高概率点
    
    for (const auto& [key, voxel] : voxel_map_) { // 遍历所有体素
        if (voxel.line_probability > threshold && voxel.confidence > min_confidence) { // 满足条件
            high_prob_points.push_back(voxelToPoint(key)); // 加入结果
        }
    }
    
    ROS_DEBUG("查询到 %zu 个高概率区域 (P > %.2f, C > %.2f)", 
             high_prob_points.size(), threshold, min_confidence); // 输出调试信息
    
    // 聚类相邻区域
    return clusterAdjacentRegions(high_prob_points); // 返回聚类中心
}

float PowerLineProbabilityMap::queryProbabilityAtPosition(const Eigen::Vector3f& position) const { // 查询某点概率
    if (!bounds_.isInBounds(position)) { // 超出边界
        return background_probability_; // 返回背景概率
    }
    
    VoxelKey key = pointToVoxel(position); // 转为体素索引
    auto it = voxel_map_.find(key); // 查找体素
    
    if (it != voxel_map_.end()) { // 找到
        return it->second.line_probability; // 返回概率
    } else {
        return background_probability_;  // 未标记区域默认为背景概率
    }
}

void PowerLineProbabilityMap::getMapStatistics(int& total_voxels, 
                                              int& high_prob_voxels, 
                                              float& avg_probability) const { // 获取统计信息
    total_voxels = voxel_map_.size(); // 总体素数
    high_prob_voxels = 0; // 高概率体素数
    float sum_probability = 0.0f; // 概率和
    
    for (const auto& [key, voxel] : voxel_map_) { // 遍历体素
        sum_probability += voxel.line_probability; // 累加概率
        if (voxel.line_probability > probability_threshold_) { // 高概率
            high_prob_voxels++;
        }
    }
    
    avg_probability = total_voxels > 0 ? sum_probability / total_voxels : 0.0f; // 平均概率
}

void PowerLineProbabilityMap::clearMap() { // 清空地图
    voxel_map_.clear(); // 清空体素哈希表
    // 清空分线相关数据
    line_specific_maps_.clear();
    line_regions_.clear();
    next_available_line_id_ = 0;

    ROS_DEBUG("概率地图已清空"); // 输出调试信息
}

// ==================== 内部实现函数 ====================

VoxelKey PowerLineProbabilityMap::pointToVoxel(const Eigen::Vector3f& point) const { // 点转体素索引
    int x = static_cast<int>(std::floor(point.x() / voxel_size_)); // X索引
    int y = static_cast<int>(std::floor(point.y() / voxel_size_)); // Y索引
    int z = static_cast<int>(std::floor(point.z() / voxel_size_)); // Z索引
    return VoxelKey(x, y, z); // 返回体素索引
}

Eigen::Vector3f PowerLineProbabilityMap::voxelToPoint(const VoxelKey& voxel_key) const { // 体素索引转点
    float x = (voxel_key.x + 0.5f) * voxel_size_; // X坐标
    float y = (voxel_key.y + 0.5f) * voxel_size_;
    float z = (voxel_key.z + 0.5f) * voxel_size_;
    return Eigen::Vector3f(x, y, z); // 返回点
}
//将样条点扩展，并加入voxel_map_
void PowerLineProbabilityMap::markLineRegion(const Eigen::Vector3f& spline_point,
                                           const Eigen::Vector3f& direction,
                                           float initial_probability) {
    // 计算需要检查的体素范围
    int radius_in_voxels = static_cast<int>(std::ceil(expansion_radius_ / voxel_size_));
    VoxelKey center_key = pointToVoxel(spline_point);  //这个点的体素索引
    
    // 遍历周围体素 扩展周围3*3*3的体素范围
    for (int dx = -radius_in_voxels; dx <= radius_in_voxels; ++dx) {
        for (int dy = -radius_in_voxels; dy <= radius_in_voxels; ++dy) {
            for (int dz = -radius_in_voxels; dz <= radius_in_voxels; ++dz) {
                VoxelKey key(center_key.x + dx, center_key.y + dy, center_key.z + dz);
                Eigen::Vector3f voxel_center = voxelToPoint(key);
                
                if (!bounds_.isInBounds(voxel_center)) continue;
                
                // 计算体素中心到样条点的距离
                float distance = (voxel_center - spline_point).norm();
                
                if (distance <= expansion_radius_) {  //加一个小判断，将扩展的体素加入voxel_map_
                    auto& voxel = voxel_map_[key];
                    
                    // 根据距离设置概率
                    float probability = calculateInitialProbability(distance);
                    voxel.line_probability = std::max(voxel.line_probability, probability);
                    voxel.observation_count++;
                    voxel.updateConfidence();
                    voxel.frames_since_last_observation = 0;
                    voxel.last_update_time = ros::Time::now();
                }
            }
        }
    }
}
void PowerLineProbabilityMap::bayesianUpdate(const std::vector<ReconstructedPowerLine>& power_lines) {
    // 标记当前帧检测到的区域
    std::unordered_set<VoxelKey> observed_voxels;
    
    for (const auto& line : power_lines) {
        for (const auto& spline_point : line.fitted_curve_points) {
            if (!bounds_.isInBounds(spline_point)) continue;
            
            // 计算需要检查的体素范围
            int radius_in_voxels = static_cast<int>(std::ceil(expansion_radius_ / voxel_size_));
            VoxelKey center_key = pointToVoxel(spline_point);
            
            // 遍历周围体素
            for (int dx = -radius_in_voxels; dx <= radius_in_voxels; ++dx) {
                for (int dy = -radius_in_voxels; dy <= radius_in_voxels; ++dy) {
                    for (int dz = -radius_in_voxels; dz <= radius_in_voxels; ++dz) {
                        VoxelKey key(center_key.x + dx, center_key.y + dy, center_key.z + dz);
                        Eigen::Vector3f voxel_center = voxelToPoint(key);
                        
                        if (!bounds_.isInBounds(voxel_center)) continue;
                        
                        // 计算体素中心到样条点的距离
                        float distance = (voxel_center - spline_point).norm();
                        
                        if (distance <= expansion_radius_) {
                            observed_voxels.insert(key);
                            auto& voxel = voxel_map_[key];
                            
                            // 贝叶斯更新：检测命中
                            voxel.line_probability = updateBayesian(
                                voxel.line_probability, hit_likelihood_, true);
                            
                            voxel.observation_count++;
                            voxel.updateConfidence();
                            voxel.frames_since_last_observation = 0;
                            voxel.last_update_time = ros::Time::now();
                        }
                    }
                }
            }
        }
    }
    
    // 对于历史存在但当前未检测到的区域，降低概率
    for (auto& [key, voxel] : voxel_map_) {
        if (observed_voxels.find(key) == observed_voxels.end()) {
            // 贝叶斯更新：检测丢失
            voxel.line_probability = updateBayesian(
                voxel.line_probability, miss_likelihood_, false);
        }
    }
}
// 贝叶斯概率更新
float PowerLineProbabilityMap::updateBayesian(float prior_probability,
                                             float likelihood,
                                             bool positive_evidence) const {
    // 修正版本：正确处理似然比
    
    // 第一步：计算似然比
    float likelihood_ratio;
    if (positive_evidence) {
        // 正证据：P(E+|H) / P(E+|¬H) = hit_likelihood / miss_likelihood
        likelihood_ratio = likelihood / miss_likelihood_;
    } else {
        // 负证据：P(E-|H) / P(E-|¬H) = (1-hit_likelihood) / (1-miss_likelihood)
        likelihood_ratio = (1.0f - likelihood) / (1.0f - miss_likelihood_);
    }
    
    // 第二步：计算先验对数几率
    float log_odds_prior = std::log(prior_probability / (1.0f - prior_probability + 1e-6f));
    
    // 第三步：计算似然比的对数
    float log_likelihood_ratio = std::log(likelihood_ratio + 1e-6f);
    
    // 第四步：贝叶斯更新
    float log_odds_posterior = log_odds_prior + log_likelihood_ratio;
    
    // 第五步：转换回概率
    float posterior = 1.0f / (1.0f + std::exp(-log_odds_posterior));
    
    // 第六步：限制在合理范围内
    return std::max(0.01f, std::min(0.99f, posterior));
}

// 衰减未观测区域
void PowerLineProbabilityMap::decayUnobservedRegions() { // 衰减未观测区域
    for (auto& [key, voxel] : voxel_map_) {
        if (voxel.frames_since_last_observation > max_frames_without_observation_) {
            // 向不确定状态衰减
            float target = 0.5f;  // 不确定状态
            voxel.line_probability = target + (voxel.line_probability - target) * decay_rate_;
            
            // 置信度也逐渐衰减
            voxel.confidence *= decay_rate_;
        }
    }
}

std::vector<Eigen::Vector3f> PowerLineProbabilityMap::clusterAdjacentRegions(
    const std::vector<Eigen::Vector3f>& points) const { // 聚类相邻区域
    
    if (points.empty()) return points; // 空则返回
    
    std::vector<bool> visited(points.size(), false); // 标记访问
    std::vector<Eigen::Vector3f> cluster_centers; // 聚类中心
    
    for (size_t i = 0; i < points.size(); ++i) {
        if (visited[i]) continue; // 已访问跳过
        
        // 开始新的聚类
        std::vector<Eigen::Vector3f> cluster;
        std::queue<size_t> queue;
        queue.push(i);
        visited[i] = true;
        
        while (!queue.empty()) {
            size_t current = queue.front();
            queue.pop();
            cluster.push_back(points[current]);
            
            // 查找相邻点
            for (size_t j = 0; j < points.size(); ++j) {
                if (!visited[j]) {
                    float distance = (points[current] - points[j]).norm();
                    if (distance <= clustering_radius_) {
                        visited[j] = true;
                        queue.push(j);
                    }
                }
            }
        }
        
        // 计算聚类中心
        Eigen::Vector3f center = Eigen::Vector3f::Zero();
        for (const auto& point : cluster) {
            center += point;
        }
        center /= cluster.size();
        cluster_centers.push_back(center);
    }
    
    ROS_DEBUG("将 %zu 个点聚类为 %zu 个区域", points.size(), cluster_centers.size()); // 输出调试信息
    return cluster_centers; // 返回聚类中心
}

Eigen::Vector3f PowerLineProbabilityMap::getPerpendicularVector(
    const Eigen::Vector3f& direction, float radius, float angle) const { // 获取垂直向量
    
    // 创建垂直于方向的两个正交向量
    Eigen::Vector3f perp1, perp2;
    
    if (std::abs(direction.z()) < 0.9f) {
        perp1 = direction.cross(Eigen::Vector3f::UnitZ()).normalized();
    } else {
        perp1 = direction.cross(Eigen::Vector3f::UnitX()).normalized();
    }
    
    perp2 = direction.cross(perp1).normalized();
    
    // 生成圆形截面上的点
    return radius * (std::cos(angle) * perp1 + std::sin(angle) * perp2);
}
//根据到电力线的采样点的距离设置体素概率
float PowerLineProbabilityMap::calculateInitialProbability(float distance_from_centerline) const { // 计算初始概率
    if (distance_from_centerline < 0.05f) {
        return initial_probability_center_;  // 中心线附近
    } else if (distance_from_centerline < 0.1f) {
        return (initial_probability_center_ + initial_probability_edge_) / 2.0f;  // 中间区域
    } else if (distance_from_centerline < expansion_radius_) {
        return initial_probability_edge_;  // 边缘区域
    } else {
        return background_probability_;  // 背景
    }
}
//计算局部方向
Eigen::Vector3f PowerLineProbabilityMap::computeLocalDirection(
    const std::vector<Eigen::Vector3f>& curve_points, size_t point_index) const { // 计算局部方向
    
    if (curve_points.size() < 2) {
        return Eigen::Vector3f::UnitX();  // 默认方向
    }
    
    if (point_index == 0) {
        // 第一个点：使用向前差分
        return (curve_points[1] - curve_points[0]).normalized();
    } else if (point_index == curve_points.size() - 1) {
        // 最后一个点：使用向后差分
        return (curve_points[point_index] - curve_points[point_index - 1]).normalized();
    } else {
        // 中间点：使用中心差分
        return (curve_points[point_index + 1] - curve_points[point_index - 1]).normalized();
    }
}

bool PowerLineProbabilityMap::validatePowerLines(
    const std::vector<ReconstructedPowerLine>& power_lines) const { // 验证输入数据
    
    if (power_lines.empty()) {
        ROS_WARN("电力线列表为空"); // 输出警告
        return false;
    }
    
    for (const auto& line : power_lines) {
        if (line.fitted_curve_points.empty()) {
            ROS_WARN("电力线 %d 没有拟合曲线点", line.line_id); // 输出警告
            continue;
        }
        
        // 检查点是否在合理范围内
        for (const auto& point : line.fitted_curve_points) {
            if (!bounds_.isInBounds(point)) {
                ROS_DEBUG("电力线 %d 包含超出边界的点: (%.2f, %.2f, %.2f)", 
                         line.line_id, point.x(), point.y(), point.z()); // 输出调试
            }
        }
    }
    
    return true;
}

// ==================== 可视化函数 ====================

void PowerLineProbabilityMap::visualizeProbabilityMap() { // 可视化概率地图
    if (!enable_visualization_ || voxel_map_.empty()) {
        return;
    }
    
    publishProbabilityMarkers(); // 发布体素可视化
    publishMapStatistics(); // 发布统计信息
    publishLineSpecificMarkers(); // 发布分线可视化

    // 更新并发布包围盒
    std::vector<AABB> line_boxes = calculateLineBoundingBoxes();
    merged_bounding_boxes_ = mergeBoundingBoxes(line_boxes);
    publishBoundingBoxes(); // <-- 添加这行

}

void PowerLineProbabilityMap::publishProbabilityMarkers() { // 发布体素可视化
    visualization_msgs::MarkerArray marker_array;
    
    int marker_id = 0;
    int published_markers = 0;
    
    for (const auto& [voxel_key, voxel] : voxel_map_) {
        // 只可视化有意义的概率区域，避免显示过多标记
        if (voxel.line_probability < 0.6f) {
            continue;
        }
        
        if (published_markers >= max_visualization_markers_) {
            break;
        }
        
        visualization_msgs::Marker marker = createVoxelMarker(voxel_key, voxel, marker_id++);
        marker_array.markers.push_back(marker);
        published_markers++;
    }
    
    // 发布清除消息（删除旧的标记）
    if (marker_array.markers.empty()) {
        visualization_msgs::Marker delete_marker;
        delete_marker.header.frame_id = frame_id_;
        delete_marker.header.stamp = ros::Time::now();
        delete_marker.ns = "probability_voxels";
        delete_marker.action = visualization_msgs::Marker::DELETEALL;
        marker_array.markers.push_back(delete_marker);
    }
    
    prob_map_pub_.publish(marker_array);
    
    ROS_DEBUG("发布了 %d 个概率地图可视化标记", published_markers);
}

void PowerLineProbabilityMap::publishMapStatistics() { // 发布统计信息
    visualization_msgs::Marker text_marker;
    text_marker.header.frame_id = frame_id_;
    text_marker.header.stamp = ros::Time::now();
    text_marker.ns = "map_statistics";
    text_marker.id = 0;
    text_marker.type = visualization_msgs::Marker::TEXT_VIEW_FACING;
    text_marker.action = visualization_msgs::Marker::ADD;
    
    // 位置设置在地图上方
    text_marker.pose.position.x = 0.0;
    text_marker.pose.position.y = 0.0;
    text_marker.pose.position.z = bounds_.z_max + 2.0f;
    text_marker.pose.orientation.w = 1.0;
    
    // 文本属性
    text_marker.scale.z = 1.0;  // 文本大小
    text_marker.color.r = 1.0;
    text_marker.color.g = 1.0;
    text_marker.color.b = 1.0;
    text_marker.color.a = 1.0;
    text_marker.lifetime = ros::Duration(visualization_duration_);
    
    // 计算统计信息
    int total_voxels, high_prob_voxels;
    float avg_probability;
    getMapStatistics(total_voxels, high_prob_voxels, avg_probability);
    
    // 构建文本内容
    std::ostringstream ss;
    ss << "概率地图统计信息\n";
    ss << "总体素数: " << total_voxels << "\n";
    ss << "高概率体素: " << high_prob_voxels << "\n";
    ss << "平均概率: " << std::fixed << std::setprecision(3) << avg_probability << "\n";
    ss << "覆盖率: " << std::fixed << std::setprecision(1) 
       << (total_voxels > 0 ? 100.0 * high_prob_voxels / total_voxels : 0.0) << "%";
    
    text_marker.text = ss.str();
    
    map_info_pub_.publish(text_marker);
}

std_msgs::ColorRGBA PowerLineProbabilityMap::probabilityToColor(
    float probability, float confidence) const { // 概率转颜色
    
    std_msgs::ColorRGBA color;
    
    // 颜色映射：蓝色(低概率) → 绿色(中概率) → 红色(高概率)
    if (probability < 0.5f) {
        // 蓝色到绿色渐变
        float ratio = probability * 2.0f;  // [0,0.5] → [0,1]
        color.r = 0.0f;
        color.g = ratio;
        color.b = 1.0f - ratio;
    } else {
        // 绿色到红色渐变
        float ratio = (probability - 0.5f) * 2.0f;  // [0.5,1] → [0,1]
        color.r = ratio;
        color.g = 1.0f - ratio;
        color.b = 0.0f;
    }
    
    // 透明度基于置信度和概率，避免遮挡其他可视化
    color.a = 0.2f + confidence * 0.5f;  // [0.2, 0.7] 范围
    
    return color;
}

visualization_msgs::Marker PowerLineProbabilityMap::createVoxelMarker(
    const VoxelKey& voxel_key, const PowerLineVoxel& voxel, int marker_id) const { // 创建体素Marker
    
    visualization_msgs::Marker marker;
    marker.header.frame_id = frame_id_;
    marker.header.stamp = ros::Time::now();
    marker.ns = "probability_voxels";
    marker.id = marker_id;
    marker.type = visualization_msgs::Marker::CUBE;
    marker.action = visualization_msgs::Marker::ADD;
    
    // 体素位置
    Eigen::Vector3f world_pos = voxelToPoint(voxel_key);
    marker.pose.position.x = world_pos.x();
    marker.pose.position.y = world_pos.y();
    marker.pose.position.z = world_pos.z();
    marker.pose.orientation.w = 1.0;
    
    // 体素大小
    marker.scale.x = voxel_size_;
    marker.scale.y = voxel_size_;
    marker.scale.z = voxel_size_;
    
    // 根据概率和置信度设置颜色
    marker.color = probabilityToColor(voxel.line_probability, voxel.confidence);
    
    // 生命周期
    marker.lifetime = ros::Duration(visualization_duration_);
    
    return marker;
}

void PowerLineProbabilityMap::publishLineSpecificMarkers() { // 发布分线可视化
    visualization_msgs::MarkerArray marker_array;
    int marker_id = 0;
    
    // 为每条活跃电力线生成可视化
    for (const auto& [line_id, line_map] : line_specific_maps_) {
        if (!isLineActive(line_id)) continue;
        
        std_msgs::ColorRGBA line_color = getLineColor(line_id);
        int line_marker_count = 0;
        
        // 为该电力线的高概率体素创建标记
        for (const auto& [voxel_key, voxel] : line_map) {
            if (voxel.line_probability < 0.6f) continue; // 只显示较高概率的体素
            
            if (line_marker_count >= max_visualization_markers_) break; // 限制每条线的标记数量
            
            visualization_msgs::Marker marker;
            marker.header.frame_id = frame_id_;
            marker.header.stamp = ros::Time::now();
            marker.ns = "line_" + std::to_string(line_id);
            marker.id = marker_id++;
            marker.type = visualization_msgs::Marker::CUBE;
            marker.action = visualization_msgs::Marker::ADD;
            
            // 体素位置
            Eigen::Vector3f world_pos = voxelToPoint(voxel_key);
            marker.pose.position.x = world_pos.x();
            marker.pose.position.y = world_pos.y();
            marker.pose.position.z = world_pos.z();
            marker.pose.orientation.w = 1.0;
            
            // 体素大小
            marker.scale.x = voxel_size_ * 0.8f; // 稍小一点避免重叠
            marker.scale.y = voxel_size_ * 0.8f;
            marker.scale.z = voxel_size_ * 0.8f;
            
            // 颜色：基于线的颜色，透明度基于概率
            marker.color = line_color;
            marker.color.a = voxel.line_probability * 0.8f;
            
            marker.lifetime = ros::Duration(visualization_duration_);
            marker_array.markers.push_back(marker);
            line_marker_count++;
        }
        
        // 为每条电力线添加ID文本标记
        auto region_it = line_regions_.find(line_id);
        if (region_it != line_regions_.end()) {
            visualization_msgs::Marker text_marker;
            text_marker.header.frame_id = frame_id_;
            text_marker.header.stamp = ros::Time::now();
            text_marker.ns = "line_text";
            text_marker.id = marker_id++;
            text_marker.type = visualization_msgs::Marker::TEXT_VIEW_FACING;
            text_marker.action = visualization_msgs::Marker::ADD;
            
            // 文本位置：电力线区域中心上方
            text_marker.pose.position.x = region_it->second.region_center.x();
            text_marker.pose.position.y = region_it->second.region_center.y();
            text_marker.pose.position.z = region_it->second.region_center.z() + 1.0f;
            text_marker.pose.orientation.w = 1.0;
            
            // 文本属性
            text_marker.scale.z = 0.5f;  // 文本大小
            text_marker.color = line_color;
            text_marker.color.a = 1.0f;
            text_marker.text = "Line " + std::to_string(line_id);
            text_marker.lifetime = ros::Duration(visualization_duration_);
            
            marker_array.markers.push_back(text_marker);
        }
    }
    
    // 如果没有标记，发布清除消息
    if (marker_array.markers.empty()) {
        visualization_msgs::Marker delete_marker;
        delete_marker.header.frame_id = frame_id_;
        delete_marker.header.stamp = ros::Time::now();
        delete_marker.action = visualization_msgs::Marker::DELETEALL;
        marker_array.markers.push_back(delete_marker);
    }
    
    line_specific_pub_.publish(marker_array);
    ROS_DEBUG("发布了 %zu 个分线概率地图标记", marker_array.markers.size());
}

std_msgs::ColorRGBA PowerLineProbabilityMap::getLineColor(int line_id) const { // 获取电力线颜色
    std_msgs::ColorRGBA color;
    
    // 为不同的线分配不同颜色，使用HSV色彩空间确保区分度
    float hue = (line_id * 137.5f) / 360.0f; // 黄金角度分布，确保颜色分散
    hue = hue - std::floor(hue); // 保持在[0,1]范围
    
    // HSV to RGB 转换 (饱和度=1, 明度=1)
    float c = 1.0f; // 饱和度
    float x = c * (1.0f - std::abs(std::fmod(hue * 6.0f, 2.0f) - 1.0f));
    float m = 0.0f;
    
    if (hue < 1.0f/6.0f) {
        color.r = c; color.g = x; color.b = 0;
    } else if (hue < 2.0f/6.0f) {
        color.r = x; color.g = c; color.b = 0;
    } else if (hue < 3.0f/6.0f) {
        color.r = 0; color.g = c; color.b = x;
    } else if (hue < 4.0f/6.0f) {
        color.r = 0; color.g = x; color.b = c;
    } else if (hue < 5.0f/6.0f) {
        color.r = x; color.g = 0; color.b = c;
    } else {
        color.r = c; color.g = 0; color.b = x;
    }
    
    color.a = 0.7f; // 默认透明度
    return color;
}

// ==================== 分线概率地图实现 ====================

std::vector<LineROIInfo> PowerLineProbabilityMap::getAllLineROIs(float threshold, float min_confidence) const {
    std::vector<LineROIInfo> result;
    
    for (const auto& [line_id, region_info] : line_regions_) {
        if (!isLineActive(line_id)) continue;
        
        auto it = line_specific_maps_.find(line_id);
        if (it == line_specific_maps_.end()) continue;
        
        LineROIInfo info;
        info.line_id = line_id;
        info.high_prob_regions = extractHighProbRegions(it->second, threshold);
        info.confidence = calculateLineConfidence(line_id);
        info.is_active = true;
        info.last_update = region_info.last_active_time;
        
        if (info.confidence >= min_confidence && !info.high_prob_regions.empty()) {
            result.push_back(info);
        }
    }
    
    ROS_DEBUG("返回 %zu 条活跃电力线的ROI信息", result.size());
    return result;
}

std::vector<Eigen::Vector3f> PowerLineProbabilityMap::getHighProbabilityRegionsForLine(
    int line_id, float threshold, float min_confidence) const {
    
    if (!isLineActive(line_id)) {
        return std::vector<Eigen::Vector3f>();
    }
    
    auto it = line_specific_maps_.find(line_id);
    if (it == line_specific_maps_.end()) {
        return std::vector<Eigen::Vector3f>();
    }
    
    float confidence = calculateLineConfidence(line_id);
    if (confidence < min_confidence) {
        return std::vector<Eigen::Vector3f>();
    }
    
    return extractHighProbRegions(it->second, threshold);
}

bool PowerLineProbabilityMap::isLineActive(int line_id) const {
    auto it = line_regions_.find(line_id);
    if (it == line_regions_.end()) return false;
    
    ros::Time current_time = ros::Time::now();
    double inactive_duration = (current_time - it->second.last_active_time).toSec();
    
    return inactive_duration <= max_inactive_duration_;
}
//为新检测的电力线分配或复用ID 检查新传入的和已经有的电力线区域的重叠度  如果匹配不到，就创建新的
int PowerLineProbabilityMap::assignLineID(const ReconstructedPowerLine& new_line) {
    // 如果输入线已有ID且在范围内，检查是否可以复用
    if (new_line.line_id >= 0) {
        auto it = line_regions_.find(new_line.line_id);
        if (it != line_regions_.end()) {
            float overlap = calculateSpatialOverlap(new_line, it->second);
            float overlao_rate = calculateSpatialOverlap(new_line, line_specific_maps_[new_line.line_id], 0.6);
            if (overlao_rate > coincidence_rate_threshold_) {
                return new_line.line_id; // 复用现有ID
            }
        }
    }
    
    // 检查是否与现有区域重叠
    for (const auto& [existing_id, region_info] : line_regions_) {
        float overlap = calculateSpatialOverlap(new_line, region_info);
        float overlao_rate = calculateSpatialOverlap(new_line, line_specific_maps_[existing_id], 0.6);
        if (overlao_rate > coincidence_rate_threshold_) {
            return existing_id; // 复用现有ID
        }
    }
    
    // 分配新ID
    int new_id = next_available_line_id_++;
    createNewLineRegion(new_id, new_line);
    return new_id;
}
// 更新分线概率地图
void PowerLineProbabilityMap::updateLineSpecificMaps(const std::vector<ReconstructedPowerLine>& power_lines) {
    // 增加所有电力线的帧计数
    for (auto& [line_id, region_info] : line_regions_) {
        region_info.frames_since_creation++;
    }
        // 增加所有分线体素的未观测帧数（添加这段）
    for (auto& [line_id, line_map] : line_specific_maps_) {
        for (auto& [key, voxel] : line_map) {
            voxel.frames_since_last_observation++;
        }
    }
    // 统计本帧活跃的line_id
    std::unordered_set<int> active_line_ids;
    // 为每条电力线更新其专属地图
    for (const auto& line : power_lines) {
        int assigned_id = assignLineID(line);
        active_line_ids.insert(assigned_id);
        
        // 更新活跃时间
        if (line_regions_.find(assigned_id) != line_regions_.end()) {
            line_regions_[assigned_id].last_active_time = ros::Time::now();
            
            // 检查是否变稳定
            if (!line_regions_[assigned_id].is_stable && 
                line_regions_[assigned_id].frames_since_creation >= min_stable_frames_) {
                line_regions_[assigned_id].is_stable = true;
                ROS_INFO("电力线 %d 变为稳定状态", assigned_id);
            }
        }
        
        // // 更新该电力线的专属概率地图
        // for (size_t i = 0; i < line.fitted_curve_points.size(); ++i) {
        //     const auto& spline_point = line.fitted_curve_points[i];
            
        //     if (!bounds_.isInBounds(spline_point)) continue;
            
        //     Eigen::Vector3f local_direction = computeLocalDirection(line.fitted_curve_points, i);
        //     markLineRegionForSpecificLine(assigned_id, spline_point, local_direction, initial_probability_center_);
        // }
    }
    // 针对每个活跃的line_id，收集其所有片段，做贝叶斯更新
    for (int line_id : active_line_ids) {
        // 收集属于该line_id的所有ReconstructedPowerLine
        std::vector<ReconstructedPowerLine> lines_for_this_id;
        for (const auto& line : power_lines) {
            int assigned_id = assignLineID(line);
            if (assigned_id == line_id) {
                lines_for_this_id.push_back(line);
            }
        }
        // 贝叶斯更新
        TrackerBayesianUpdate(lines_for_this_id, line_specific_maps_[line_id]);
    }

    // 对所有分线概率地图做衰减
    for (auto& [line_id, line_map] : line_specific_maps_) {
        TrackerDecayUnobservedRegions(line_map);
    }
}
//管理电力线生命周期
void PowerLineProbabilityMap::manageLineLifecycles() {
    ros::Time current_time = ros::Time::now();
    
    for (auto it = line_regions_.begin(); it != line_regions_.end();) {
        double inactive_duration = (current_time - it->second.last_active_time).toSec();  //就是这次到上次的检查时间 过长的话就清理
        
        if (inactive_duration > max_inactive_duration_) {
            ROS_INFO("撤销电力线ID %d（非活跃时间: %.1f秒）", it->first, inactive_duration);
            
            // 清理相关数据
            line_specific_maps_.erase(it->first);
            it = line_regions_.erase(it);
        } else {
            ++it;
        }
    }
    
    // 如果电力线数量超限，清理最老的非稳定线
    while (line_regions_.size() > static_cast<size_t>(max_line_count_)) {
        auto oldest_it = line_regions_.end();
        ros::Time oldest_time = ros::Time::now();
        
        for (auto it = line_regions_.begin(); it != line_regions_.end(); ++it) {
            if (!it->second.is_stable && it->second.creation_time < oldest_time) {
                oldest_time = it->second.creation_time;
                oldest_it = it;
            }
        }
        
        if (oldest_it != line_regions_.end()) {
            ROS_INFO("清理最老的非稳定电力线 %d", oldest_it->first);
            line_specific_maps_.erase(oldest_it->first);
            line_regions_.erase(oldest_it);
        } else {
            break; // 所有线都是稳定的，停止清理
        }
    }
}
//创建新的电力线区域
void PowerLineProbabilityMap::createNewLineRegion(int line_id, const ReconstructedPowerLine& line) {
    if (line.fitted_curve_points.empty()) return;
    
    LineRegionInfo region_info;
    region_info.line_id = line_id;
    region_info.start_point = line.fitted_curve_points.front();
    region_info.end_point = line.fitted_curve_points.back();
    
    // 计算区域中心和半径
    Eigen::Vector3f min_point = line.fitted_curve_points[0];
    Eigen::Vector3f max_point = line.fitted_curve_points[0];
    
    for (const auto& point : line.fitted_curve_points) {  //找到整条线最大和最小的xyz，就是勾画出AABB框
        min_point = min_point.cwiseMin(point);
        max_point = max_point.cwiseMax(point);
    }
    
    region_info.region_center = (min_point + max_point) * 0.5f;
    region_info.region_radius = (max_point - min_point).norm() * 0.5f + expansion_radius_;  //电力线长度的一半加扩展半径
    
    line_regions_[line_id] = region_info;
    
    ROS_INFO("创建新电力线区域 ID: %d，中心: (%.2f, %.2f, %.2f)，半径: %.2f", 
             line_id, region_info.region_center.x(), region_info.region_center.y(), 
             region_info.region_center.z(), region_info.region_radius);
}
// 计算传入的电力线与区域的空间重叠度  就是计算电力线拟合曲线点与区域中心的距离
float PowerLineProbabilityMap::calculateSpatialOverlap(const ReconstructedPowerLine& line, 
                                                      const LineRegionInfo& region) const {
    if (line.fitted_curve_points.empty()) return 0.0f;
    
    int points_in_region = 0;
    for (const auto& point : line.fitted_curve_points) {
        float distance = (point - region.region_center).norm();
        if (distance <= region.region_radius) {
            points_in_region++;
        }
    }
    
    return static_cast<float>(points_in_region) / line.fitted_curve_points.size();
}

float PowerLineProbabilityMap::calculateSpatialOverlap(
    const ReconstructedPowerLine& line, 
    const std::unordered_map<VoxelKey, PowerLineVoxel>& line_map,
    float prob_threshold) const
{
    if (line.fitted_curve_points.empty() || line_map.empty()) return 0.0f;

    int points_in_map = 0;
    for (const auto& point : line.fitted_curve_points) {
        VoxelKey key = pointToVoxel(point);
        auto it = line_map.find(key);
        if (it != line_map.end() && it->second.line_probability > prob_threshold) {
            points_in_map++;
        }
    }
    return static_cast<float>(points_in_map) / line.fitted_curve_points.size();
}

std::vector<Eigen::Vector3f> PowerLineProbabilityMap::extractHighProbRegions(
    const std::unordered_map<VoxelKey, PowerLineVoxel>& line_map, float threshold) const {
    
    std::vector<Eigen::Vector3f> high_prob_points;
    
    for (const auto& [key, voxel] : line_map) {
        if (voxel.line_probability > threshold) {
            high_prob_points.push_back(voxelToPoint(key));
        }
    }
    
    return clusterAdjacentRegions(high_prob_points);
}

float PowerLineProbabilityMap::calculateLineConfidence(int line_id) const {
    auto line_it = line_specific_maps_.find(line_id);
    if (line_it == line_specific_maps_.end()) return 0.0f;
    
    if (line_it->second.empty()) return 0.0f;
    
    float total_confidence = 0.0f;
    for (const auto& [key, voxel] : line_it->second) {
        total_confidence += voxel.confidence;
    }
    
    return total_confidence / line_it->second.size();
}

void PowerLineProbabilityMap::markLineRegionForSpecificLine(int line_id, 
                                                           const Eigen::Vector3f& spline_point,
                                                           const Eigen::Vector3f& direction,
                                                           float initial_probability) {
    // 计算需要检查的体素范围
    int radius_in_voxels = static_cast<int>(std::ceil(expansion_radius_ / voxel_size_));
    VoxelKey center_key = pointToVoxel(spline_point);
    
    // 获取或创建该电力线的专属地图
    auto& line_map = line_specific_maps_[line_id];
    
    // 遍历周围体素
    for (int dx = -radius_in_voxels; dx <= radius_in_voxels; ++dx) {
        for (int dy = -radius_in_voxels; dy <= radius_in_voxels; ++dy) {
            for (int dz = -radius_in_voxels; dz <= radius_in_voxels; ++dz) {
                VoxelKey key(center_key.x + dx, center_key.y + dy, center_key.z + dz);
                Eigen::Vector3f voxel_center = voxelToPoint(key);
                
                if (!bounds_.isInBounds(voxel_center)) continue;
                
                float distance = (voxel_center - spline_point).norm();
                
                if (distance <= expansion_radius_) {
                    auto& voxel = line_map[key];
                    
                    float probability = calculateInitialProbability(distance);
                    voxel.line_probability = std::max(voxel.line_probability, probability);
                    voxel.observation_count++;
                    voxel.updateConfidence();
                    voxel.frames_since_last_observation = 0;
                    voxel.last_update_time = ros::Time::now();
                }
            }
        }
    }
}


std::vector<AABB> PowerLineProbabilityMap::calculateLineBoundingBoxes() const {
    std::vector<AABB> boxes;
    
    for (const auto& [line_id, line_map] : line_specific_maps_) {
        if (!isLineActive(line_id) || line_map.empty()) continue;
        
        AABB box;
        box.line_id = line_id;
        
        // 遍历该电力线的所有高概率体素
        for (const auto& [voxel_key, voxel] : line_map) {
            if (voxel.line_probability > 0.5f) { // 只考虑高概率体素
                Eigen::Vector3f voxel_center = voxelToPoint(voxel_key);
                // 扩展半个体素确保完全包含
                Eigen::Vector3f half_voxel(voxel_size_/2, voxel_size_/2, voxel_size_/2);
                box.expandToInclude(voxel_center - half_voxel);
                box.expandToInclude(voxel_center + half_voxel);
            }
        }
        
        // 只有包围盒有效才添加
        if (box.min_point.x() < box.max_point.x()) {
            boxes.push_back(box);
        }
    }
    
    ROS_DEBUG("计算了 %zu 个电力线包围盒", boxes.size());
    return boxes;
}

std::vector<AABB> PowerLineProbabilityMap::mergeBoundingBoxes(const std::vector<AABB>& boxes) const {
    if (boxes.empty()) return boxes;
    
    std::vector<AABB> result;
    std::vector<bool> merged(boxes.size(), false);
    
    for (size_t i = 0; i < boxes.size(); ++i) {
        if (merged[i]) continue;
        
        AABB current_box = boxes[i];
        merged[i] = true;
        
        // 检查是否可以与其他包围盒合并
        bool found_overlap = true;
        while (found_overlap) {
            found_overlap = false;
            for (size_t j = 0; j < boxes.size(); ++j) {
                if (merged[j]) continue;
                
                if (current_box.overlaps(boxes[j])) {
                    current_box = current_box.merge(boxes[j]);
                    merged[j] = true;
                    found_overlap = true;
                }
            }
        }
        
        result.push_back(current_box);
    }
    
    ROS_DEBUG("合并后包围盒数量: %zu -> %zu", boxes.size(), result.size());
    return result;
}

void PowerLineProbabilityMap::publishBoundingBoxes() {
    visualization_msgs::MarkerArray marker_array;
    
    for (size_t i = 0; i < merged_bounding_boxes_.size(); ++i) {
        const auto& box = merged_bounding_boxes_[i];
        
        visualization_msgs::Marker marker;
        marker.header.frame_id = frame_id_;
        marker.header.stamp = ros::Time::now();
        marker.ns = "bounding_boxes";
        marker.id = i;
        marker.type = visualization_msgs::Marker::CUBE;
        marker.action = visualization_msgs::Marker::ADD;
        
        // 包围盒中心
        Eigen::Vector3f center = box.getCenter();
        marker.pose.position.x = center.x();
        marker.pose.position.y = center.y();
        marker.pose.position.z = center.z();
        marker.pose.orientation.w = 1.0;
        
        // 包围盒尺寸
        Eigen::Vector3f size = box.getSize();
        marker.scale.x = size.x();
        marker.scale.y = size.y();
        marker.scale.z = size.z();
        
        // 包围盒颜色：半透明红色
        marker.color.r = 1.0f;
        marker.color.g = 0.0f;
        marker.color.b = 0.0f;
        marker.color.a = 0.3f;
        
        marker.lifetime = ros::Duration(visualization_duration_);
        marker_array.markers.push_back(marker);
        
        // 添加文本标签
        visualization_msgs::Marker text_marker;
        text_marker.header.frame_id = frame_id_;
        text_marker.header.stamp = ros::Time::now();
        text_marker.ns = "box_labels";
        text_marker.id = i;
        text_marker.type = visualization_msgs::Marker::TEXT_VIEW_FACING;
        text_marker.action = visualization_msgs::Marker::ADD;
        
        text_marker.pose.position.x = center.x();
        text_marker.pose.position.y = center.y();
        text_marker.pose.position.z = center.z() + size.z()/2 + 0.5f;
        text_marker.pose.orientation.w = 1.0;
        
        text_marker.scale.z = 0.5f;
        text_marker.color.r = 1.0f;
        text_marker.color.g = 1.0f;
        text_marker.color.b = 1.0f;
        text_marker.color.a = 1.0f;
        text_marker.text = "ROI_" + std::to_string(i);
        text_marker.lifetime = ros::Duration(visualization_duration_);
        
        marker_array.markers.push_back(text_marker);
    }
    
    // 清除旧标记
    if (marker_array.markers.empty()) {
        visualization_msgs::Marker delete_marker;
        delete_marker.header.frame_id = frame_id_;
        delete_marker.header.stamp = ros::Time::now();
        delete_marker.action = visualization_msgs::Marker::DELETEALL;
        marker_array.markers.push_back(delete_marker);
    }
    
    bounding_box_pub_.publish(marker_array);
    ROS_DEBUG("发布了 %zu 个包围盒标记", merged_bounding_boxes_.size());
}
// roi裁减点云
pcl::PointCloud<pcl::PointXYZI>::Ptr PowerLineProbabilityMap::processEnvironmentPointCloud(
    const pcl::PointCloud<pcl::PointXYZI>::Ptr& input_cloud) {
    
    pcl::PointCloud<pcl::PointXYZI>::Ptr result_cloud(new pcl::PointCloud<pcl::PointXYZI>);
    
    if (merged_bounding_boxes_.empty() || !input_cloud || input_cloud->empty()) {
        ROS_WARN("包围盒为空或输入点云无效，返回空点云");
        return result_cloud;
    }
    
    // 对每个包围盒进行裁剪并合并结果
    for (const auto& box : merged_bounding_boxes_) {
        pcl::CropBox<pcl::PointXYZI> crop_filter;
        crop_filter.setInputCloud(input_cloud);
        
        // 设置裁剪范围
        Eigen::Vector4f min_point(box.min_point.x(), box.min_point.y(), box.min_point.z(), 1.0f);
        Eigen::Vector4f max_point(box.max_point.x(), box.max_point.y(), box.max_point.z(), 1.0f);
        crop_filter.setMin(min_point);
        crop_filter.setMax(max_point);
        
        pcl::PointCloud<pcl::PointXYZI>::Ptr cropped_segment(new pcl::PointCloud<pcl::PointXYZI>);
        crop_filter.filter(*cropped_segment);
        
        // 合并到结果点云
        *result_cloud += *cropped_segment;
    }
    
    // 发布可视化
    publishCroppedPointCloud(result_cloud);
    
    ROS_INFO("点云裁剪完成: %zu -> %zu 点", input_cloud->size(), result_cloud->size());
    return result_cloud;
}

pcl::PointCloud<pcl::PointXYZI>::Ptr PowerLineProbabilityMap::processEnvironmentPointCloud(
    const sensor_msgs::PointCloud2::ConstPtr& cloud_msg) {
    
    // 转换ROS消息到PCL
    pcl::PointCloud<pcl::PointXYZI>::Ptr pcl_cloud(new pcl::PointCloud<pcl::PointXYZI>);
    pcl::fromROSMsg(*cloud_msg, *pcl_cloud);
    
    return processEnvironmentPointCloud(pcl_cloud);
}

void PowerLineProbabilityMap::publishCroppedPointCloud(const pcl::PointCloud<pcl::PointXYZI>::Ptr& cloud) {
    if (!enable_visualization_ || !cloud || cloud->empty()) return;
    
    sensor_msgs::PointCloud2 cloud_msg;
    pcl::toROSMsg(*cloud, cloud_msg);
    cloud_msg.header.frame_id = frame_id_;
    cloud_msg.header.stamp = ros::Time::now();
    
    cropped_cloud_pub_.publish(cloud_msg);
    ROS_DEBUG("发布裁剪后点云，点数: %zu", cloud->size());
}


// ==================== 跟踪器支持接口实现 ====================
// 根据样条的位置，查询该位置的概率，取均值归一化返回结果
float PowerLineProbabilityMap::validateDetectionCredibility(
    const ReconstructedPowerLine& detection, float min_probability) const {
    
    if (detection.fitted_curve_points.empty()) {
        return 0.0f;
    }
    
    float total_probability = 0.0f;
    int valid_points = 0;
    
    for (const auto& point : detection.fitted_curve_points) {
        if (bounds_.isInBounds(point)) {
            float prob = queryProbabilityAtPosition(point);
            total_probability += prob;
            valid_points++;
        }
    }
    
    if (valid_points == 0) {
        return 0.0f;
    }
    
    float avg_probability = total_probability / valid_points;
    
    // 转换为可信度分数
    if (avg_probability < min_probability) {
        return 0.0f;
    }
    
    // 归一化到[0,1]范围
    return std::min(1.0f, (avg_probability - min_probability) / (1.0f - min_probability));
}

float PowerLineProbabilityMap::getPredictionRegionProbability(
    const Eigen::Vector3f& center_point, float search_radius) const {
    
    if (!bounds_.isInBounds(center_point)) {
        return background_probability_;
    }
    
    float total_probability = 0.0f;
    int valid_voxels = 0;
    
    // 计算搜索范围内的体素
    int radius_in_voxels = static_cast<int>(std::ceil(search_radius / voxel_size_));
    VoxelKey center_key = pointToVoxel(center_point);
    
    for (int dx = -radius_in_voxels; dx <= radius_in_voxels; ++dx) {
        for (int dy = -radius_in_voxels; dy <= radius_in_voxels; ++dy) {
            for (int dz = -radius_in_voxels; dz <= radius_in_voxels; ++dz) {
                VoxelKey key(center_key.x + dx, center_key.y + dy, center_key.z + dz);
                Eigen::Vector3f voxel_center = voxelToPoint(key);
                
                if (!bounds_.isInBounds(voxel_center)) continue;
                
                float distance = (voxel_center - center_point).norm();
                if (distance <= search_radius) {
                    auto it = voxel_map_.find(key);
                    if (it != voxel_map_.end()) {
                        total_probability += it->second.line_probability;
                    } else {
                        total_probability += background_probability_;
                    }
                    valid_voxels++;
                }
            }
        }
    }
    
    return valid_voxels > 0 ? total_probability / valid_voxels : background_probability_;
}

std::vector<float> PowerLineProbabilityMap::batchQueryProbability(
    const std::vector<Eigen::Vector3f>& points) const {
    
    std::vector<float> probabilities;
    probabilities.reserve(points.size());
    
    for (const auto& point : points) {
        probabilities.push_back(queryProbabilityAtPosition(point));
    }
    
    return probabilities;
}

//跟踪器分线函数
void PowerLineProbabilityMap::TrackerBayesianUpdate(const std::vector<ReconstructedPowerLine>& power_lines,
                                                    std::unordered_map<VoxelKey, PowerLineVoxel>& line_tracker_map){
    // 标记当前帧检测到的区域
    std::unordered_set<VoxelKey> observed_voxels;
    // 判断是否需要同时更新全局地图
    bool update_global = (&line_tracker_map != &voxel_map_);
    
    for (const auto& line : power_lines) {
        for (const auto& spline_point : line.fitted_curve_points) {
            if (!bounds_.isInBounds(spline_point)) continue;
            
            // 计算需要检查的体素范围
            int radius_in_voxels = static_cast<int>(std::ceil(expansion_radius_ / voxel_size_));
            VoxelKey center_key = pointToVoxel(spline_point);
            
            // 遍历周围体素
            for (int dx = -radius_in_voxels; dx <= radius_in_voxels; ++dx) {
                for (int dy = -radius_in_voxels; dy <= radius_in_voxels; ++dy) {
                    for (int dz = -radius_in_voxels; dz <= radius_in_voxels; ++dz) {
                        VoxelKey key(center_key.x + dx, center_key.y + dy, center_key.z + dz);
                        Eigen::Vector3f voxel_center = voxelToPoint(key);
                        
                        if (!bounds_.isInBounds(voxel_center)) continue;
                        
                        // 计算体素中心到样条点的距离
                        float distance = (voxel_center - spline_point).norm();
                        
                        if (distance <= expansion_radius_) {
                            observed_voxels.insert(key);
                            auto& voxel = line_tracker_map[key];
                            
                            // 贝叶斯更新：检测命中
                            voxel.line_probability = updateBayesian(
                                voxel.line_probability, hit_likelihood_, true);
                            
                            voxel.observation_count++;
                            voxel.updateConfidence();
                            voxel.frames_since_last_observation = 0;
                            voxel.last_update_time = ros::Time::now();

                            // 同时更新全局地图
                            if (update_global) {
                                auto& global_voxel = voxel_map_[key];
                                global_voxel.line_probability = updateBayesian(
                                    global_voxel.line_probability, hit_likelihood_, true);
                                global_voxel.observation_count++;
                                global_voxel.updateConfidence();
                                global_voxel.frames_since_last_observation = 0;
                                global_voxel.last_update_time = ros::Time::now();
                            }

                            
                        }
                    }
                }
            }
        }
    }
    
    // 对于历史存在但当前未检测到的区域，降低概率
    for (auto& [key, voxel] : line_tracker_map) {
        if (observed_voxels.find(key) == observed_voxels.end()) {
            // 贝叶斯更新：检测丢失
            voxel.line_probability = updateBayesian(
                voxel.line_probability, miss_likelihood_, false);
        }
    }

    // 同步更新全局地图中未观测区域
    if (update_global) {
        for (auto& [key, voxel] : voxel_map_) {
            if (observed_voxels.find(key) == observed_voxels.end()) {
                // 贝叶斯更新：检测丢失
                voxel.line_probability = updateBayesian(
                    voxel.line_probability, miss_likelihood_, false);
            }
        }
    }
    
}


// 跟踪器分线衰减未观测区域
void PowerLineProbabilityMap::TrackerDecayUnobservedRegions(std::unordered_map<VoxelKey, PowerLineVoxel>& line_tracker_map) { 
    // 判断是否需要同时更新全局地图
    bool update_global = (&line_tracker_map != &voxel_map_);
        // 处理传入地图的衰减
    auto it = line_tracker_map.begin();
    while (it != line_tracker_map.end()) {
        auto& voxel = it->second;
        if (voxel.frames_since_last_observation > max_frames_without_observation_) {
            // 向不确定状态衰减
            float target = 0.5f;  // 不确定状态
            voxel.line_probability = target + (voxel.line_probability - target) * decay_rate_;
            
            // 置信度也逐渐衰减
            voxel.confidence *= decay_rate_;
            
            // 如果概率接近0.5且置信度很低，直接移除体素
            if (std::abs(voxel.line_probability - 0.5f) < 0.05f && voxel.confidence < 0.1f) {
                it = line_tracker_map.erase(it);
                continue;
            }
        }
        ++it;
    }
    
    // 同步更新全局地图的衰减
    if (update_global) {
        auto it_global = voxel_map_.begin();
        while (it_global != voxel_map_.end()) {
            auto& voxel = it_global->second;
            if (voxel.frames_since_last_observation > max_frames_without_observation_) {
                float target = 0.5f;
                voxel.line_probability = target + (voxel.line_probability - target) * decay_rate_;
                voxel.confidence *= decay_rate_;
                
                if (std::abs(voxel.line_probability - 0.5f) < 0.05f && voxel.confidence < 0.1f) {
                    it_global = voxel_map_.erase(it_global);
                    continue;
                }
            }
            ++it_global;
        }
    }
    
    
    // for (auto& [key, voxel] : line_tracker_map) {
    //     if (voxel.frames_since_last_observation > max_frames_without_observation_) {
    //         // 向不确定状态衰减
    //         float target = 0.5f;  // 不确定状态
    //         voxel.line_probability = target + (voxel.line_probability - target) * decay_rate_;
            
    //         // 置信度也逐渐衰减
    //         voxel.confidence *= decay_rate_;


    //     }
    // }
}

