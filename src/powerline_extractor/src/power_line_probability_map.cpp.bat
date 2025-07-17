#include "power_line_probability_map.h"

#include <pcl_conversions/pcl_conversions.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/kdtree/kdtree.h>
#include <pcl/segmentation/extract_clusters.h>
#include <algorithm>
#include <cmath>

PowerLineProbabilityMap::PowerLineProbabilityMap(ros::NodeHandle& nh) : nh_(nh) {
    // 读取参数
    loadParameters();
    
    // 初始化ROS发布器
    prob_map_vis_pub_ = nh_.advertise<visualization_msgs::MarkerArray>(
        "power_line_probability_map/visualization", 1);
    roi_vis_pub_ = nh_.advertise<visualization_msgs::MarkerArray>(
        "power_line_probability_map/roi_regions", 1);
    roi_pointcloud_pub_ = nh_.advertise<sensor_msgs::PointCloud2>(
        "power_line_probability_map/roi_pointcloud", 1);
    
    // 初始化时间
    last_visualization_time_ = ros::Time::now();
    
    ROS_INFO("PowerLineProbabilityMap 初始化完成");
    ROS_INFO("体素大小: %.3f m", voxel_size_);
    ROS_INFO("电力线扩展半径: %.3f m", expansion_radius_);
    ROS_INFO("空间范围: [%.1f,%.1f] x [%.1f,%.1f] x [%.1f,%.1f]", 
             space_bounds_.x_min, space_bounds_.x_max,
             space_bounds_.y_min, space_bounds_.y_max,
             space_bounds_.z_min, space_bounds_.z_max);
}

PowerLineProbabilityMap::~PowerLineProbabilityMap() {
    ROS_INFO("PowerLineProbabilityMap 析构，总共创建 %d 个体素，执行 %d 次更新", 
             total_voxels_created_, total_updates_performed_);
}

void PowerLineProbabilityMap::loadParameters() {
    // 体素化参数
    nh_.param("probability_map/voxel_size", voxel_size_, 0.1f);
    nh_.param("probability_map/expansion_radius", expansion_radius_, 0.15f);
    
    // 贝叶斯更新参数
    nh_.param("probability_map/prob_hit", prob_hit_, 0.8f);
    nh_.param("probability_map/prob_miss", prob_miss_, 0.3f);
    nh_.param("probability_map/decay_rate", decay_rate_, 0.98f);
    nh_.param("probability_map/max_frames_without_observation", max_frames_without_observation_, 10);
    
    // 概率阈值参数
    nh_.param("probability_map/initial_line_probability", initial_line_probability_, 0.75f);
    nh_.param("probability_map/background_probability", background_probability_, 0.1f);
    nh_.param("probability_map/uncertainty_probability", uncertainty_probability_, 0.5f);
    
    // 可视化参数
    nh_.param("probability_map/enable_visualization", enable_visualization_, true);
    nh_.param("probability_map/visualization_update_rate", visualization_update_rate_, 2.0f);
    nh_.param("probability_map/marker_lifetime", marker_lifetime_, 5.0f);
    nh_.param("probability_map/max_markers_per_publish", max_markers_per_publish_, 2000);
    
    // 空间参数
    nh_.param("probability_map/space_x_min", space_bounds_.x_min, -35.0f);
    nh_.param("probability_map/space_x_max", space_bounds_.x_max, 35.0f);
    nh_.param("probability_map/space_y_min", space_bounds_.y_min, -35.0f);
    nh_.param("probability_map/space_y_max", space_bounds_.y_max, 35.0f);
    nh_.param("probability_map/space_z_min", space_bounds_.z_min, 0.0f);
    nh_.param("probability_map/space_z_max", space_bounds_.z_max, 35.0f);
    nh_.param("probability_map/frame_id", frame_id_, std::string("map"));
}

void PowerLineProbabilityMap::initializeProbabilityMap(const std::vector<ReconstructedPowerLine>& lines) {
    ROS_INFO("初始化概率地图，输入电力线数量: %zu", lines.size());
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // 清空现有地图
    voxel_map_.clear();
    total_voxels_created_ = 0;
    
    // 为每条电力线创建概率区域
    for (const auto& line : lines) {
        if (line.fitted_curve_points.empty()) {
            ROS_WARN("电力线 %d 的拟合曲线点为空，跳过", line.line_id);
            continue;
        }
        
        // 沿着三次样条点标记概率区域
        for (size_t i = 0; i < line.fitted_curve_points.size(); ++i) {
            const auto& spline_point = line.fitted_curve_points[i];
            
            // 计算局部方向
            Eigen::Vector3f direction;
            if (i == 0 && line.fitted_curve_points.size() > 1) {
                direction = (line.fitted_curve_points[i + 1] - spline_point).normalized();
            } else if (i == line.fitted_curve_points.size() - 1 && i > 0) {
                direction = (spline_point - line.fitted_curve_points[i - 1]).normalized();
            } else if (i > 0 && i < line.fitted_curve_points.size() - 1) {
                direction = (line.fitted_curve_points[i + 1] - line.fitted_curve_points[i - 1]).normalized();
            } else {
                direction = line.main_direction;  // 使用整体方向作为备选
            }
            
            // 标记圆柱体区域
            markLineRegion(spline_point, direction, initial_line_probability_);
        }
        
        ROS_INFO("电力线 %d 初始化完成，样条点数: %zu", line.line_id, line.fitted_curve_points.size());
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    
    ROS_INFO("概率地图初始化完成，创建体素数量: %d，耗时: %ld ms", 
             total_voxels_created_, duration.count());
    
    // 可视化初始结果
    if (enable_visualization_) {
        visualizeProbabilityMap();
    }
}

void PowerLineProbabilityMap::updateProbabilityMap(const std::vector<ReconstructedPowerLine>& lines) {
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // 增加所有体素的未观测帧数
    for (auto& [key, voxel] : voxel_map_) {
        voxel.frames_since_last_observation++;
    }
    
    // 对新检测到的电力线进行贝叶斯更新
    for (const auto& line : lines) {
        if (line.fitted_curve_points.empty()) continue;
        
        // 沿着样条点更新概率
        for (size_t i = 0; i < line.fitted_curve_points.size(); ++i) {
            const auto& spline_point = line.fitted_curve_points[i];
            
            // 计算局部方向
            Eigen::Vector3f direction;
            if (i == 0 && line.fitted_curve_points.size() > 1) {
                direction = (line.fitted_curve_points[i + 1] - spline_point).normalized();
            } else if (i == line.fitted_curve_points.size() - 1 && i > 0) {
                direction = (spline_point - line.fitted_curve_points[i - 1]).normalized();
            } else if (i > 0 && i < line.fitted_curve_points.size() - 1) {
                direction = (line.fitted_curve_points[i + 1] - line.fitted_curve_points[i - 1]).normalized();
            } else {
                direction = line.main_direction;
            }
            
            // 在扩展半径内更新概率
            auto circular_points = generateCircularPoints(spline_point, direction, expansion_radius_, 16);
            
            for (const auto& point : circular_points) {
                if (!space_bounds_.isInBounds(point)) continue;
                
                VoxelKey key = worldToVoxel(point);
                auto& voxel = voxel_map_[key];
                
                // 计算距离中心线的距离
                float distance = (point - spline_point).norm();
                float likelihood = calculateLikelihood(distance);
                
                // 贝叶斯更新
                voxel.line_probability = updateBayesian(voxel.line_probability, likelihood, true);
                voxel.local_direction = direction;
                voxel.frames_since_last_observation = 0;
                voxel.observation_count++;
                voxel.last_update_time = ros::Time::now();
                
                // 更新置信度
                updateConfidence(voxel);
            }
        }
    }
    
    // 衰减未观测区域
    decayUnobservedRegions();
    
    total_updates_performed_++;
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    
    ROS_DEBUG("概率地图更新完成，处理电力线数量: %zu，耗时: %ld ms", lines.size(), duration.count());
    
    // 可视化更新结果
    if (enable_visualization_) {
        ros::Time current_time = ros::Time::now();
        if ((current_time - last_visualization_time_).toSec() > 1.0 / visualization_update_rate_) {
            visualizeProbabilityMap();
            last_visualization_time_ = current_time;
        }
    }
}

std::vector<ROIRegion> PowerLineProbabilityMap::getHighProbabilityRegions(
    float probability_threshold, float confidence_threshold) const {
    
    // 收集高概率体素
    std::vector<VoxelKey> high_prob_voxels;
    for (const auto& [key, voxel] : voxel_map_) {
        if (voxel.line_probability > probability_threshold && 
            voxel.confidence > confidence_threshold) {
            high_prob_voxels.push_back(key);
        }
    }
    
    ROS_DEBUG("找到 %zu 个高概率体素 (概率>%.2f, 置信度>%.2f)", 
              high_prob_voxels.size(), probability_threshold, confidence_threshold);
    
    // 聚类相邻的高概率区域
    return clusterHighProbabilityRegions(high_prob_voxels);
}

pcl::PointCloud<pcl::PointXYZI>::Ptr PowerLineProbabilityMap::extractROIPointCloud(
    const pcl::PointCloud<pcl::PointXYZI>::Ptr& input_cloud,
    const std::vector<ROIRegion>& roi_regions,
    float expansion_factor) const {
    
    if (roi_regions.empty()) {
        ROS_WARN("ROI区域为空，返回空点云");
        return pcl::PointCloud<pcl::PointXYZI>::Ptr(new pcl::PointCloud<pcl::PointXYZI>());
    }
    
    pcl::PointCloud<pcl::PointXYZI>::Ptr roi_cloud(new pcl::PointCloud<pcl::PointXYZI>());
    
    // 为每个点检查是否在任何ROI区域内
    for (const auto& point : input_cloud->points) {
        Eigen::Vector3f pt(point.x, point.y, point.z);
        
        bool in_roi = false;
        for (const auto& roi : roi_regions) {
            float distance = (pt - roi.center).norm();
            if (distance <= roi.radius * expansion_factor) {
                in_roi = true;
                break;
            }
        }
        
        if (in_roi) {
            roi_cloud->points.push_back(point);
        }
    }
    
    roi_cloud->width = roi_cloud->size();
    roi_cloud->height = 1;
    roi_cloud->is_dense = true;
    
    ROS_DEBUG("ROI裁剪：输入 %zu 个点，输出 %zu 个点 (%.1f%%)", 
              input_cloud->size(), roi_cloud->size(), 
              100.0 * roi_cloud->size() / std::max(1ul, input_cloud->size()));
    
    // 发布ROI点云用于可视化
    if (enable_visualization_) {
        sensor_msgs::PointCloud2 roi_msg;
        pcl::toROSMsg(*roi_cloud, roi_msg);
        roi_msg.header.frame_id = frame_id_;
        roi_msg.header.stamp = ros::Time::now();
        roi_pointcloud_pub_.publish(roi_msg);
    }
    
    return roi_cloud;
}

std::vector<std::vector<Eigen::Vector3f>> PowerLineProbabilityMap::getHistoricalSplinePoints(
    float min_confidence) const {
    
    std::vector<std::vector<Eigen::Vector3f>> historical_splines;
    std::vector<Eigen::Vector3f> current_spline;
    
    // 遍历高置信度的体素，重构样条线
    std::vector<VoxelKey> sorted_voxels;
    for (const auto& [key, voxel] : voxel_map_) {
        if (voxel.confidence >= min_confidence && voxel.line_probability > 0.7f) {
            sorted_voxels.push_back(key);
        }
    }
    
    // 简单的连续性检查（可以进一步优化）
    if (!sorted_voxels.empty()) {
        std::sort(sorted_voxels.begin(), sorted_voxels.end(), 
                 [this](const VoxelKey& a, const VoxelKey& b) {
                     Eigen::Vector3f pos_a = voxelToWorld(a);
                     Eigen::Vector3f pos_b = voxelToWorld(b);
                     return pos_a.x() < pos_b.x();  // 简单按x坐标排序
                 });
        
        current_spline.clear();
        for (const auto& key : sorted_voxels) {
            current_spline.push_back(voxelToWorld(key));
        }
        
        if (!current_spline.empty()) {
            historical_splines.push_back(current_spline);
        }
    }
    
    ROS_DEBUG("获取历史样条线数量: %zu", historical_splines.size());
    return historical_splines;
}

void PowerLineProbabilityMap::visualizeProbabilityMap() {
    if (!enable_visualization_) return;
    
    visualization_msgs::MarkerArray marker_array;
    int marker_id = 0;
    
    // 限制发布的标记数量
    int published_count = 0;
    
    for (const auto& [key, voxel] : voxel_map_) {
        if (published_count >= max_markers_per_publish_) break;
        
        // 只可视化有意义的概率区域
        if (voxel.line_probability < 0.3f || voxel.line_probability > 0.95f) {
            continue;
        }
        
        auto marker = createVoxelMarker(key, voxel, marker_id++);
        marker_array.markers.push_back(marker);
        published_count++;
    }
    
    // 如果没有标记，发布一个删除所有标记的消息
    if (marker_array.markers.empty()) {
        visualization_msgs::Marker delete_marker;
        delete_marker.header.frame_id = frame_id_;
        delete_marker.header.stamp = ros::Time::now();
        delete_marker.ns = "probability_voxels";
        delete_marker.action = visualization_msgs::Marker::DELETEALL;
        marker_array.markers.push_back(delete_marker);
    }
    
    prob_map_vis_pub_.publish(marker_array);
    ROS_DEBUG("发布概率地图可视化，标记数量: %zu", marker_array.markers.size());
}

void PowerLineProbabilityMap::visualizeROIRegions(const std::vector<ROIRegion>& roi_regions) {
    if (!enable_visualization_) return;
    
    visualization_msgs::MarkerArray marker_array;
    
    for (size_t i = 0; i < roi_regions.size(); ++i) {
        auto marker = createROIMarker(roi_regions[i], static_cast<int>(i));
        marker_array.markers.push_back(marker);
    }
    
    // 如果没有ROI，发布删除消息
    if (marker_array.markers.empty()) {
        visualization_msgs::Marker delete_marker;
        delete_marker.header.frame_id = frame_id_;
        delete_marker.header.stamp = ros::Time::now();
        delete_marker.ns = "roi_regions";
        delete_marker.action = visualization_msgs::Marker::DELETEALL;
        marker_array.markers.push_back(delete_marker);
    }
    
    roi_vis_pub_.publish(marker_array);
    ROS_DEBUG("发布ROI区域可视化，区域数量: %zu", roi_regions.size());
}

float PowerLineProbabilityMap::queryProbabilityAtPosition(const Eigen::Vector3f& position) const {
    if (!space_bounds_.isInBounds(position)) {
        return background_probability_;
    }
    
    VoxelKey key = worldToVoxel(position);
    auto it = voxel_map_.find(key);
    
    if (it != voxel_map_.end()) {
        return it->second.line_probability;
    } else {
        return uncertainty_probability_;
    }
}

void PowerLineProbabilityMap::printMapStatistics() const {
    if (voxel_map_.empty()) {
        ROS_INFO("概率地图统计: 空地图");
        return;
    }
    
    float total_prob = 0.0f;
    float max_prob = 0.0f;
    float min_prob = 1.0f;
    int high_conf_count = 0;
    
    for (const auto& [key, voxel] : voxel_map_) {
        total_prob += voxel.line_probability;
        max_prob = std::max(max_prob, voxel.line_probability);
        min_prob = std::min(min_prob, voxel.line_probability);
        if (voxel.confidence > 0.7f) high_conf_count++;
    }
    
    float avg_prob = total_prob / voxel_map_.size();
    
    ROS_INFO("=== 概率地图统计 ===");
    ROS_INFO("总体素数量: %zu", voxel_map_.size());
    ROS_INFO("平均概率: %.3f", avg_prob);
    ROS_INFO("概率范围: [%.3f, %.3f]", min_prob, max_prob);
    ROS_INFO("高置信度体素: %d (%.1f%%)", high_conf_count, 
             100.0 * high_conf_count / voxel_map_.size());
    ROS_INFO("总更新次数: %d", total_updates_performed_);
}

void PowerLineProbabilityMap::clearMap() {
    voxel_map_.clear();
    total_voxels_created_ = 0;
    total_updates_performed_ = 0;
    ROS_INFO("概率地图已清空");
}

// ==================== 私有辅助函数实现 ====================

VoxelKey PowerLineProbabilityMap::worldToVoxel(const Eigen::Vector3f& world_pos) const {
    return VoxelKey(
        static_cast<int>(std::floor(world_pos.x() / voxel_size_)),
        static_cast<int>(std::floor(world_pos.y() / voxel_size_)),
        static_cast<int>(std::floor(world_pos.z() / voxel_size_))
    );
}

Eigen::Vector3f PowerLineProbabilityMap::voxelToWorld(const VoxelKey& voxel_key) const {
    return Eigen::Vector3f(
        (voxel_key.x + 0.5f) * voxel_size_,
        (voxel_key.y + 0.5f) * voxel_size_,
        (voxel_key.z + 0.5f) * voxel_size_
    );
}

void PowerLineProbabilityMap::markLineRegion(const Eigen::Vector3f& spline_point, 
                                            const Eigen::Vector3f& direction,
                                            float initial_probability) {
    // 生成圆柱体区域的点
    auto circular_points = generateCircularPoints(spline_point, direction, expansion_radius_, 16);
    
    for (const auto& point : circular_points) {
        if (!space_bounds_.isInBounds(point)) continue;
        
        VoxelKey key = worldToVoxel(point);
        
        // 如果体素不存在，创建新体素
        if (voxel_map_.find(key) == voxel_map_.end()) {
            total_voxels_created_++;
        }
        
        auto& voxel = voxel_map_[key];
        
        // 计算距离中心线的距离来确定概率
        float distance = (point - spline_point).norm();
        float probability = calculateInitialProbability(distance);
        
        voxel.line_probability = std::max(voxel.line_probability, probability);
        voxel.local_direction = direction;
        voxel.observation_count++;
        voxel.last_update_time = ros::Time::now();
        voxel.frames_since_last_observation = 0;
        
        updateConfidence(voxel);
    }
}

float PowerLineProbabilityMap::updateBayesian(float prior_prob, float likelihood, bool positive_evidence) const {
    // 使用对数几率形式避免数值问题
    float log_odds_prior = std::log(prior_prob / (1.0f - prior_prob + 1e-6f));
    
    float evidence_prob = positive_evidence ? likelihood : (1.0f - likelihood);
    float log_odds_evidence = std::log(evidence_prob / (1.0f - evidence_prob + 1e-6f));
    
    float log_odds_posterior = log_odds_prior + log_odds_evidence;
    
    // 转换回概率
    float posterior = 1.0f / (1.0f + std::exp(-log_odds_posterior));
    
    // 限制在合理范围内
    return std::max(0.01f, std::min(0.99f, posterior));
}

float PowerLineProbabilityMap::calculateInitialProbability(float distance_from_centerline) const {
    if (distance_from_centerline < 0.05f) {
        return 0.9f;    // 中心线附近：90%概率
    } else if (distance_from_centerline < 0.1f) {
        return 0.75f;   // 近距离：75%概率  
    } else if (distance_from_centerline < expansion_radius_) {
        return 0.6f;    // 边缘：60%概率
    }
    return background_probability_;
}

float PowerLineProbabilityMap::calculateLikelihood(float distance_from_centerline) const {
    // 距离越近，似然概率越高
    float normalized_distance = distance_from_centerline / expansion_radius_;
    return prob_hit_ * std::exp(-normalized_distance * normalized_distance);
}

void PowerLineProbabilityMap::decayUnobservedRegions() {
    for (auto& [key, voxel] : voxel_map_) {
        if (voxel.frames_since_last_observation > max_frames_without_observation_) {
            // 向不确定状态衰减
            voxel.line_probability = uncertainty_probability_ + 
                (voxel.line_probability - uncertainty_probability_) * decay_rate_;
            
            // 置信度也衰减
            voxel.confidence *= decay_rate_;
        }
    }
}

void PowerLineProbabilityMap::updateConfidence(PowerLineVoxel& voxel) {
    // 基于观测次数计算置信度，但有上限
    voxel.confidence = std::min(1.0f, voxel.observation_count * 0.1f);
}

std::vector<ROIRegion> PowerLineProbabilityMap::clusterHighProbabilityRegions(
    const std::vector<VoxelKey>& high_prob_voxels) const {
    
    std::vector<ROIRegion> roi_regions;
    
    if (high_prob_voxels.empty()) {
        return roi_regions;
    }
    
    // 简单聚类：基于空间距离
    std::vector<bool> clustered(high_prob_voxels.size(), false);
    
    for (size_t i = 0; i < high_prob_voxels.size(); ++i) {
        if (clustered[i]) continue;
        
        // 开始新聚类
        std::vector<VoxelKey> cluster;
        std::queue<size_t> queue;
        queue.push(i);
        clustered[i] = true;
        
        while (!queue.empty()) {
            size_t current_idx = queue.front();
            queue.pop();
            cluster.push_back(high_prob_voxels[current_idx]);
            
            // 查找邻近的未聚类体素
            for (size_t j = 0; j < high_prob_voxels.size(); ++j) {
                if (clustered[j]) continue;
                
                // 检查是否相邻
                const auto& key1 = high_prob_voxels[current_idx];
                const auto& key2 = high_prob_voxels[j];
                
                int dx = std::abs(key1.x - key2.x);
                int dy = std::abs(key1.y - key2.y);
                int dz = std::abs(key1.z - key2.z);
                
                if (dx <= 2 && dy <= 2 && dz <= 2) {  // 相邻或近邻
                    clustered[j] = true;
                    queue.push(j);
                }
            }
        }
        
        // 为聚类创建ROI区域
        if (cluster.size() >= 3) {  // 最小聚类大小
            Eigen::Vector3f center = Eigen::Vector3f::Zero();
            float total_prob = 0.0f;
            float total_conf = 0.0f;
            
            for (const auto& key : cluster) {
                center += voxelToWorld(key);
                const auto& voxel = voxel_map_.at(key);
                total_prob += voxel.line_probability;
                total_conf += voxel.confidence;
            }
            
            center /= cluster.size();
            float avg_prob = total_prob / cluster.size();
            float avg_conf = total_conf / cluster.size();
            
            // 计算聚类半径
            float max_distance = 0.0f;
            for (const auto& key : cluster) {
                float distance = (voxelToWorld(key) - center).norm();
                max_distance = std::max(max_distance, distance);
            }
            
            float radius = std::max(expansion_radius_, max_distance + voxel_size_);
            roi_regions.emplace_back(center, radius, avg_prob, avg_conf);
        }
    }
    
    ROS_DEBUG("聚类结果: %zu 个高概率体素 → %zu 个ROI区域", 
              high_prob_voxels.size(), roi_regions.size());
    
    return roi_regions;
}

std_msgs::ColorRGBA PowerLineProbabilityMap::probabilityToColor(float probability, float confidence) const {
    std_msgs::ColorRGBA color;
    
    // 颜色映射：蓝色(低概率) → 绿色(中概率) → 红色(高概率)
    if (probability < 0.5f) {
        // 蓝色到绿色渐变
        float ratio = probability * 2.0f;
        color.r = 0.0f;
        color.g = ratio;
        color.b = 1.0f - ratio;
    } else {
        // 绿色到红色渐变
        float ratio = (probability - 0.5f) * 2.0f;
        color.r = ratio;
        color.g = 1.0f - ratio;
        color.b = 0.0f;
    }
    
    // 透明度基于置信度
    color.a = 0.2f + confidence * 0.5f;  // [0.2, 0.7] 范围，保持透明
    
    return color;
}

visualization_msgs::Marker PowerLineProbabilityMap::createVoxelMarker(
    const VoxelKey& key, const PowerLineVoxel& voxel, int marker_id) const {
    
    visualization_msgs::Marker marker;
    marker.header.frame_id = frame_id_;
    marker.header.stamp = ros::Time::now();
    marker.ns = "probability_voxels";
    marker.id = marker_id;
    marker.type = visualization_msgs::Marker::CUBE;
    marker.action = visualization_msgs::Marker::ADD;
    
    // 位置
    Eigen::Vector3f world_pos = voxelToWorld(key);
    marker.pose.position.x = world_pos.x();
    marker.pose.position.y = world_pos.y();
    marker.pose.position.z = world_pos.z();
    marker.pose.orientation.w = 1.0;
    
    // 大小
    marker.scale.x = voxel_size_;
    marker.scale.y = voxel_size_;
    marker.scale.z = voxel_size_;
    
    // 颜色
    marker.color = probabilityToColor(voxel.line_probability, voxel.confidence);
    
    // 生命周期
    marker.lifetime = ros::Duration(marker_lifetime_);
    
    return marker;
}

visualization_msgs::Marker PowerLineProbabilityMap::createROIMarker(
    const ROIRegion& roi, int marker_id) const {
    
    visualization_msgs::Marker marker;
    marker.header.frame_id = frame_id_;
    marker.header.stamp = ros::Time::now();
    marker.ns = "roi_regions";
    marker.id = marker_id;
    marker.type = visualization_msgs::Marker::SPHERE;
    marker.action = visualization_msgs::Marker::ADD;
    
    // 位置
    marker.pose.position.x = roi.center.x();
    marker.pose.position.y = roi.center.y();
    marker.pose.position.z = roi.center.z();
    marker.pose.orientation.w = 1.0;
    
    // 大小（直径 = 2 * 半径）
    float diameter = roi.radius * 2.0f;
    marker.scale.x = diameter;
    marker.scale.y = diameter;
    marker.scale.z = diameter;
    
    // 颜色（黄色，半透明）
    marker.color.r = 1.0f;
    marker.color.g = 1.0f;
    marker.color.b = 0.0f;
    marker.color.a = 0.3f;
    
    // 生命周期
    marker.lifetime = ros::Duration(marker_lifetime_);
    
    return marker;
}

Eigen::Vector3f PowerLineProbabilityMap::getPerpendicularVector(
    const Eigen::Vector3f& direction, float radius, float angle) const {
    
    // 找到垂直于direction的两个正交向量
    Eigen::Vector3f arbitrary(1, 0, 0);
    if (std::abs(direction.dot(arbitrary)) > 0.9f) {
        arbitrary = Eigen::Vector3f(0, 1, 0);
    }
    
    Eigen::Vector3f perp1 = direction.cross(arbitrary).normalized();
    Eigen::Vector3f perp2 = direction.cross(perp1).normalized();
    
    // 生成圆周上的点
    return radius * (std::cos(angle) * perp1 + std::sin(angle) * perp2);
}

std::vector<Eigen::Vector3f> PowerLineProbabilityMap::generateCircularPoints(
    const Eigen::Vector3f& center, const Eigen::Vector3f& direction, 
    float radius, int num_points) const {
    
    std::vector<Eigen::Vector3f> points;
    
    // 在半径范围内生成多层圆形点
    int num_layers = static_cast<int>(radius / voxel_size_) + 1;
    
    for (int layer = 0; layer <= num_layers; ++layer) {
        float layer_radius = (layer * radius) / num_layers;
        
        int points_in_layer = layer == 0 ? 1 : num_points;
        
        for (int i = 0; i < points_in_layer; ++i) {
            float angle = 2.0f * M_PI * i / points_in_layer;
            Eigen::Vector3f offset = getPerpendicularVector(direction, layer_radius, angle);
            points.push_back(center + offset);
        }
    }
    
    return points;
}