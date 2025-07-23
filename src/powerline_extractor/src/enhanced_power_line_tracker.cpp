#include "enhanced_power_line_tracker.h"
#include <algorithm>
#include <chrono>
#include <pcl/common/common.h>
#include <pcl/kdtree/kdtree_flann.h>

// ==================== LineTracker 实现 ====================

LineTracker::LineTracker(int line_id, const PowerLineProbabilityMap& prob_map) 
    : line_id_(line_id), status_(TRACK_INITIALIZING), confidence_(0.3f), 
      consecutive_updates_(0), consecutive_misses_(0), estimated_total_length_(0.0f) {
    
    creation_time_ = ros::Time::now();
    last_update_time_ = ros::Time::now();
    
    // 初始化状态向量和协方差矩阵
    state_vector_ = Eigen::VectorXf::Zero(STATE_DIM);
    covariance_matrix_ = Eigen::MatrixXf::Identity(STATE_DIM, STATE_DIM) * 1.0f;
    process_noise_ = Eigen::MatrixXf::Identity(STATE_DIM, STATE_DIM);
    process_noise_.block(0, 0, 30, 30) *= 0.01f;  // 位置噪声
    process_noise_.block(30, 30, 6, 6) *= 0.005f; // 形状噪声
    
    // 从概率地图初始化
    initializeFromProbabilityMap(prob_map);
}

void LineTracker::initializeFromProbabilityMap(const PowerLineProbabilityMap& prob_map) {
    // 获取该line_id的高概率区域
    std::vector<Eigen::Vector3f> high_prob_points = 
        prob_map.getHighProbabilityRegionsForLine(line_id_, 0.6f, 0.3f);
    
    if (high_prob_points.empty()) {
        ROS_WARN("Line %d: 无法从概率地图获取初始化点", line_id_);
        return;
    }
    
    // 对点进行排序，形成有序的控制点
    std::sort(high_prob_points.begin(), high_prob_points.end(),
              [](const Eigen::Vector3f& a, const Eigen::Vector3f& b) {
                  return a.x() < b.x(); // 简单按x坐标排序，实际可以更智能
              });
    
    // 重采样为固定数量的控制点
    std::vector<Eigen::Vector3f> control_points(NUM_CONTROL_POINTS);
    if (high_prob_points.size() >= NUM_CONTROL_POINTS) {
        for (int i = 0; i < NUM_CONTROL_POINTS; ++i) {
            float t = static_cast<float>(i) / (NUM_CONTROL_POINTS - 1);
            int idx = static_cast<int>(t * (high_prob_points.size() - 1));
            control_points[i] = high_prob_points[idx];
        }
    } else {
        // 插值生成足够的控制点
        for (int i = 0; i < NUM_CONTROL_POINTS; ++i) {
            float t = static_cast<float>(i) / (NUM_CONTROL_POINTS - 1);
            int idx = static_cast<int>(t * (high_prob_points.size() - 1));
            idx = std::min(idx, static_cast<int>(high_prob_points.size() - 1));
            control_points[i] = high_prob_points[idx];
        }
    }
    
    // 设置状态向量
    setControlPointsToState(control_points);
    
    // 估计总长度
    estimated_total_length_ = 0.0f;
    for (int i = 1; i < NUM_CONTROL_POINTS; ++i) {
        estimated_total_length_ += (control_points[i] - control_points[i-1]).norm();
    }
    
    ROS_INFO("Line %d: 从概率地图初始化完成，估计长度: %.2f", line_id_, estimated_total_length_);
}

void LineTracker::predict(float dt) {
    // 由于电力线基本静止，预测主要是增加不确定性
    // F矩阵是单位矩阵（状态不变）
    // 预测协方差：P = F*P*F^T + Q
    covariance_matrix_ += process_noise_ * dt;
    
    // 增加预测不确定性
    confidence_ = std::max(0.1f, confidence_ * 0.98f);
    
    // 状态基本保持不变（电力线静止）
    status_ = TRACK_PREDICTED;
}

void LineTracker::update(const FusedObservation& observation, const PowerLineProbabilityMap& prob_map) {
    if (observation.observed_points.empty()) {
        consecutive_misses_++;
        return;
    }
    
    // 构建观测向量（只观测到部分控制点）
    Eigen::VectorXf measurement = Eigen::VectorXf::Zero(observation.observed_points.size() * 3);
    for (size_t i = 0; i < observation.observed_points.size(); ++i) {
        measurement.segment(i * 3, 3) = observation.observed_points[i];
    }
    
    // 构建观测矩阵H（将控制点映射到观测）
    Eigen::MatrixXf H = Eigen::MatrixXf::Zero(measurement.size(), STATE_DIM);
    for (size_t i = 0; i < observation.observed_points.size(); ++i) {
        // 简化：假设观测点直接对应控制点的某个子集
        int control_idx = (i * NUM_CONTROL_POINTS) / observation.observed_points.size();
        control_idx = std::min(control_idx, NUM_CONTROL_POINTS - 1);
        H.block(i * 3, control_idx * 3, 3, 3) = Eigen::Matrix3f::Identity();
    }
    
    // 测量噪声矩阵
    Eigen::MatrixXf R = Eigen::MatrixXf::Identity(measurement.size(), measurement.size()) * 0.1f;
    
    // 卡尔曼更新
    Eigen::VectorXf predicted_measurement = H * state_vector_;
    Eigen::VectorXf innovation = measurement - predicted_measurement;
    Eigen::MatrixXf S = H * covariance_matrix_ * H.transpose() + R;
    Eigen::MatrixXf K = covariance_matrix_ * H.transpose() * S.inverse();
    
    // 状态更新
    state_vector_ += K * innovation;
    Eigen::MatrixXf I = Eigen::MatrixXf::Identity(STATE_DIM, STATE_DIM);
    covariance_matrix_ = (I - K * H) * covariance_matrix_;
    
    // 约束状态在概率地图范围内
    std::vector<Eigen::Vector3f> control_points = getControlPointsFromState();
    for (auto& point : control_points) {
        if (!isPointInProbabilityMap(point, prob_map)) {
            // 如果点超出范围，拉回到最近的有效点
            float min_dist = std::numeric_limits<float>::max();
            Eigen::Vector3f closest_point = point;
            auto valid_points = prob_map.getHighProbabilityRegionsForLine(line_id_, 0.4f, 0.2f);
            for (const auto& valid_point : valid_points) {
                float dist = (point - valid_point).norm();
                if (dist < min_dist) {
                    min_dist = dist;
                    closest_point = valid_point;
                }
            }
            point = closest_point;
        }
    }
    setControlPointsToState(control_points);
    
    // 更新统计信息
    consecutive_updates_++;
    consecutive_misses_ = 0;
    last_update_time_ = ros::Time::now();
    confidence_ = std::min(1.0f, confidence_ + 0.1f * observation.completeness_ratio);
    
    // 更新状态
    if (observation.completeness_ratio > 0.8f) {
        status_ = TRACK_STABLE;
    } else if (observation.completeness_ratio > 0.3f) {
        status_ = TRACK_PARTIAL;
    } else {
        status_ = TRACK_PREDICTED;
    }
    
    // 保存观测历史
    observation_history_.push_back(observation);
    if (observation_history_.size() > 5) {
        observation_history_.erase(observation_history_.begin());
    }
}

ReconstructedPowerLine LineTracker::generateCompleteLine(const PowerLineProbabilityMap& prob_map) const {
    ReconstructedPowerLine complete_line;
    complete_line.line_id = line_id_;
    
    // 从状态向量生成完整的样条曲线
    std::vector<Eigen::Vector3f> complete_curve = interpolateCompleteCurve(prob_map);
    complete_line.fitted_curve_points = complete_curve;

    // 添加标记向量，标识哪些点是检测的，哪些是补全的
    complete_line.point_types.resize(complete_curve.size(), 0); // 0=补全, 1=检测
    
    if (!complete_curve.empty()) {

        // 标记检测到的部分
        markDetectedParts(complete_line);
        complete_line.main_direction = (complete_curve.back() - complete_curve.front()).normalized();
        
        // 计算总长度
        complete_line.total_length = 0.0;
        for (size_t i = 1; i < complete_curve.size(); ++i) {
            complete_line.total_length += (complete_curve[i] - complete_curve[i-1]).norm();
        }
        
        // 创建点云（可选）
        complete_line.points = pcl::PointCloud<pcl::PointXYZI>::Ptr(new pcl::PointCloud<pcl::PointXYZI>);
        for (const auto& point : complete_curve) {
            pcl::PointXYZI pcl_point;
            pcl_point.x = point.x();
            pcl_point.y = point.y();
            pcl_point.z = point.z();
            pcl_point.intensity = confidence_ * 100; // 用强度表示置信度
            complete_line.points->push_back(pcl_point);
        }
    }
    
    return complete_line;
}

std::vector<Eigen::Vector3f> LineTracker::interpolateCompleteCurve(const PowerLineProbabilityMap& prob_map) const {
    // 获取控制点
    std::vector<Eigen::Vector3f> control_points = getControlPointsFromState();
    
    // 使用样条插值生成密集的曲线点
    std::vector<Eigen::Vector3f> curve_points;
    int points_per_segment = 10;
    
    for (int i = 0; i < NUM_CONTROL_POINTS - 1; ++i) {
        for (int j = 0; j < points_per_segment; ++j) {
            float t = static_cast<float>(j) / points_per_segment;
            
            // 线性插值（可以改为更高阶的样条插值）
            Eigen::Vector3f interpolated = (1.0f - t) * control_points[i] + t * control_points[i + 1];
            
            // 确保插值点在概率地图约束范围内
            if (isPointInProbabilityMap(interpolated, prob_map)) {
                curve_points.push_back(interpolated);
            }
        }
    }
    
    // 添加最后一个控制点
    if (isPointInProbabilityMap(control_points.back(), prob_map)) {
        curve_points.push_back(control_points.back());
    }
    
    return curve_points;
}

std::vector<Eigen::Vector3f> LineTracker::getControlPointsFromState() const {
    std::vector<Eigen::Vector3f> control_points(NUM_CONTROL_POINTS);
    for (int i = 0; i < NUM_CONTROL_POINTS; ++i) {
        control_points[i] = state_vector_.segment(i * 3, 3);
    }
    return control_points;
}

void LineTracker::setControlPointsToState(const std::vector<Eigen::Vector3f>& control_points) {
    for (int i = 0; i < NUM_CONTROL_POINTS && i < control_points.size(); ++i) {
        state_vector_.segment(i * 3, 3) = control_points[i];
    }
}

bool LineTracker::isPointInProbabilityMap(const Eigen::Vector3f& point, const PowerLineProbabilityMap& prob_map) const {
    float probability = prob_map.queryProbabilityAtPosition(point);
    return probability > 0.2f; // 较低的阈值，允许轻微超出
}

void LineTracker::markAsLost() {
    status_ = TRACK_LOST;
    consecutive_misses_++;
    confidence_ = std::max(0.0f, confidence_ - 0.3f);
}

bool LineTracker::shouldBeRemoved() const {
    return status_ == TRACK_LOST && (consecutive_misses_ > 10 || confidence_ < 0.1f);
}

bool LineTracker::isStable() const {
    return status_ == TRACK_STABLE && consecutive_updates_ >= 3;
}

// ==================== EnhancedPowerLineTracker 实现 ====================

EnhancedPowerLineTracker::EnhancedPowerLineTracker(ros::NodeHandle& nh) 
    : nh_(nh), is_initialized_(false) {
    
    // 读取参数
    loadParameters();
    
    // 初始化ROS发布器
    if (enable_visualization_) {
        track_visualization_pub_ = nh_.advertise<visualization_msgs::MarkerArray>(
            "enhanced_tracker/track_visualization", 1);
        detected_parts_pub_ = nh_.advertise<visualization_msgs::MarkerArray>(
            "enhanced_tracker/detected_parts", 1);
        completed_lines_pub_ = nh_.advertise<visualization_msgs::MarkerArray>(
            "enhanced_tracker/completed_lines", 1);
        track_statistics_pub_ = nh_.advertise<visualization_msgs::Marker>(
            "enhanced_tracker/statistics", 1);
    }
    
    last_update_time_ = ros::Time::now();
    
    ROS_INFO("EnhancedPowerLineTracker 初始化完成");
    ROS_INFO("参数配置:");
    ROS_INFO("  卡尔曼滤波 - 位置噪声: %.4f, 速度噪声: %.4f", process_noise_position_, process_noise_velocity_);
    ROS_INFO("  片段融合 - 合并距离: %.2f, 重叠阈值: %.2f", segment_merge_distance_, segment_overlap_threshold_);
    ROS_INFO("  补全功能: %s, 置信度阈值: %.2f", enable_completion_ ? "启用" : "禁用", completion_confidence_threshold_);
}

EnhancedPowerLineTracker::~EnhancedPowerLineTracker() {
    clearAllTracks();
    ROS_INFO("EnhancedPowerLineTracker 析构完成");
}

void EnhancedPowerLineTracker::loadParameters() {
    // 卡尔曼滤波参数
    nh_.param("enhanced_tracker/process_noise_position", process_noise_position_, 0.01f);
    nh_.param("enhanced_tracker/process_noise_velocity", process_noise_velocity_, 0.005f);
    nh_.param("enhanced_tracker/process_noise_shape", process_noise_shape_, 0.001f);
    nh_.param("enhanced_tracker/measurement_noise", measurement_noise_, 0.1f);
    
    // 片段融合参数
    nh_.param("enhanced_tracker/segment_merge_distance", segment_merge_distance_, 2.0f);
    nh_.param("enhanced_tracker/segment_overlap_threshold", segment_overlap_threshold_, 0.3f);
    nh_.param("enhanced_tracker/min_segment_length", min_segment_length_, 1.0f);
    nh_.param("enhanced_tracker/max_gap_distance", max_gap_distance_, 5.0f);
    
    // 补全参数
    nh_.param("enhanced_tracker/completion_confidence_threshold", completion_confidence_threshold_, 0.4f);
    nh_.param("enhanced_tracker/max_extrapolation_distance", max_extrapolation_distance_, 10.0f);
    nh_.param("enhanced_tracker/interpolation_resolution", interpolation_resolution_, 0.1f);
    nh_.param("enhanced_tracker/enable_completion", enable_completion_, true);
    
    // 轨迹管理参数
    nh_.param("enhanced_tracker/max_consecutive_misses", max_consecutive_misses_, 5);
    nh_.param("enhanced_tracker/min_updates_for_stable", min_updates_for_stable_, 3);
    nh_.param("enhanced_tracker/min_confidence_threshold", min_confidence_threshold_, 0.2f);
    nh_.param("enhanced_tracker/max_track_count", max_track_count_, 15);
    nh_.param("enhanced_tracker/track_timeout_duration", track_timeout_duration_, 30.0f);
    
    // 可视化参数
    nh_.param("enhanced_tracker/enable_visualization", enable_visualization_, true);
    nh_.param("enhanced_tracker/visualization_duration", visualization_duration_, 2.0f);
    nh_.param("enhanced_tracker/frame_id", frame_id_, std::string("map"));
}

bool EnhancedPowerLineTracker::initializeTracker(const PowerLineProbabilityMap& prob_map) {
    // 清空现有轨迹
    clearAllTracks();
    
    // 从概率地图获取所有分线地图
    auto all_line_maps = prob_map.getAllLineSpecificMaps();
    
    if (all_line_maps.empty()) {
        ROS_WARN("概率地图中没有分线地图，无法初始化跟踪器");
        return false;
    }
    
    // 为每个存在的line_id创建轨迹
    for (const auto& [line_id, line_map] : all_line_maps) {
        if (!line_map.empty()) {
            // 检查该line是否活跃（有足够的高概率体素）
            int high_prob_voxels = 0;
            for (const auto& [voxel_key, voxel] : line_map) {
                if (voxel.line_probability > 0.5f && voxel.confidence > 0.3f) {
                    high_prob_voxels++;
                }
            }
            
            if (high_prob_voxels >= 5) { // 至少5个高概率体素才创建轨迹
                auto tracker = std::make_unique<LineTracker>(line_id, prob_map);
                line_trackers_[line_id] = std::move(tracker);
                ROS_INFO("为line_id %d 创建轨迹（基于概率地图，%d个高概率体素）", 
                         line_id, high_prob_voxels);
            } else {
                ROS_DEBUG("line_id %d 的高概率体素太少 (%d)，跳过", line_id, high_prob_voxels);
            }
        }
    }
    
    if (line_trackers_.empty()) {
        ROS_WARN("没有足够质量的分线地图来创建轨迹");
        return false;
    }
    
    is_initialized_ = true;
    ROS_INFO("跟踪器初始化完成，从概率地图创建了 %zu 条轨迹", line_trackers_.size());
    return true;
}
bool EnhancedPowerLineTracker::initializeTracker(
    PowerLineProbabilityMap& prob_map,
    const std::vector<ReconstructedPowerLine>& initial_detections) {
    
    // 先用概率地图初始化
    if (!initializeTracker(prob_map)) {
        ROS_WARN("基于概率地图的初始化失败，尝试基于检测结果初始化");
        
        // 清空轨迹，基于检测结果重新初始化
        clearAllTracks();
        return initializeFromDetections(initial_detections, prob_map);
    }
    
    // 如果有初始检测结果，用它们来改进已创建的轨迹
    if (!initial_detections.empty()) {
        ROS_INFO("使用 %zu 个初始检测结果来改进轨迹", initial_detections.size());
        
        // 将检测结果按line_id分组
        auto grouped_detections = groupDetectionsByLineID(initial_detections, prob_map);
        
        // 融合检测结果并更新对应的轨迹
        for (const auto& [line_id, segments] : grouped_detections) {
            auto tracker_it = line_trackers_.find(line_id);
            if (tracker_it != line_trackers_.end() && !segments.empty()) {
                FusedObservation initial_obs = fuseSegments(segments);
                tracker_it->second->update(initial_obs, prob_map);
                ROS_INFO("用初始检测更新了轨迹 %d（%zu个片段）", line_id, segments.size());
            }
        }
        
        // 为没有对应轨迹的检测创建新轨迹
        for (const auto& [line_id, segments] : grouped_detections) {
            if (line_trackers_.find(line_id) == line_trackers_.end() && !segments.empty()) {
                auto new_tracker = std::make_unique<LineTracker>(line_id, prob_map);
                FusedObservation initial_obs = fuseSegments(segments);
                new_tracker->update(initial_obs, prob_map);
                line_trackers_[line_id] = std::move(new_tracker);
                ROS_INFO("基于初始检测创建新轨迹 %d", line_id);
            }
        }
    }
    
    ROS_INFO("跟踪器初始化完成，总计 %zu 条轨迹", line_trackers_.size());
    return true;
}

bool EnhancedPowerLineTracker::initializeFromDetections(
    const std::vector<ReconstructedPowerLine>& detections,
    PowerLineProbabilityMap& prob_map) {
    
    if (detections.empty()) {
        ROS_ERROR("检测结果为空，无法初始化");
        return false;
    }
    
    // 将检测结果按line_id分组
    auto grouped_detections = groupDetectionsByLineID(detections, prob_map);
    
    if (grouped_detections.empty()) {
        ROS_ERROR("检测结果分组失败，无法初始化");
        return false;
    }
    
    // 为每个line_id创建轨迹
    for (const auto& [line_id, segments] : grouped_detections) {
        if (!segments.empty()) {
            // 创建轨迹
            auto new_tracker = std::make_unique<LineTracker>(line_id, prob_map);
            
            // 用检测结果初始化轨迹
            FusedObservation initial_obs = fuseSegments(segments);
            new_tracker->update(initial_obs, prob_map);
            
            line_trackers_[line_id] = std::move(new_tracker);
            ROS_INFO("基于检测结果创建轨迹 %d（%zu个片段）", line_id, segments.size());
        }
    }
    
    if (line_trackers_.empty()) {
        ROS_ERROR("没有成功创建任何轨迹");
        return false;
    }
    
    is_initialized_ = true;
    ROS_INFO("基于检测结果初始化完成，创建了 %zu 条轨迹", line_trackers_.size());
    return true;
}

std::vector<ReconstructedPowerLine> EnhancedPowerLineTracker::updateTracker(
    const std::vector<ReconstructedPowerLine>& current_detections,
    PowerLineProbabilityMap& prob_map) {
    
    if (!is_initialized_) {
        ROS_WARN("跟踪器未初始化，请先调用initializeTracker");
        return current_detections;
    }
    
    auto start_time = std::chrono::high_resolution_clock::now();
    ros::Time current_time = ros::Time::now();
    float dt = (current_time - last_update_time_).toSec();
    
    // 1. 按line_id分组检测结果
    auto grouped_detections = groupDetectionsByLineID(current_detections, prob_map);
    
    // 2. 预测所有轨迹
    predictAllTracks(dt);
    
    // 3. 融合每个line_id的多个片段
    std::map<int, FusedObservation> fused_observations;
    for (const auto& [line_id, segments] : grouped_detections) {
        if (!segments.empty()) {
            fused_observations[line_id] = fuseSegments(segments);
        }
    }
    
    // 4. 更新现有轨迹
    updateTracksWithObservations(fused_observations, prob_map);
    
    // 5. 创建新轨迹（对于新出现的line_id）
    createNewTracks(fused_observations, prob_map);
    
    // 6. 清理无效轨迹
    removeInactiveTracks();
    
    // 7. 生成输出结果
    std::vector<ReconstructedPowerLine> result = generateOutputLines(prob_map);
    
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end_time - start_time;
    
    ROS_INFO("跟踪器更新完成，耗时: %.3f 秒", duration.count());
    ROS_INFO("活跃轨迹: %zu，输入检测: %zu，输出结果: %zu", 
             line_trackers_.size(), current_detections.size(), result.size());
    
    // 发布可视化
    if (enable_visualization_) {
        publishTrackVisualization();
        publishCompletedLinesVisualization(result);
        publishTrackStatistics();
    }
    
    last_update_time_ = current_time;
    return result;
}

std::map<int, std::vector<DetectionSegment>> EnhancedPowerLineTracker::groupDetectionsByLineID(
    const std::vector<ReconstructedPowerLine>& detections,
    PowerLineProbabilityMap& prob_map) {
    
    std::map<int, std::vector<DetectionSegment>> grouped_segments;
    
    // 获取检测结果的line_id分配
    std::vector<int> line_ids = prob_map.assignLineIDsForDetections(detections);
    
    for (size_t i = 0; i < detections.size() && i < line_ids.size(); ++i) {
        int line_id = line_ids[i];
        DetectionSegment segment = createSegmentFromDetection(detections[i], i);
        segment.confidence = calculateSegmentConfidence(segment, prob_map);
        
        if (segment.segment_length >= min_segment_length_ && segment.confidence > 0.1f) {
            grouped_segments[line_id].push_back(segment);
        }
    }
    
    ROS_DEBUG("检测分组结果: %zu 个line_id", grouped_segments.size());
    return grouped_segments;
}

FusedObservation EnhancedPowerLineTracker::fuseSegments(const std::vector<DetectionSegment>& segments) {
    FusedObservation fused;
    fused.segments = segments;
    
    if (segments.empty()) return fused;
    
    // 收集所有观测点
    std::vector<Eigen::Vector3f> all_points;
    for (const auto& segment : segments) {
        all_points.insert(all_points.end(), segment.segment_points.begin(), segment.segment_points.end());
        fused.total_observed_length += segment.segment_length;
    }
    
    // 对点进行空间排序
    fused.observed_points = sortPointsSpatially(all_points);
    
    // 标记每个点的有效性（简化：所有观测点都有效）
    fused.point_validity.resize(fused.observed_points.size(), true);
    
    // 计算完整度（基于观测长度与期望长度的比例）
    float estimated_total_length = 0.0f;
    if (!fused.observed_points.empty()) {
        estimated_total_length = (fused.observed_points.back() - fused.observed_points.front()).norm();
    }
    
    fused.completeness_ratio = estimated_total_length > 0 ? 
        std::min(1.0f, fused.total_observed_length / estimated_total_length) : 0.0f;
    
    ROS_DEBUG("融合了 %zu 个片段，观测点: %zu，完整度: %.2f", 
              segments.size(), fused.observed_points.size(), fused.completeness_ratio);
    
    return fused;
}

void EnhancedPowerLineTracker::predictAllTracks(float dt) {
    for (auto& [line_id, tracker] : line_trackers_) {
        tracker->predict(dt);
    }
}

void EnhancedPowerLineTracker::updateTracksWithObservations(
    const std::map<int, FusedObservation>& observations,
    const PowerLineProbabilityMap& prob_map) {
    
    for (auto& [line_id, tracker] : line_trackers_) {
        auto obs_it = observations.find(line_id);
        if (obs_it != observations.end()) {
            tracker->update(obs_it->second, prob_map);
        } else {
            tracker->markAsLost();
        }
    }
}

void EnhancedPowerLineTracker::createNewTracks(
    const std::map<int, FusedObservation>& observations,
    const PowerLineProbabilityMap& prob_map) {
    
    for (const auto& [line_id, observation] : observations) {
        // 如果这个line_id还没有对应的轨迹
        if (line_trackers_.find(line_id) == line_trackers_.end()) {
            // 检查是否为有效的新轨迹
            if (observation.completeness_ratio > 0.2f && observation.total_observed_length > min_segment_length_) {
                auto new_tracker = std::make_unique<LineTracker>(line_id, prob_map);
                new_tracker->update(observation, prob_map);
                line_trackers_[line_id] = std::move(new_tracker);
                
                ROS_INFO("创建新轨迹 line_id: %d", line_id);
            }
        }
    }
}

void EnhancedPowerLineTracker::removeInactiveTracks() {
    auto it = line_trackers_.begin();
    while (it != line_trackers_.end()) {
        if (it->second->shouldBeRemoved()) {
            ROS_INFO("移除轨迹 line_id: %d", it->first);
            it = line_trackers_.erase(it);
        } else {
            ++it;
        }
    }
    
    // 限制轨迹数量
    if (line_trackers_.size() > static_cast<size_t>(max_track_count_)) {
        // 按置信度排序，保留置信度最高的轨迹
        std::vector<std::pair<float, int>> confidence_line_pairs;
        for (const auto& [line_id, tracker] : line_trackers_) {
            confidence_line_pairs.emplace_back(tracker->confidence_, line_id);
        }
        
        std::sort(confidence_line_pairs.begin(), confidence_line_pairs.end(), std::greater<>());
        
        // 移除置信度最低的轨迹
        for (size_t i = max_track_count_; i < confidence_line_pairs.size(); ++i) {
            int line_id = confidence_line_pairs[i].second;
            line_trackers_.erase(line_id);
            ROS_WARN("由于数量限制，移除轨迹 line_id: %d", line_id);
        }
    }
}

std::vector<ReconstructedPowerLine> EnhancedPowerLineTracker::generateOutputLines(
    const PowerLineProbabilityMap& prob_map) {
    
    std::vector<ReconstructedPowerLine> output_lines;
    
    for (const auto& [line_id, tracker] : line_trackers_) {
        // 只输出稳定或部分观测的轨迹
        if (tracker->status_ == TRACK_STABLE || 
            tracker->status_ == TRACK_PARTIAL ||
            (tracker->status_ == TRACK_PREDICTED && enable_completion_ && 
             tracker->confidence_ >= completion_confidence_threshold_)) {
            
            ReconstructedPowerLine complete_line = tracker->generateCompleteLine(prob_map);
            if (!complete_line.fitted_curve_points.empty()) {
                output_lines.push_back(complete_line);
            }
        }
    }
    
    return output_lines;
}

// ==================== 辅助函数实现 ====================

DetectionSegment EnhancedPowerLineTracker::createSegmentFromDetection(
    const ReconstructedPowerLine& detection, int index) {
    
    DetectionSegment segment;
    segment.detection_index = index;
    segment.segment_points = detection.fitted_curve_points;
    
    if (!detection.fitted_curve_points.empty()) {
        segment.start_point = detection.fitted_curve_points.front();
        segment.end_point = detection.fitted_curve_points.back();
        segment.segment_length = detection.total_length;
    }
    
    return segment;
}

std::vector<Eigen::Vector3f> EnhancedPowerLineTracker::sortPointsSpatially(
    const std::vector<Eigen::Vector3f>& points) {
    
    if (points.empty()) return points;
    
    std::vector<Eigen::Vector3f> sorted_points = points;
    
    // 简单的空间排序：从一端开始，每次选择距离最近的下一个点
    std::vector<bool> used(points.size(), false);
    std::vector<Eigen::Vector3f> result;
    
    // 选择起始点（可以选择x坐标最小的点）
    int current_idx = 0;
    for (size_t i = 1; i < points.size(); ++i) {
        if (points[i].x() < points[current_idx].x()) {
            current_idx = i;
        }
    }
    
    result.push_back(points[current_idx]);
    used[current_idx] = true;
    
    // 贪心选择最近的下一个点
    for (size_t count = 1; count < points.size(); ++count) {
        float min_dist = std::numeric_limits<float>::max();
        int next_idx = -1;
        
        for (size_t i = 0; i < points.size(); ++i) {
            if (!used[i]) {
                float dist = (points[i] - result.back()).norm();
                if (dist < min_dist) {
                    min_dist = dist;
                    next_idx = i;
                }
            }
        }
        
        if (next_idx >= 0) {
            result.push_back(points[next_idx]);
            used[next_idx] = true;
        }
    }
    
    return result;
}

float EnhancedPowerLineTracker::calculateSegmentConfidence(
    const DetectionSegment& segment, const PowerLineProbabilityMap& prob_map) {
    
    if (segment.segment_points.empty()) return 0.0f;
    
    float total_probability = 0.0f;
    int valid_points = 0;
    
    for (const auto& point : segment.segment_points) {
        float prob = prob_map.queryProbabilityAtPosition(point);
        if (prob > 0.1f) {
            total_probability += prob;
            valid_points++;
        }
    }
    
    return valid_points > 0 ? total_probability / valid_points : 0.0f;
}

// ==================== 可视化函数实现 ====================

void EnhancedPowerLineTracker::publishTrackVisualization() {
    if (!enable_visualization_) return;
    
    visualization_msgs::MarkerArray marker_array;
    int marker_id = 0;
    
    for (const auto& [line_id, tracker] : line_trackers_) {
        visualization_msgs::Marker marker = createTrackMarker(*tracker, marker_id++);
        marker_array.markers.push_back(marker);
    }
    
    track_visualization_pub_.publish(marker_array);
}

void EnhancedPowerLineTracker::publishCompletedLinesVisualization(
    const std::vector<ReconstructedPowerLine>& completed_lines) {
    
    if (!enable_visualization_) return;
    
    visualization_msgs::MarkerArray complete_tracks;
    visualization_msgs::MarkerArray detected_parts;
    visualization_msgs::MarkerArray completed_parts;
    
    int marker_id = 0;
    
    for (const auto& line : completed_lines) {
        if (line.fitted_curve_points.empty()) continue;
        
        // 1. 完整轨迹（所有点，统一颜色）
        visualization_msgs::Marker complete_marker;
        complete_marker.header.frame_id = frame_id_;
        complete_marker.header.stamp = ros::Time::now();
        complete_marker.ns = "complete_tracks";
        complete_marker.id = marker_id;
        complete_marker.type = visualization_msgs::Marker::LINE_STRIP;
        complete_marker.action = visualization_msgs::Marker::ADD;
        
        for (const auto& point : line.fitted_curve_points) {
            geometry_msgs::Point p;
            p.x = point.x(); p.y = point.y(); p.z = point.z();
            complete_marker.points.push_back(p);
        }
        
        complete_marker.scale.x = 0.05; // 细线显示完整轨迹
        complete_marker.color.r = 0.5f; complete_marker.color.g = 0.5f; 
        complete_marker.color.b = 0.5f; complete_marker.color.a = 0.6f; // 灰色
        complete_marker.lifetime = ros::Duration(visualization_duration_);
        complete_tracks.markers.push_back(complete_marker);
        
        // 2. 分别处理检测部分和补全部分
        if (!line.point_types.empty()) {
            // 检测部分（绿色，粗线）
            visualization_msgs::Marker detected_marker;
            detected_marker.header.frame_id = frame_id_;
            detected_marker.header.stamp = ros::Time::now();
            detected_marker.ns = "detected_parts";
            detected_marker.id = marker_id;
            detected_marker.type = visualization_msgs::Marker::LINE_STRIP;
            detected_marker.action = visualization_msgs::Marker::ADD;
            
            // 补全部分（红色，虚线效果）
            visualization_msgs::Marker completed_marker;
            completed_marker.header.frame_id = frame_id_;
            completed_marker.header.stamp = ros::Time::now();
            completed_marker.ns = "completed_parts";
            completed_marker.id = marker_id;
            completed_marker.type = visualization_msgs::Marker::LINE_STRIP;
            completed_marker.action = visualization_msgs::Marker::ADD;
            
            // 分类添加点
            std::vector<geometry_msgs::Point> detected_points;
            std::vector<geometry_msgs::Point> completed_points;
            
            for (size_t i = 0; i < line.fitted_curve_points.size() && i < line.point_types.size(); ++i) {
                geometry_msgs::Point p;
                p.x = line.fitted_curve_points[i].x();
                p.y = line.fitted_curve_points[i].y(); 
                p.z = line.fitted_curve_points[i].z();
                
                if (line.point_types[i] == 1) { // 检测到的
                    detected_points.push_back(p);
                } else { // 补全的
                    completed_points.push_back(p);
                }
            }
            
            // 处理检测部分的连续段
            if (!detected_points.empty()) {
                detected_marker.points = detected_points;
                detected_marker.scale.x = 0.15; // 粗线
                detected_marker.color.r = 0.0f; detected_marker.color.g = 1.0f; 
                detected_marker.color.b = 0.0f; detected_marker.color.a = 0.9f; // 绿色
                detected_marker.lifetime = ros::Duration(visualization_duration_);
                detected_parts.markers.push_back(detected_marker);
            }
            
            // 处理补全部分的连续段  
            if (!completed_points.empty()) {
                completed_marker.points = completed_points;
                completed_marker.scale.x = 0.12; // 中等粗细
                completed_marker.color.r = 1.0f; completed_marker.color.g = 0.0f; 
                completed_marker.color.b = 0.0f; completed_marker.color.a = 0.7f; // 红色
                completed_marker.lifetime = ros::Duration(visualization_duration_);
                completed_parts.markers.push_back(completed_marker);
            }
        }
        
        // 添加线段ID文本
        visualization_msgs::Marker text_marker;
        text_marker.header.frame_id = frame_id_;
        text_marker.header.stamp = ros::Time::now();
        text_marker.ns = "line_ids";
        text_marker.id = marker_id;
        text_marker.type = visualization_msgs::Marker::TEXT_VIEW_FACING;
        text_marker.action = visualization_msgs::Marker::ADD;
        
        // 文本位置在线段中点上方
        Eigen::Vector3f mid_point = (line.fitted_curve_points.front() + 
                                    line.fitted_curve_points.back()) * 0.5f;
        text_marker.pose.position.x = mid_point.x();
        text_marker.pose.position.y = mid_point.y();
        text_marker.pose.position.z = mid_point.z() + 1.5f;
        text_marker.pose.orientation.w = 1.0;
        
        text_marker.scale.z = 0.6f;
        text_marker.color.r = 1.0f; text_marker.color.g = 1.0f; 
        text_marker.color.b = 0.0f; text_marker.color.a = 1.0f; // 黄色
        text_marker.text = "Line_" + std::to_string(line.line_id);
        text_marker.lifetime = ros::Duration(visualization_duration_);
        complete_tracks.markers.push_back(text_marker);
        
        marker_id++;
    }
    
    // 发布三个不同的话题
    track_visualization_pub_.publish(complete_tracks);
    detected_parts_pub_.publish(detected_parts);
    completed_lines_pub_.publish(completed_parts);
    
    ROS_DEBUG("发布可视化: 完整轨迹 %zu, 检测部分 %zu, 补全部分 %zu", 
              complete_tracks.markers.size(), detected_parts.markers.size(), 
              completed_parts.markers.size());
}

void EnhancedPowerLineTracker::publishTrackStatistics() {
    if (!enable_visualization_) return;
    
    int total_tracks, stable_tracks, partial_tracks;
    float avg_confidence;
    getTrackerStatistics(total_tracks, stable_tracks, partial_tracks, avg_confidence);
    
    visualization_msgs::Marker text_marker;
    text_marker.header.frame_id = frame_id_;
    text_marker.header.stamp = ros::Time::now();
    text_marker.ns = "tracker_statistics";
    text_marker.id = 0;
    text_marker.type = visualization_msgs::Marker::TEXT_VIEW_FACING;
    text_marker.action = visualization_msgs::Marker::ADD;
    
    text_marker.pose.position.x = 0.0;
    text_marker.pose.position.y = 0.0;
    text_marker.pose.position.z = 15.0f;
    text_marker.pose.orientation.w = 1.0;
    
    std::ostringstream ss;
    ss << "增强型电力线跟踪器\n";
    ss << "总轨迹: " << total_tracks << "\n";
    ss << "稳定: " << stable_tracks << " | 部分: " << partial_tracks << "\n";
    ss << "平均置信度: " << std::fixed << std::setprecision(2) << avg_confidence;
    
    text_marker.text = ss.str();
    text_marker.scale.z = 0.8f;
    text_marker.color.r = 1.0f; text_marker.color.g = 1.0f; 
    text_marker.color.b = 1.0f; text_marker.color.a = 1.0f;
    text_marker.lifetime = ros::Duration(1.0);
    
    track_statistics_pub_.publish(text_marker);
}

visualization_msgs::Marker EnhancedPowerLineTracker::createTrackMarker(
    const LineTracker& tracker, int marker_id) const {
    
    visualization_msgs::Marker marker;
    marker.header.frame_id = frame_id_;
    marker.header.stamp = ros::Time::now();
    marker.ns = "line_tracks";
    marker.id = marker_id;
    marker.type = visualization_msgs::Marker::LINE_STRIP;
    marker.action = visualization_msgs::Marker::ADD;
    
    // 获取控制点作为轨迹显示
    std::vector<Eigen::Vector3f> control_points = tracker.getControlPointsFromState();
    for (const auto& point : control_points) {
        geometry_msgs::Point p;
        p.x = point.x(); p.y = point.y(); p.z = point.z();
        marker.points.push_back(p);
    }
    
    marker.scale.x = 0.15;
    marker.color = getTrackStatusColor(tracker.status_);
    marker.lifetime = ros::Duration(visualization_duration_);
    
    return marker;
}

std_msgs::ColorRGBA EnhancedPowerLineTracker::getTrackStatusColor(LineTrackStatus status) const {
    std_msgs::ColorRGBA color;
    color.a = 0.8f;
    
    switch (status) {
        case TRACK_INITIALIZING:
            color.r = 1.0f; color.g = 1.0f; color.b = 0.0f;  // 黄色
            break;
        case TRACK_STABLE:
            color.r = 0.0f; color.g = 1.0f; color.b = 0.0f;  // 绿色
            break;
        case TRACK_PARTIAL:
            color.r = 0.0f; color.g = 0.5f; color.b = 1.0f;  // 蓝色
            break;
        case TRACK_PREDICTED:
            color.r = 1.0f; color.g = 0.5f; color.b = 0.0f;  // 橙色
            break;
        case TRACK_LOST:
            color.r = 1.0f; color.g = 0.0f; color.b = 0.0f;  // 红色
            break;
        default:
            color.r = 0.5f; color.g = 0.5f; color.b = 0.5f;  // 灰色
            break;
    }
    
    return color;
}

void EnhancedPowerLineTracker::getTrackerStatistics(int& total_tracks, int& stable_tracks, 
                                                   int& partial_tracks, float& avg_confidence) const {
    total_tracks = line_trackers_.size();
    stable_tracks = 0;
    partial_tracks = 0;
    float total_confidence = 0.0f;
    
    for (const auto& [line_id, tracker] : line_trackers_) {
        if (tracker->status_ == TRACK_STABLE) stable_tracks++;
        else if (tracker->status_ == TRACK_PARTIAL) partial_tracks++;
        total_confidence += tracker->confidence_;
    }
    
    avg_confidence = total_tracks > 0 ? total_confidence / total_tracks : 0.0f;
}

void EnhancedPowerLineTracker::clearAllTracks() {
    line_trackers_.clear();
    is_initialized_ = false;
}