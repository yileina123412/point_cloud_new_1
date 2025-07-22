#include "power_line_tracker.h"
#include <algorithm>
#include <chrono>
#include <queue>

// ==================== TrackedPowerLine 实现 ====================

TrackedPowerLine::TrackedPowerLine(int id, const ReconstructedPowerLine& initial_detection) 
    : track_id_(id), status_(TRACK_NEW), confidence_(0.8f), consecutive_misses_(0), 
      consecutive_hits_(1), total_updates_(1) {
    
    creation_time_ = ros::Time::now();
    last_update_time_ = ros::Time::now();
    
    // 初始化状态
    state_.start_point = initial_detection.fitted_curve_points.front();
    state_.end_point = initial_detection.fitted_curve_points.back();
    state_.main_direction = initial_detection.main_direction;
    state_.total_length = initial_detection.total_length;
    
    // 计算平均曲率
    state_.average_curvature = 0.0f;
    if (initial_detection.fitted_curve_points.size() > 2) {
        for (size_t i = 1; i < initial_detection.fitted_curve_points.size() - 1; ++i) {
            Eigen::Vector3f v1 = initial_detection.fitted_curve_points[i] - initial_detection.fitted_curve_points[i-1];
            Eigen::Vector3f v2 = initial_detection.fitted_curve_points[i+1] - initial_detection.fitted_curve_points[i];
            float angle = std::acos(v1.normalized().dot(v2.normalized()));
            state_.average_curvature += angle;
        }
        state_.average_curvature /= (initial_detection.fitted_curve_points.size() - 2);
    }
    
    // 提取控制点
    state_.control_points = extractControlPoints(initial_detection);
    
    // 初始化卡尔曼滤波器
    initializeKalmanFilter(initial_detection);
    
    // 保存历史
    history_.push_back(initial_detection);
}
// 初始化状态向量，协方差矩阵和过程噪声
void TrackedPowerLine::initializeKalmanFilter(const ReconstructedPowerLine& initial_detection) {
    // 11维状态向量: [start(3) + end(3) + direction(3) + length(1) + curvature(1)]
    state_vector_ = Eigen::VectorXf::Zero(11);
    
    // 初始状态
    state_vector_.segment(0, 3) = state_.start_point;
    state_vector_.segment(3, 3) = state_.end_point;
    state_vector_.segment(6, 3) = state_.main_direction;
    state_vector_(9) = state_.total_length;
    state_vector_(10) = state_.average_curvature;
    
    // 初始协方差矩阵
    covariance_ = Eigen::MatrixXf::Identity(11, 11) * 0.1f;
    
    // 过程噪声矩阵（电力线基本静止，噪声很小）
    process_noise_ = Eigen::MatrixXf::Identity(11, 11);
    process_noise_.block(0, 0, 6, 6) *= 0.01f;  // 位置噪声
    process_noise_.block(6, 6, 3, 3) *= 0.005f; // 方向噪声
    process_noise_(9, 9) = 0.01f;               // 长度噪声
    process_noise_(10, 10) = 0.001f;            // 曲率噪声
}

void TrackedPowerLine::predict() {
    // 由于电力线基本静止，预测主要是保持当前状态
    predicted_state_ = state_;
    
    // 卡尔曼预测步骤
    // 状态预测：x_k|k-1 = F * x_k-1|k-1 (这里F是单位矩阵，因为静止)
    // 协方差预测：P_k|k-1 = F * P_k-1|k-1 * F^T + Q
    covariance_ += process_noise_;
    
    // 更新预测状态的不确定性（随时间增加）
    predicted_state_.position_drift *= 0.98f;  // 衰减漂移
}

void TrackedPowerLine::update(const ReconstructedPowerLine& detection) {
    // 构建测量向量
    Eigen::VectorXf measurement = Eigen::VectorXf::Zero(11);
    measurement.segment(0, 3) = detection.fitted_curve_points.front();
    measurement.segment(3, 3) = detection.fitted_curve_points.back();
    measurement.segment(6, 3) = detection.main_direction;
    measurement(9) = detection.total_length;
    
    // 计算新的平均曲率
    float new_curvature = 0.0f;
    if (detection.fitted_curve_points.size() > 2) {
        for (size_t i = 1; i < detection.fitted_curve_points.size() - 1; ++i) {
            Eigen::Vector3f v1 = detection.fitted_curve_points[i] - detection.fitted_curve_points[i-1];
            Eigen::Vector3f v2 = detection.fitted_curve_points[i+1] - detection.fitted_curve_points[i];
            float angle = std::acos(std::max(-1.0f, std::min(1.0f, v1.normalized().dot(v2.normalized()))));
            new_curvature += angle;
        }
        new_curvature /= (detection.fitted_curve_points.size() - 2);
    }
    measurement(10) = new_curvature;
    
    // 测量噪声矩阵
    Eigen::MatrixXf R = Eigen::MatrixXf::Identity(11, 11) * 0.05f;
    
    // 卡尔曼更新步骤
    Eigen::MatrixXf H = Eigen::MatrixXf::Identity(11, 11);  // 测量矩阵（直接观测）
    
    // 创新（残差）
    Eigen::VectorXf innovation = measurement - state_vector_;
    
    // 创新协方差
    Eigen::MatrixXf S = H * covariance_ * H.transpose() + R;
    
    // 卡尔曼增益
    Eigen::MatrixXf K = covariance_ * H.transpose() * S.inverse();
    
    // 状态更新
    state_vector_ += K * innovation;
    
    // 协方差更新
    Eigen::MatrixXf I = Eigen::MatrixXf::Identity(11, 11);
    covariance_ = (I - K * H) * covariance_;
    
    // 从状态向量更新状态结构
    vectorToState();
    
    // 更新统计信息
    consecutive_hits_++;
    consecutive_misses_ = 0;
    total_updates_++;
    last_update_time_ = ros::Time::now();
    
    // 更新置信度
    confidence_ = std::min(1.0f, confidence_ + 0.1f);
    
    // 更新历史
    updateHistory(detection);
    
    // 更新状态
    if (status_ == TRACK_NEW && consecutive_hits_ >= 3) {
        status_ = TRACK_CONFIRMED;
    } else if (status_ == TRACK_PREDICTED) {
        status_ = TRACK_ACTIVE;
    }
}
// 从状态向量更新状态结构
void TrackedPowerLine::vectorToState() {
    state_.start_point = state_vector_.segment(0, 3);
    state_.end_point = state_vector_.segment(3, 3);
    state_.main_direction = state_vector_.segment(6, 3).normalized();
    state_.total_length = state_vector_(9);
    state_.average_curvature = state_vector_(10);
}
// 从样条点提取控制点
std::vector<Eigen::Vector3f> TrackedPowerLine::extractControlPoints(const ReconstructedPowerLine& line) const {
    std::vector<Eigen::Vector3f> control_points(5);
    
    if (line.fitted_curve_points.empty()) {
        return control_points;
    }
    
    size_t n = line.fitted_curve_points.size();
    control_points[0] = line.fitted_curve_points[0];                    // 起点
    control_points[1] = line.fitted_curve_points[n * 1 / 4];           // 1/4点
    control_points[2] = line.fitted_curve_points[n / 2];               // 中点
    control_points[3] = line.fitted_curve_points[n * 3 / 4];           // 3/4点
    control_points[4] = line.fitted_curve_points[n - 1];               // 终点
    
    return control_points;
}
// 计算与检测结果的相似度
float TrackedPowerLine::calculateSimilarity(const ReconstructedPowerLine& detection) const {
    float spatial_overlap = 0.0f;
    float direction_similarity = state_.main_direction.dot(detection.main_direction);
    float length_ratio = std::min(state_.total_length, (float)detection.total_length) /
                        std::max(state_.total_length, (float)detection.total_length);
    
    // 计算空间重叠（基于控制点）
    std::vector<Eigen::Vector3f> det_control_points = extractControlPoints(detection);
    float total_distance = 0.0f;
    for (size_t i = 0; i < 5; ++i) {
        total_distance += (state_.control_points[i] - det_control_points[i]).norm();
    }
    spatial_overlap = std::exp(-total_distance / 5.0f);  // 平均距离转相似度
    
    return 0.4f * spatial_overlap + 0.3f * direction_similarity + 0.3f * length_ratio;
}


// 生成补全的电力线
ReconstructedPowerLine TrackedPowerLine::generateCompletedLine() const {
    ReconstructedPowerLine completed_line;
    
    if (history_.empty()) {
        return completed_line;
    }
    
    // 基于最近历史和当前预测状态生成补全线
    const auto& last_detection = history_.back();
    
    completed_line.line_id = track_id_;
    completed_line.main_direction = predicted_state_.main_direction;
    completed_line.total_length = predicted_state_.total_length;
    
    // 使用预测状态的控制点插值生成拟合曲线
    int num_points = std::max(10, (int)(predicted_state_.total_length * 20));  // 每米20个点
    completed_line.fitted_curve_points.reserve(num_points);
    
    for (int i = 0; i < num_points; ++i) {
        float t = static_cast<float>(i) / (num_points - 1);
        Eigen::Vector3f point = interpolateControlPoint(t);
        completed_line.fitted_curve_points.push_back(point);
    }
    
    // 复制点云（如果需要）
    if (last_detection.points && !last_detection.points->empty()) {
        completed_line.points = pcl::PointCloud<pcl::PointXYZI>::Ptr(
            new pcl::PointCloud<pcl::PointXYZI>(*last_detection.points));
    }
    
    return completed_line;
}

Eigen::Vector3f TrackedPowerLine::interpolateControlPoint(float t) const {
    // 使用三次样条插值在控制点之间插值
    if (predicted_state_.control_points.size() < 5) {
        return Eigen::Vector3f::Zero();
    }
    
    // 将t映射到控制点区间
    float scaled_t = t * 4.0f;  // [0,1] -> [0,4]
    int segment = std::min(3, (int)std::floor(scaled_t));
    float local_t = scaled_t - segment;
    
    // 使用Catmull-Rom样条插值
    Eigen::Vector3f p0 = predicted_state_.control_points[std::max(0, segment - 1)];
    Eigen::Vector3f p1 = predicted_state_.control_points[segment];
    Eigen::Vector3f p2 = predicted_state_.control_points[segment + 1];
    Eigen::Vector3f p3 = predicted_state_.control_points[std::min(4, segment + 2)];
    
    float t2 = local_t * local_t;
    float t3 = t2 * local_t;
    
    return 0.5f * ((2.0f * p1) + 
                   (-p0 + p2) * local_t +
                   (2.0f * p0 - 5.0f * p1 + 4.0f * p2 - p3) * t2 +
                   (-p0 + 3.0f * p1 - 3.0f * p2 + p3) * t3);
}

void TrackedPowerLine::updateHistory(const ReconstructedPowerLine& detection) {
    history_.push_back(detection);
    
    // 限制历史缓存大小
    if (history_.size() > 10) {  // 保持最近10帧
        history_.erase(history_.begin());
    }
}

void TrackedPowerLine::markAsLost() {
    status_ = TRACK_LOST;
    consecutive_misses_++;
    confidence_ = std::max(0.0f, confidence_ - 0.2f);
}

void TrackedPowerLine::markAsConfirmed() {
    status_ = TRACK_CONFIRMED;
}

bool TrackedPowerLine::isStable() const {
    return status_ == TRACK_CONFIRMED && consecutive_hits_ >= 5;
}

bool TrackedPowerLine::shouldBeRemoved() const {
    return status_ == TRACK_LOST && (consecutive_misses_ > 10 || confidence_ < 0.1f);
}





// ==================== PowerLineTracker 实现 ====================

PowerLineTracker::PowerLineTracker(ros::NodeHandle& nh) 
    : nh_(nh), next_track_id_(0), is_initialized_(false) {
    
    // 读取参数
    loadParameters();

    if (enable_track_visualization_) {
        track_markers_pub_ = nh_.advertise<visualization_msgs::MarkerArray>(
            "power_line_tracker/track_markers", 1);
        prediction_markers_pub_ = nh_.advertise<visualization_msgs::MarkerArray>(
            "power_line_tracker/prediction_markers", 1);
        completed_lines_pub_ = nh_.advertise<visualization_msgs::MarkerArray>(
            "power_line_tracker/completed_lines", 1);
        track_info_pub_ = nh_.advertise<visualization_msgs::Marker>(
            "power_line_tracker/track_info", 1);
    }
    
    
    ROS_INFO("PowerLineTracker 初始化完成");
    ROS_INFO("参数配置:");
    ROS_INFO("  关联阈值: %.2f", association_threshold_);
    ROS_INFO("  最大连续未匹配: %d", max_consecutive_misses_);
    ROS_INFO("  最大轨迹数量: %d", max_track_count_);
    ROS_INFO("  启用补全: %s", enable_completion_ ? "是" : "否");
}

PowerLineTracker::~PowerLineTracker() {
    clearAllTracks();
    ROS_INFO("PowerLineTracker 析构完成");
}

void PowerLineTracker::loadParameters() {
    // 相似度计算参数
    nh_.param("tracker/spatial_overlap_weight", spatial_overlap_weight_, 0.4f);
    nh_.param("tracker/direction_similarity_weight", direction_similarity_weight_, 0.25f);
    nh_.param("tracker/length_similarity_weight", length_similarity_weight_, 0.2f);
    nh_.param("tracker/shape_similarity_weight", shape_similarity_weight_, 0.15f);
    
    // 关联阈值
    nh_.param("tracker/association_threshold", association_threshold_, 0.6f);
    nh_.param("tracker/max_association_distance", max_association_distance_, 5.0f);
    
    // 轨迹管理参数
    nh_.param("tracker/max_consecutive_misses", max_consecutive_misses_, 5);
    nh_.param("tracker/min_hits_for_confirmation", min_hits_for_confirmation_, 3);
    nh_.param("tracker/min_confidence_threshold", min_confidence_threshold_, 0.3f);
    nh_.param("tracker/max_track_count", max_track_count_, 20);
    nh_.param("tracker/history_buffer_size", history_buffer_size_, 10);
    
    // 卡尔曼滤波参数
    nh_.param("tracker/process_noise_position", process_noise_position_, 0.01f);
    nh_.param("tracker/process_noise_direction", process_noise_direction_, 0.005f);
    nh_.param("tracker/process_noise_length", process_noise_length_, 0.01f);
    nh_.param("tracker/measurement_noise", measurement_noise_, 0.05f);
    
    // 预测和补全参数
    nh_.param("tracker/prediction_uncertainty", prediction_uncertainty_, 0.1f);
    nh_.param("tracker/enable_completion", enable_completion_, true);
    nh_.param("tracker/completion_confidence_threshold", completion_confidence_threshold_, 0.5f);
    // 可视化参数
    nh_.param("tracker/enable_track_visualization", enable_track_visualization_, true);
    nh_.param("tracker/track_visualization_duration", track_visualization_duration_, 2.0f);
    nh_.param("tracker/frame_id", frame_id_, std::string("map"));

    // 片段处理参数
    nh_.param("tracker/fragment_continuity_threshold", fragment_continuity_threshold_, 0.6f);
    nh_.param("tracker/max_fragment_gap_distance", max_fragment_gap_distance_, 3.0f);
    nh_.param("tracker/direction_consistency_threshold", direction_consistency_threshold_, 0.8f);
    nh_.param("tracker/fragment_merge_confidence", fragment_merge_confidence_, 0.7f);
}

bool PowerLineTracker::initializeTracker(const std::vector<ReconstructedPowerLine>& initial_detections) {
    if (initial_detections.empty()) {
        ROS_WARN("初始检测列表为空，无法初始化跟踪器");
        return false;
    }
    
    // 清空现有轨迹
    clearAllTracks();
    
    // 为每个初始检测创建轨迹
    for (const auto& detection : initial_detections) {
        TrackedPowerLine new_track(next_track_id_++, detection);
        new_track.status_ = TRACK_ACTIVE;
        active_tracks_.push_back(new_track);
    }
    
    is_initialized_ = true;
    
    ROS_INFO("跟踪器初始化完成，创建了 %zu 条轨迹", active_tracks_.size());
    return true;
}

std::vector<ReconstructedPowerLine> PowerLineTracker::updateTracker(
    const std::vector<ReconstructedPowerLine>& current_detections,
    const PowerLineProbabilityMap& prob_map) {
    
    if (!is_initialized_) {
        ROS_WARN("跟踪器未初始化，请先调用initializeTracker");
        return current_detections;
    }
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // 1. 预测步骤
    predictAllTracks();
    
    // 2. 数据关联
    AssociationResult association = associateDetections(current_detections, &prob_map);

  
    
    // 3. 处理关联结果
    processAssociationResults(association, current_detections);





    
    // 4. 清理无效轨迹
    removeInactiveTracks();
    
    // 5. 限制轨迹数量
    limitTrackCount();
    
    // 6. 生成输出结果
    std::vector<ReconstructedPowerLine> result = current_detections;
    
    // 7. 补全丢失的电力线
    std::vector<ReconstructedPowerLine> completed_lines;
    if (enable_completion_) {
        completed_lines = generateCompletedLines();
        result.insert(result.end(), completed_lines.begin(), completed_lines.end());
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end_time - start_time;
    
    ROS_INFO("跟踪器更新完成，耗时: %.3f 秒", duration.count());
    ROS_INFO("活跃轨迹: %zu，当前检测: %zu，输出结果: %zu", 
             active_tracks_.size(), current_detections.size(), result.size());
    // 可视化
    if (enable_track_visualization_) {
    publishTrackVisualization();
    publishCompletedLinesVisualization(completed_lines);
    publishTrackStatistics();
}
    
    return result;
}

void PowerLineTracker::predictAllTracks() {
    for (auto& track : active_tracks_) {
        track.predict();
    }
}
// 数据关联，那些检测对应哪些轨迹
AssociationResult PowerLineTracker::associateDetections(
    const std::vector<ReconstructedPowerLine>& detections,
    const PowerLineProbabilityMap* prob_map) const {
    
    AssociationResult result;
    
    if (active_tracks_.empty() || detections.empty()) {
        // 如果没有轨迹或检测，直接返回空关联
        for (size_t i = 0; i < detections.size(); ++i) {
            result.unmatched_detections.push_back(i);
        }
        for (size_t i = 0; i < active_tracks_.size(); ++i) {
            result.unmatched_tracks.push_back(i);
        }
        return result;
    }
    
    // 构建相似度矩阵
    Eigen::MatrixXf similarity_matrix(active_tracks_.size(), detections.size());
    
    for (size_t i = 0; i < active_tracks_.size(); ++i) {
        for (size_t j = 0; j < detections.size(); ++j) {
            similarity_matrix(i, j) = calculateSimilarity(active_tracks_[i], detections[j], prob_map);
        }
    }
    
    // 转换为代价矩阵（代价 = 1 - 相似度） 转换为代价值，代价越大越不相似，转化为求最小值的最优化问题
    Eigen::MatrixXf cost_matrix = Eigen::MatrixXf::Ones(active_tracks_.size(), detections.size()) - similarity_matrix;
    
    // 使用匈牙利算法求解最优关联  找到相似度最高的轨迹和检测之间的匹配
    std::vector<std::pair<int, int>> assignments = hungarianAssignment(cost_matrix);
    
    // 过滤低于阈值的关联
    for (const auto& assignment : assignments) {
        int track_idx = assignment.first;
        int detection_idx = assignment.second;
        
        if (similarity_matrix(track_idx, detection_idx) >= association_threshold_) {
            result.matched_pairs.push_back(std::make_pair(track_idx, detection_idx));
        } else {
            result.unmatched_tracks.push_back(track_idx);
            result.unmatched_detections.push_back(detection_idx);
        }
    }
    
    // 添加未分配的轨迹和检测
    std::vector<bool> track_matched(active_tracks_.size(), false);
    std::vector<bool> detection_matched(detections.size(), false);
    
    for (const auto& pair : result.matched_pairs) {
        track_matched[pair.first] = true;
        detection_matched[pair.second] = true;
    }
    
    for (size_t i = 0; i < active_tracks_.size(); ++i) {
        if (!track_matched[i]) {
            result.unmatched_tracks.push_back(i);
        }
    }
    
    for (size_t i = 0; i < detections.size(); ++i) {
        if (!detection_matched[i]) {
            result.unmatched_detections.push_back(i);
        }
    }
    
    return result;
}

// 计算轨迹和检测的轨迹的相似度 一方面是通过对不同指标加权分析综合的相似度，另一方面是计算概率地图中预测位置和检测位置的概率大小
float PowerLineTracker::calculateSimilarity(const TrackedPowerLine& track,
                                           const ReconstructedPowerLine& detection,
                                           const PowerLineProbabilityMap* prob_map) const {
    
    float spatial_overlap = calculateSpatialOverlap(track, detection);  //空间重叠相似度
    float direction_similarity = calculateDirectionSimilarity(track, detection);
    float length_similarity = calculateLengthSimilarity(track, detection);
    float shape_similarity = calculateShapeSimilarity(track, detection);
    // 通过对不同指标加权分析综合的相似度
    float basic_similarity = spatial_overlap_weight_ * spatial_overlap +
                            direction_similarity_weight_ * direction_similarity +
                            length_similarity_weight_ * length_similarity +
                            shape_similarity_weight_ * shape_similarity;
    
    // 实际使用概率地图接口进行验证
    if (prob_map != nullptr) {
        float credibility = prob_map->validateDetectionCredibility(detection, 0.3f);
        
        // 同时检查预测区域的概率
        Eigen::Vector3f predicted_center = (track.predicted_state_.start_point + track.predicted_state_.end_point) * 0.5f;
        float region_probability = prob_map->getPredictionRegionProbability(predicted_center, 2.0f);
        
        // 综合概率因子
        float probability_factor = 0.3f + 0.4f * credibility + 0.3f * region_probability;
        basic_similarity *= probability_factor;
    }
    
    return std::max(0.0f, std::min(1.0f, basic_similarity));
}
// 计算空间重叠相似度  通过计算控制点之间的平均距离来衡量
float PowerLineTracker::calculateSpatialOverlap(const TrackedPowerLine& track,
                                               const ReconstructedPowerLine& detection) const {
    
    // 计算控制点之间的平均距离
    std::vector<Eigen::Vector3f> det_control_points = extractControlPoints(detection);
    
    float total_distance = 0.0f;
    for (size_t i = 0; i < std::min(track.state_.control_points.size(), det_control_points.size()); ++i) {
        total_distance += (track.state_.control_points[i] - det_control_points[i]).norm();
    }
    
    float avg_distance = total_distance / std::min(track.state_.control_points.size(), det_control_points.size());
    
    // 距离转相似度（指数衰减）
    return std::exp(-avg_distance / max_association_distance_);
}
// 计算方向相似度
float PowerLineTracker::calculateDirectionSimilarity(const TrackedPowerLine& track,
                                                    const ReconstructedPowerLine& detection) const {
    
    float dot_product = track.state_.main_direction.dot(detection.main_direction);
    return std::max(0.0f, dot_product);  // 确保非负
}
// 计算长度相似度，就是两个线长度的比值
float PowerLineTracker::calculateLengthSimilarity(const TrackedPowerLine& track,
                                                 const ReconstructedPowerLine& detection) const {
    
    float min_length = std::min(track.state_.total_length, (float)detection.total_length);
    float max_length = std::max(track.state_.total_length, (float)detection.total_length);
    
    if (max_length < 1e-6) return 1.0f;  // 避免除零
    
    return min_length / max_length;
}
// 计算形状相似度，基于曲率
float PowerLineTracker::calculateShapeSimilarity(const TrackedPowerLine& track,
                                                const ReconstructedPowerLine& detection) const {

    // 计算检测的曲率
    float detection_curvature = 0.0f;
    if (detection.fitted_curve_points.size() > 2) {
    for (size_t i = 1; i < detection.fitted_curve_points.size() - 1; ++i) {
        Eigen::Vector3f v1 = detection.fitted_curve_points[i] - detection.fitted_curve_points[i-1];
        Eigen::Vector3f v2 = detection.fitted_curve_points[i+1] - detection.fitted_curve_points[i];
        float angle = std::acos(v1.normalized().dot(v2.normalized()));
        detection_curvature += angle;
    }
    detection_curvature /= (detection.fitted_curve_points.size() - 2);
    }
    
    // 比较曲率相似度
    float curvature_diff = std::abs(track.state_.average_curvature - 
                                   detection_curvature);  // 这里应该计算detection的曲率
    
    return std::exp(-curvature_diff * 10.0f);  // 曲率相似度
}
// 提取控制点
std::vector<Eigen::Vector3f> PowerLineTracker::extractControlPoints(const ReconstructedPowerLine& line) const {
    std::vector<Eigen::Vector3f> control_points(5);
    
    if (line.fitted_curve_points.empty()) {
        return control_points;
    }
    
    size_t n = line.fitted_curve_points.size();
    control_points[0] = line.fitted_curve_points[0];
    control_points[1] = line.fitted_curve_points[n * 1 / 4];
    control_points[2] = line.fitted_curve_points[n / 2];
    control_points[3] = line.fitted_curve_points[n * 3 / 4];
    control_points[4] = line.fitted_curve_points[n - 1];
    
    return control_points;
}
// 处理关联结果
void PowerLineTracker::processAssociationResults(const AssociationResult& results,
                                                const std::vector<ReconstructedPowerLine>& detections) {
    
    // 更新匹配的轨迹
    updateMatchedTracks(results.matched_pairs, detections);
    
    // 处理未匹配的轨迹
    handleUnmatchedTracks(results.unmatched_tracks);
    
    // 创建新轨迹
    createNewTracks(results.unmatched_detections, detections);
}



void PowerLineTracker::updateMatchedTracks(const std::vector<std::pair<int, int>>& matched_pairs,
                                         const std::vector<ReconstructedPowerLine>& detections) {
    
    for (const auto& pair : matched_pairs) {
        int track_idx = pair.first;
        int detection_idx = pair.second;
        
        if (track_idx < active_tracks_.size() && detection_idx < detections.size()) {
            active_tracks_[track_idx].update(detections[detection_idx]);
        }
    }
}

void PowerLineTracker::handleUnmatchedTracks(const std::vector<int>& unmatched_tracks) {
    for (int track_idx : unmatched_tracks) {
        if (track_idx < active_tracks_.size()) {
            active_tracks_[track_idx].markAsLost();
            active_tracks_[track_idx].status_ = TRACK_PREDICTED;
        }
    }
}

void PowerLineTracker::createNewTracks(const std::vector<int>& unmatched_detections,
                                     const std::vector<ReconstructedPowerLine>& detections,
                                     const PowerLineProbabilityMap* prob_map) {
    
    for (int detection_idx : unmatched_detections) {
        if (detection_idx < detections.size()) {
            // 验证新检测是否合理
            if (validateNewDetection(detections[detection_idx], prob_map)) {
                TrackedPowerLine new_track(next_track_id_++, detections[detection_idx]);
                active_tracks_.push_back(new_track);
                
                ROS_DEBUG("创建新轨迹 ID: %d", new_track.track_id_);
            }
        }
    }
}

// 验证新检测是否合理
bool PowerLineTracker::validateNewDetection(const ReconstructedPowerLine& detection,
                                           const PowerLineProbabilityMap* prob_map) const {
    
    // 基本验证：长度和点数
    if (detection.total_length < 2.0f || detection.fitted_curve_points.size() < 5) {
        return false;
    }
    
    // 使用概率地图接口进行详细验证
    if (prob_map != nullptr) {
        float credibility = prob_map->validateDetectionCredibility(detection, 0.25f);
        
        // 可信度太低，可能是噪声
        if (credibility < 0.3f) {
            ROS_DEBUG("新检测可信度过低: %.2f", credibility);
            return false;
        }
    }
    
    return true;
}
// 清理无效轨迹
void PowerLineTracker::removeInactiveTracks() {
    active_tracks_.erase(
        std::remove_if(active_tracks_.begin(), active_tracks_.end(),
                      [](const TrackedPowerLine& track) {
                          return track.shouldBeRemoved();
                      }),
        active_tracks_.end());
}
// 限制轨迹的数量
void PowerLineTracker::limitTrackCount() {
    if (active_tracks_.size() <= static_cast<size_t>(max_track_count_)) {
        return;
    }
    
    // 按置信度排序，保留置信度最高的轨迹
    std::sort(active_tracks_.begin(), active_tracks_.end(),
              [](const TrackedPowerLine& a, const TrackedPowerLine& b) {
                  return a.confidence_ > b.confidence_;
              });
    
    // 使用 erase 删除多余的轨迹，而不是 resize
    active_tracks_.erase(active_tracks_.begin() + max_track_count_, active_tracks_.end());
    
    ROS_WARN("轨迹数量超限，已清理到 %d 条", max_track_count_);
}
// 补全丢失的电力线
std::vector<ReconstructedPowerLine> PowerLineTracker::generateCompletedLines() const {
    std::vector<ReconstructedPowerLine> completed_lines;
    
    for (const auto& track : active_tracks_) {
        if (track.status_ == TRACK_PREDICTED && 
            track.confidence_ >= completion_confidence_threshold_ &&
            track.consecutive_misses_ <= 3) {
            
            ReconstructedPowerLine completed = track.generateCompletedLine();
            if (!completed.fitted_curve_points.empty()) {
                completed_lines.push_back(completed);
                
                ROS_DEBUG("补全轨迹 %d，置信度: %.2f", track.track_id_, track.confidence_);
            }
        }
    }
    
    return completed_lines;
}

// 简化版匈牙利算法实现
std::vector<std::pair<int, int>> PowerLineTracker::hungarianAssignment(const Eigen::MatrixXf& cost_matrix) const {
    std::vector<std::pair<int, int>> assignments;  //存储匹配结果
    
    int rows = cost_matrix.rows();  //获取代价矩阵的行数
    int cols = cost_matrix.cols();
    
    // 简化实现：贪心算法（可以后续优化为真正的匈牙利算法）
    std::vector<bool> row_assigned(rows, false);
    std::vector<bool> col_assigned(cols, false);
    
    // 按代价排序所有可能的分配
    std::vector<std::tuple<float, int, int>> candidates; //取出代价值
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            candidates.emplace_back(cost_matrix(i, j), i, j);
        }
    }
    
    std::sort(candidates.begin(), candidates.end());  //按照代价大小排序
    
    // 贪心分配
    for (const auto& candidate : candidates) {
        int row = std::get<1>(candidate);
        int col = std::get<2>(candidate);
        
        if (!row_assigned[row] && !col_assigned[col]) {
            assignments.emplace_back(row, col);
            row_assigned[row] = true;
            col_assigned[col] = true;
        }
    }
    
    return assignments;
}

int PowerLineTracker::getActiveTrackCount() const {
    return active_tracks_.size();
}

std::vector<TrackedPowerLine> PowerLineTracker::getActiveTracksStatus() const {
    return active_tracks_;
}

void PowerLineTracker::clearAllTracks() {
    active_tracks_.clear();
    next_track_id_ = 0;
    is_initialized_ = false;
}

void PowerLineTracker::setProbabilityMapReference(std::shared_ptr<PowerLineProbabilityMap> prob_map) {
    prob_map_ptr_ = prob_map;
}


void PowerLineTracker::publishTrackVisualization() {
    if (!enable_track_visualization_) return;
    
    visualization_msgs::MarkerArray marker_array;
    int marker_id = 0;
    
    for (const auto& track : active_tracks_) {
        // 轨迹线段
        visualization_msgs::Marker line_marker;
        line_marker.header.frame_id = frame_id_;
        line_marker.header.stamp = ros::Time::now();
        line_marker.ns = "track_lines";
        line_marker.id = marker_id++;
        line_marker.type = visualization_msgs::Marker::LINE_STRIP;
        line_marker.action = visualization_msgs::Marker::ADD;
        
        // 轨迹路径
        for (const auto& point : track.state_.control_points) {
            geometry_msgs::Point p;
            p.x = point.x(); p.y = point.y(); p.z = point.z();
            line_marker.points.push_back(p);
        }
        
        line_marker.scale.x = 0.1;  // 线宽
        line_marker.color = getTrackStatusColor(track.status_);
        line_marker.lifetime = ros::Duration(track_visualization_duration_);
        marker_array.markers.push_back(line_marker);
        
        // 轨迹ID文本
        visualization_msgs::Marker text_marker;
        text_marker.header.frame_id = frame_id_;
        text_marker.header.stamp = ros::Time::now();
        text_marker.ns = "track_ids";
        text_marker.id = marker_id++;
        text_marker.type = visualization_msgs::Marker::TEXT_VIEW_FACING;
        text_marker.action = visualization_msgs::Marker::ADD;
        
        Eigen::Vector3f mid_point = (track.state_.start_point + track.state_.end_point) * 0.5f;
        text_marker.pose.position.x = mid_point.x();
        text_marker.pose.position.y = mid_point.y();
        text_marker.pose.position.z = mid_point.z() + 1.0f;
        text_marker.pose.orientation.w = 1.0;
        
        text_marker.scale.z = 0.5f;
        text_marker.color = getTrackStatusColor(track.status_);
        text_marker.text = "T" + std::to_string(track.track_id_) + 
                          " (" + std::to_string((int)(track.confidence_ * 100)) + "%)";
        text_marker.lifetime = ros::Duration(track_visualization_duration_);
        marker_array.markers.push_back(text_marker);
    }
    
    track_markers_pub_.publish(marker_array);
}

std_msgs::ColorRGBA PowerLineTracker::getTrackStatusColor(TrackStatus status) const {
    std_msgs::ColorRGBA color;
    color.a = 0.8f;
    
    switch (status) {
        case TRACK_NEW:
            color.r = 1.0f; color.g = 1.0f; color.b = 0.0f;  // 黄色：新轨迹
            break;
        case TRACK_ACTIVE:
            color.r = 0.0f; color.g = 1.0f; color.b = 0.0f;  // 绿色：活跃
            break;
        case TRACK_PREDICTED:
            color.r = 1.0f; color.g = 0.5f; color.b = 0.0f;  // 橙色：预测
            break;
        case TRACK_LOST:
            color.r = 1.0f; color.g = 0.0f; color.b = 0.0f;  // 红色：丢失
            break;
        case TRACK_CONFIRMED:
            color.r = 0.0f; color.g = 0.0f; color.b = 1.0f;  // 蓝色：确认
            break;
        default:
            color.r = 0.5f; color.g = 0.5f; color.b = 0.5f;  // 灰色：未知
            break;
    }
    
    return color;
}

void PowerLineTracker::publishCompletedLinesVisualization(const std::vector<ReconstructedPowerLine>& completed_lines) {
    if (!enable_track_visualization_ || completed_lines.empty()) return;
    
    visualization_msgs::MarkerArray marker_array;
    int marker_id = 0;
    
    for (const auto& line : completed_lines) {
        visualization_msgs::Marker line_marker;
        line_marker.header.frame_id = frame_id_;
        line_marker.header.stamp = ros::Time::now();
        line_marker.ns = "completed_lines";
        line_marker.id = marker_id++;
        line_marker.type = visualization_msgs::Marker::LINE_STRIP;
        line_marker.action = visualization_msgs::Marker::ADD;
        
        for (const auto& point : line.fitted_curve_points) {
            geometry_msgs::Point p;
            p.x = point.x(); p.y = point.y(); p.z = point.z();
            line_marker.points.push_back(p);
        }
        
        line_marker.scale.x = 0.15;  // 补全线稍粗
        line_marker.color.r = 1.0f; line_marker.color.g = 0.0f; 
        line_marker.color.b = 1.0f; line_marker.color.a = 0.6f;  // 紫色虚线效果
        line_marker.lifetime = ros::Duration(track_visualization_duration_);
        marker_array.markers.push_back(line_marker);
    }
    
    completed_lines_pub_.publish(marker_array);
}

void PowerLineTracker::publishTrackStatistics() {
    if (!enable_track_visualization_) return;
    
    visualization_msgs::Marker text_marker;
    text_marker.header.frame_id = frame_id_;
    text_marker.header.stamp = ros::Time::now();
    text_marker.ns = "track_statistics";
    text_marker.id = 0;
    text_marker.type = visualization_msgs::Marker::TEXT_VIEW_FACING;
    text_marker.action = visualization_msgs::Marker::ADD;
    
    text_marker.pose.position.x = 0.0;
    text_marker.pose.position.y = 0.0;
    text_marker.pose.position.z = 10.0f;  // 高度显示
    text_marker.pose.orientation.w = 1.0;
    
    // 统计信息
    int active_count = 0, predicted_count = 0, confirmed_count = 0;
    float avg_confidence = 0.0f;
    
    for (const auto& track : active_tracks_) {
        if (track.status_ == TRACK_ACTIVE) active_count++;
        else if (track.status_ == TRACK_PREDICTED) predicted_count++;
        else if (track.status_ == TRACK_CONFIRMED) confirmed_count++;
        avg_confidence += track.confidence_;
    }
    
    if (!active_tracks_.empty()) {
        avg_confidence /= active_tracks_.size();
    }
    
    std::ostringstream ss;
    ss << "电力线跟踪统计\n";
    ss << "总轨迹: " << active_tracks_.size() << "\n";
    ss << "活跃: " << active_count << " | 预测: " << predicted_count << " | 确认: " << confirmed_count << "\n";
    ss << "平均置信度: " << std::fixed << std::setprecision(2) << avg_confidence;
    
    text_marker.text = ss.str();
    text_marker.scale.z = 0.8f;
    text_marker.color.r = 1.0f; text_marker.color.g = 1.0f; 
    text_marker.color.b = 1.0f; text_marker.color.a = 1.0f;
    text_marker.lifetime = ros::Duration(1.0);
    
    track_info_pub_.publish(text_marker);
}


