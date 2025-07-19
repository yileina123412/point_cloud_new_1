#include "power_line_reconstruction.h"
#include <chrono>
#include <random>
#include <sstream>
#include <iomanip>

PowerLineReconstructor::PowerLineReconstructor(ros::NodeHandle& nh) : nh_(nh) {
    // 从ROS参数服务器读取参数
    nh.param("reconstruction/separation_distance", separation_distance_, 0.05);
    nh.param("reconstruction/min_segment_points", min_segment_points_, 20.0);
    nh.param("reconstruction/max_connection_distance", max_connection_distance_, 2.0);
    nh.param("reconstruction/max_perpendicular_distance", max_perpendicular_distance_, 0.3);
    nh.param("reconstruction/direction_angle_threshold", direction_angle_threshold_, 0.85);
    nh.param("reconstruction/min_line_length", min_line_length_, 0.3);
    nh.param("reconstruction/spline_resolution", spline_resolution_, 0.1);
    nh.param("reconstruction/enable_visualization", enable_visualization_, true);
    nh.param("reconstruction/enable_separation_visualization", enable_separation_visualization_, false);
    nh.param("reconstruction/parallel_threshold", parallel_threshold_, 0.9);
    nh.param("reconstruction/connection_weight_distance", connection_weight_distance_, 0.6);
    nh.param("reconstruction/connection_weight_angle", connection_weight_angle_, 0.4);
    nh.param("reconstruction/visualization_duration", visualization_duration_, 5.0);
    nh.param("reconstruction/frame_id", frame_id_, std::string("base_link"));

    // 初始化ROS发布器
    original_cloud_pub_ = nh_.advertise<sensor_msgs::PointCloud2>("power_line_reconstruction/original_cloud", 1);
    separated_segments_pub_ = nh_.advertise<sensor_msgs::PointCloud2>("power_line_reconstruction/separated_segments", 1);
    reconstructed_lines_pub_ = nh_.advertise<sensor_msgs::PointCloud2>("power_line_reconstruction/reconstructed_lines", 1);
    curve_markers_pub_ = nh_.advertise<visualization_msgs::MarkerArray>("power_line_reconstruction/curve_markers", 1);
    segment_info_pub_ = nh_.advertise<visualization_msgs::MarkerArray>("power_line_reconstruction/segment_info", 1);
    segment_endpoints_pub_ = nh_.advertise<sensor_msgs::PointCloud2>("power_line_reconstruction/segment_endpoints", 1);
    segment_endpoint_lines_pub_ = nh_.advertise<visualization_msgs::MarkerArray>("power_line_reconstruction/segment_endpoint_lines", 1);

    ROS_INFO("PowerLineReconstructor 初始化参数如下：");
    ROS_INFO("片段分离距离阈值: %f", separation_distance_);
    ROS_INFO("片段最小点数: %f", min_segment_points_);
    ROS_INFO("连接最大距离: %f", max_connection_distance_);
    ROS_INFO("垂直方向最大距离: %f", max_perpendicular_distance_);
    ROS_INFO("方向角度阈值: %f", direction_angle_threshold_);
    ROS_INFO("最小线长度: %f", min_line_length_);
    ROS_INFO("样条拟合分辨率: %f", spline_resolution_);
    ROS_INFO("启用可视化: %s", enable_visualization_ ? "是" : "否");
    ROS_INFO("启用分离可视化: %s", enable_separation_visualization_ ? "是" : "否");
    ROS_INFO("可视化持续时间: %f 秒", visualization_duration_);
    ROS_INFO("坐标系ID: %s", frame_id_.c_str());
    ROS_INFO("平行线判断阈值: %f", parallel_threshold_);
    ROS_INFO("连接判断距离权重: %f", connection_weight_distance_);
    ROS_INFO("连接判断角度权重: %f", connection_weight_angle_);
}

void PowerLineReconstructor::reconstructPowerLines(const pcl::PointCloud<pcl::PointXYZI>::Ptr& input_cloud,
                                                  pcl::PointCloud<pcl::PointXYZI>::Ptr& output_cloud,
                                                  std::vector<ReconstructedPowerLine>& power_lines) {
    // 记录开始时间
    auto start = std::chrono::high_resolution_clock::now();
    
    ROS_INFO("开始电力线重构，输入点云点数: %zu", input_cloud->size());
    
    // 清空输出
    output_cloud->clear();
    power_lines.clear();
    
    if (input_cloud->empty()) {
        ROS_WARN("输入点云为空，跳过处理");
        return;
    }
    
    // 第一步：片段分离
    std::vector<PowerLineSegment> segments;
    separateSegments(input_cloud, segments);
    ROS_INFO("分离得到 %zu 个片段", segments.size());
    
    if (segments.empty()) {
        ROS_WARN("未检测到有效片段");
        return;
    }
    
    // 第二步：创建片段结构体
    createSegmentStructures(segments);

    // 发布片段端点用于检查
    publishSegmentEndpoints(segments);
    
    // 第三步：判断连接性
    std::vector<std::vector<int>> connected_groups;
    judgeConnectivity(segments, connected_groups);
    ROS_INFO("检测到 %zu 个连通组", connected_groups.size());
    
    // 第四步：重构电力线
    reconstructLines(segments, connected_groups, power_lines);
    ROS_INFO("重构得到 %zu 条电力线", power_lines.size());
    
    // 第五步：合并所有有效电力线的点云
    for (const auto& line : power_lines) {
        *output_cloud += *line.points;
    }
    
    output_cloud->width = output_cloud->size();
    output_cloud->height = 1;
    output_cloud->is_dense = true;
    
    // 可视化结果
    if (enable_visualization_) {
        visualizeResults(segments, power_lines);
    }
    
    // 记录结束时间并计算耗时
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    ROS_INFO("电力线重构 执行时间: %f 秒", duration.count());
    ROS_INFO("最终输出点云点数: %zu", output_cloud->size());
}
// 将输入的点云分离成独立的线性片段
void PowerLineReconstructor::separateSegments(const pcl::PointCloud<pcl::PointXYZI>::Ptr& input_cloud,
                                             std::vector<PowerLineSegment>& segments) {
    // 使用欧几里得聚类进行片段分离
    pcl::search::KdTree<pcl::PointXYZI>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZI>());
    tree->setInputCloud(input_cloud);
    
    std::vector<pcl::PointIndices> cluster_indices;
    pcl::EuclideanClusterExtraction<pcl::PointXYZI> ec;
    ec.setClusterTolerance(separation_distance_);
    ec.setMinClusterSize(static_cast<int>(min_segment_points_));
    ec.setMaxClusterSize(input_cloud->size());
    ec.setSearchMethod(tree);
    ec.setInputCloud(input_cloud);
    ec.extract(cluster_indices);
    
    // 为每个聚类创建片段
    segments.clear();
    for (size_t i = 0; i < cluster_indices.size(); ++i) {
        PowerLineSegment segment;
        segment.cluster_id = static_cast<int>(i);
        
        // 提取聚类点云
        pcl::ExtractIndices<pcl::PointXYZI> extract;
        extract.setInputCloud(input_cloud);
        extract.setIndices(boost::make_shared<pcl::PointIndices>(cluster_indices[i]));
        extract.filter(*segment.points);
        
        segments.push_back(segment);
    }
    
    ROS_INFO("欧几里得聚类完成，分离距离: %f，得到 %zu 个片段", separation_distance_, segments.size());
    
    // 可视化分离结果
    if (enable_separation_visualization_) {
        visualizeSeparationResults(input_cloud, segments);
    }
}

void PowerLineReconstructor::createSegmentStructures(std::vector<PowerLineSegment>& segments) {
    for (auto& segment : segments) {
        computeSegmentProperties(segment);
    }
    ROS_INFO("片段属性计算完成");
}
// 计算片段属性
void PowerLineReconstructor::computeSegmentProperties(PowerLineSegment& segment) {
    if (segment.points->empty()) return;
    
    // 计算中心点
    Eigen::Vector4f centroid;
    pcl::compute3DCentroid(*segment.points, centroid);
    segment.center = centroid.head<3>();
    
    // 使用PCA计算主方向
    pcl::PCA<pcl::PointXYZI> pca;
    pca.setInputCloud(segment.points);
    Eigen::Matrix3f eigenvectors = pca.getEigenVectors();
    Eigen::Vector3f eigenvalues = pca.getEigenValues();
    
    // 主方向是第一主成分
    segment.overall_direction = eigenvectors.col(0);
    
    // 计算端点（沿主方向的投影极值）
    float min_proj = std::numeric_limits<float>::max();
    float max_proj = std::numeric_limits<float>::lowest();
    pcl::PointXYZI min_point, max_point;
    
    for (const auto& pt : segment.points->points) {
        Eigen::Vector3f p(pt.x, pt.y, pt.z);
        float proj = (p - segment.center).dot(segment.overall_direction);
        if (proj < min_proj) {
            min_proj = proj;
            min_point = pt;
        }
        if (proj > max_proj) {
            max_proj = proj;
            max_point = pt;
        }
    }
    
    segment.start_point = Eigen::Vector3f(min_point.x, min_point.y, min_point.z);
    segment.end_point = Eigen::Vector3f(max_point.x, max_point.y, max_point.z);
    
    // 计算局部方向（从起点到终点）
    segment.local_direction = (segment.end_point - segment.start_point).normalized();
    
    // 计算长度
    segment.length = (segment.end_point - segment.start_point).norm();

     // 新增：检查与z轴的角度，过滤掉接近垂直的线段
    Eigen::Vector3f z_axis(0, 0, 1);
    float z_angle_cos = std::abs(segment.overall_direction.dot(z_axis));
    float z_angle_deg = std::acos(z_angle_cos) * 180.0 / M_PI;
    
    // 如果与z轴夹角小于20度（即接近垂直），标记为无效
    if (z_angle_deg < 45.0) {
        segment.length = 0.0;  // 通过设置长度为0来标记为无效
        ROS_DEBUG("片段 %d 与z轴角度过小 (%.1f度)，已过滤", segment.cluster_id, z_angle_deg);
    }
}
// 判断连接性
void PowerLineReconstructor::judgeConnectivity(const std::vector<PowerLineSegment>& segments,
                                              std::vector<std::vector<int>>& connected_groups) {
    int n = segments.size();
    std::vector<std::vector<bool>> connectivity_matrix(n, std::vector<bool>(n, false));
    
    // 计算连接矩阵
    for (int i = 0; i < n; ++i) {
        for (int j = i + 1; j < n; ++j) {
            if (isConnectable(segments[i], segments[j])) {
                connectivity_matrix[i][j] = connectivity_matrix[j][i] = true;
            }
        }
    }
    
    // 使用DFS找连通组
    std::vector<bool> visited(n, false);
    connected_groups.clear();
    
    for (int i = 0; i < n; ++i) {
        if (!visited[i]) {
            std::vector<int> current_group;
            dfsConnectivity(i, segments, connectivity_matrix, visited, current_group);
            if (!current_group.empty()) {
                connected_groups.push_back(current_group);
            }
        }
    }
}

// 判断两个线段之间是否能连通
bool PowerLineReconstructor::isConnectable(const PowerLineSegment& seg1, const PowerLineSegment& seg2) {
    // 1. 计算两个片段中心点之间的距离
    // float center_distance = (seg1.center - seg2.center).norm();
    // if (center_distance > max_connection_distance_) {
    //     return false;
    // }

    // 计算所有四种首尾组合距离（seg1的start/end 和 seg2的start/end）
    float d_ss = pointDistance(seg1.start_point, seg2.start_point);
    float d_se = pointDistance(seg1.start_point, seg2.end_point);
    float d_es = pointDistance(seg1.end_point, seg2.start_point);
    float d_ee = pointDistance(seg1.end_point, seg2.end_point);

     // 找到最小距离
    float min_distance = std::min({d_ss, d_se, d_es, d_ee});
    if (min_distance > max_connection_distance_) {
        return false;
    }

    // 计算两聚类体中心连线方向
    Eigen::Vector3f center_line_dir = seg2.center - seg1.center;
    center_line_dir.normalize();

    // 计算两个聚类体整体主方向的角度（单位：度）
    float angle_between_directions = std::acos(seg1.overall_direction.dot(seg2.overall_direction)) * 180.0 / M_PI;

    // 计算聚类体主方向与中心连线方向的夹角（防止两线段首尾相平平行）
    float angle_dir1_center = std::acos(seg1.overall_direction.dot(center_line_dir)) * 180.0 / M_PI;
    float angle_dir2_center = std::acos(seg2.overall_direction.dot(center_line_dir)) * 180.0 / M_PI;
    float max_angle_diff_degree_ = 45.0f;
    // 设置一个阈值，防止平行线段连接（比如大于某个角度认为不是平行）
    // if (std::abs(angle_dir1_center) > max_angle_diff_degree_ || std::abs(angle_dir2_center) > max_angle_diff_degree_) {
    //     // 表示两条线段整体方向与中心线方向过于平行，不连接
    //     return false;
    // }
    
    // // 2. 计算两个片段主方向的角度差（余弦值）
    // float direction_cos = std::abs(seg1.overall_direction.dot(seg2.overall_direction));
    // if (direction_cos < direction_angle_threshold_) {
    //     return false;
    // }
    
    // 3. 计算连接向量与片段方向的垂直距离
    Eigen::Vector3f connection_vec = seg2.center - seg1.center;
    Eigen::Vector3f avg_direction = (seg1.overall_direction + seg2.overall_direction).normalized();
    
    // 计算垂直分量
    float parallel_proj = connection_vec.dot(avg_direction);
    Eigen::Vector3f perpendicular_vec = connection_vec - parallel_proj * avg_direction;
    float perpendicular_distance = perpendicular_vec.norm();
    
    if (perpendicular_distance > max_perpendicular_distance_) {
        return false;
    }
    
    return true;
}


double PowerLineReconstructor::calculateConnectionScore(const PowerLineSegment& seg1, const PowerLineSegment& seg2) {
    // 计算所有端点组合的连接向量
    std::vector<std::pair<Eigen::Vector3f, std::pair<Eigen::Vector3f, Eigen::Vector3f>>> connections = {
        {seg1.start_point - seg2.start_point, {seg1.overall_direction, seg2.overall_direction}},
        {seg1.start_point - seg2.end_point, {seg1.overall_direction, seg2.overall_direction}},
        {seg1.end_point - seg2.start_point, {seg1.overall_direction, seg2.overall_direction}},
        {seg1.end_point - seg2.end_point, {seg1.overall_direction, seg2.overall_direction}}
    };
    
    double best_score = 0.0;
    
    for (const auto& conn : connections) {
        Eigen::Vector3f connection_vec = conn.first;
        Eigen::Vector3f dir1 = conn.second.first.normalized();
        Eigen::Vector3f dir2 = conn.second.second.normalized();
        
        // 使用平均方向作为主方向
        Eigen::Vector3f avg_direction = (dir1 + dir2).normalized();
        
        // 计算连接向量在主方向和垂直方向的分量
        float parallel_distance = std::abs(connection_vec.dot(avg_direction));
        Eigen::Vector3f perpendicular_vec = connection_vec - parallel_distance * avg_direction;
        float perpendicular_distance = perpendicular_vec.norm();
        
        // 检查距离约束
        if (parallel_distance > max_connection_distance_ || 
            perpendicular_distance > max_perpendicular_distance_) {
            continue;  // 超出距离限制，跳过这个连接
        }
        
        // 计算评分
        double parallel_score = std::exp(-parallel_distance / max_connection_distance_);
        double perpendicular_score = std::exp(-perpendicular_distance / max_perpendicular_distance_);
        
        // 方向一致性评分
        double direction_dot = std::abs(dir1.dot(dir2));
        
        // 连接合理性评分（连接方向与片段方向的一致性）
        Eigen::Vector3f connection_dir = connection_vec.normalized();
        double connection_consistency1 = std::abs(connection_dir.dot(dir1));
        double connection_consistency2 = std::abs(connection_dir.dot(dir2));
        double connection_score = (connection_consistency1 + connection_consistency2) / 2.0;
        
        // 综合评分
        double total_score = connection_weight_distance_ * parallel_score * perpendicular_score + 
                            connection_weight_angle_ * direction_dot * connection_score;
        
        best_score = std::max(best_score, total_score);
    }
    
    return best_score;
}
// 用DFS判断连通性
void PowerLineReconstructor::dfsConnectivity(int segment_idx,
                                           const std::vector<PowerLineSegment>& segments,
                                           const std::vector<std::vector<bool>>& connectivity_matrix,
                                           std::vector<bool>& visited,
                                           std::vector<int>& current_group) {
    visited[segment_idx] = true;
    current_group.push_back(segment_idx);
    
    for (size_t i = 0; i < segments.size(); ++i) {
        if (!visited[i] && connectivity_matrix[segment_idx][i]) {
            dfsConnectivity(i, segments, connectivity_matrix, visited, current_group);
        }
    }
}
// 重构电力线
void PowerLineReconstructor::reconstructLines(const std::vector<PowerLineSegment>& segments,
                                             const std::vector<std::vector<int>>& connected_groups,
                                             std::vector<ReconstructedPowerLine>& power_lines) {
    power_lines.clear();
    
    
    for (size_t i = 0; i < connected_groups.size(); ++i) {
        const auto& group = connected_groups[i];

        // 新增：过滤掉包含无效片段的组
        bool has_valid_segments = false;
        for (int idx : group) {
            if (segments[idx].length > 0.0) {  // 长度>0表示有效
                has_valid_segments = true;
                break;
            }
        }
        if (!has_valid_segments) {
            continue;  // 跳过无效组
        }
        
        ReconstructedPowerLine power_line;
        power_line.line_id = static_cast<int>(i);
        power_line.segment_indices = group;
        
        // 合并片段点云
        mergeSegmentPoints(segments, group, power_line.points);
        
        // 计算总长度（所有片段长度之和）
        power_line.total_length = 0.0;
        for (int idx : group) {
            power_line.total_length += segments[idx].length;
        }
        
        // 计算主方向（所有片段方向的平均）
        Eigen::Vector3f avg_direction = Eigen::Vector3f::Zero();
        for (int idx : group) {
            avg_direction += segments[idx].overall_direction;
        }
        power_line.main_direction = avg_direction.normalized();
        
        // 过滤长度太短的线
        if (power_line.total_length >= min_line_length_) {
            // 拟合样条曲线
            fitSplineCurve(power_line);
            power_lines.push_back(power_line);
        }
    }
    
    ROS_INFO("重构完成，过滤后保留 %zu 条电力线（长度 >= %f）", power_lines.size(), min_line_length_);
}

void PowerLineReconstructor::mergeSegmentPoints(const std::vector<PowerLineSegment>& segments,
                                               const std::vector<int>& indices,
                                               pcl::PointCloud<pcl::PointXYZI>::Ptr& merged_cloud) {
    merged_cloud->clear();
    for (int idx : indices) {
        *merged_cloud += *segments[idx].points;
    }
    merged_cloud->width = merged_cloud->size();
    merged_cloud->height = 1;
    merged_cloud->is_dense = true;
}

void PowerLineReconstructor::fitSplineCurve(ReconstructedPowerLine& power_line) {
    power_line.fitted_curve_points.clear();
    
    if (power_line.points->empty()) return;
    
    // 简化的样条拟合：沿主方向采样点
    // 计算中心点
    Eigen::Vector4f centroid;
    pcl::compute3DCentroid(*power_line.points, centroid);
    Eigen::Vector3f center = centroid.head<3>();
    
    // 计算沿主方向的投影范围
    float min_proj = std::numeric_limits<float>::max();
    float max_proj = std::numeric_limits<float>::lowest();
    
    for (const auto& pt : power_line.points->points) {
        Eigen::Vector3f p(pt.x, pt.y, pt.z);
        float proj = (p - center).dot(power_line.main_direction);
        min_proj = std::min(min_proj, proj);
        max_proj = std::max(max_proj, proj);
    }
    
    // 沿主方向生成采样点
    int num_samples = static_cast<int>((max_proj - min_proj) / spline_resolution_) + 1;
    for (int i = 0; i < num_samples; ++i) {
        float t = min_proj + i * spline_resolution_;
        Eigen::Vector3f sample_point = center + t * power_line.main_direction;
        
        // 找到最近的实际点进行修正
        float min_dist = std::numeric_limits<float>::max();
        Eigen::Vector3f closest_point = sample_point;
        
        for (const auto& pt : power_line.points->points) {
            Eigen::Vector3f p(pt.x, pt.y, pt.z);
            float dist = (p - sample_point).norm();
            if (dist < min_dist) {
                min_dist = dist;
                closest_point = p;
            }
        }
        
        // 如果距离合理，使用修正后的点
        if (min_dist < 0.5) {  // 0.5米阈值
            power_line.fitted_curve_points.push_back(closest_point);
        } else {
            power_line.fitted_curve_points.push_back(sample_point);
        }
    }
}

void PowerLineReconstructor::visualizeResults(const std::vector<PowerLineSegment>& segments,
                                             const std::vector<ReconstructedPowerLine>& power_lines) {
    try {
        ROS_INFO("=== 发布电力线重构结果到RViz ===");
        
        // 1. 创建带颜色的重构电力线点云
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr colored_lines(new pcl::PointCloud<pcl::PointXYZRGB>());
        std::vector<Eigen::Vector3f> colors = generateColorPalette(power_lines.size());
        
        for (size_t i = 0; i < power_lines.size(); ++i) {
            const auto& line = power_lines[i];
            const auto& color = colors[i];
            
            for (const auto& pt : line.points->points) {
                pcl::PointXYZRGB colored_pt;
                colored_pt.x = pt.x;
                colored_pt.y = pt.y;
                colored_pt.z = pt.z;
                colored_pt.r = static_cast<uint8_t>(color[0] * 255);
                colored_pt.g = static_cast<uint8_t>(color[1] * 255);
                colored_pt.b = static_cast<uint8_t>(color[2] * 255);
                colored_lines->push_back(colored_pt);
            }
        }
        
        // 发布重构后的电力线点云
        sensor_msgs::PointCloud2 lines_msg;
        pcl::toROSMsg(*colored_lines, lines_msg);
        lines_msg.header.frame_id = frame_id_;
        lines_msg.header.stamp = ros::Time::now();
        reconstructed_lines_pub_.publish(lines_msg);
        
        // 2. 发布拟合曲线标记
        publishCurveMarkers(power_lines);
        
        // 3. 显示统计信息
        ROS_INFO("重构电力线数量: %zu", power_lines.size());
        for (size_t i = 0; i < power_lines.size(); ++i) {
            const auto& line = power_lines[i];
            ROS_INFO("电力线 %d: 包含 %zu 个片段, 总长度 %.2f 米, 点云点数 %zu", 
                    line.line_id, 
                    line.segment_indices.size(), 
                    line.total_length, 
                    line.points->size());
        }
        
        ROS_INFO("电力线重构可视化已发布到以下话题：");
        ROS_INFO("  - 重构电力线: %s", reconstructed_lines_pub_.getTopic().c_str());
        ROS_INFO("  - 拟合曲线: %s", curve_markers_pub_.getTopic().c_str());
        ROS_INFO("请在RViz中添加对应的显示项目以查看结果");
        
        // 等待一段时间确保消息发布
        ros::Duration(0.1).sleep();
        
    } catch (const std::exception& e) {
        ROS_ERROR("电力线重构可视化发布失败: %s", e.what());
    }
}

std::vector<Eigen::Vector3f> PowerLineReconstructor::generateColorPalette(int num_colors) {
    std::vector<Eigen::Vector3f> colors;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(0.3f, 1.0f);  // 避免太暗的颜色
    
    for (int i = 0; i < num_colors; ++i) {
        // 使用HSV色彩空间生成鲜明的颜色
        float hue = static_cast<float>(i) / num_colors * 360.0f;  // 色相
        float saturation = 0.8f;  // 饱和度
        float value = 0.9f;       // 亮度
        
        // HSV转RGB
        float c = value * saturation;
        float x = c * (1 - std::abs(std::fmod(hue / 60.0f, 2) - 1));
        float m = value - c;
        
        float r, g, b;
        if (hue < 60) { r = c; g = x; b = 0; }
        else if (hue < 120) { r = x; g = c; b = 0; }
        else if (hue < 180) { r = 0; g = c; b = x; }
        else if (hue < 240) { r = 0; g = x; b = c; }
        else if (hue < 300) { r = x; g = 0; b = c; }
        else { r = c; g = 0; b = x; }
        
        colors.emplace_back(r + m, g + m, b + m);
    }
    
    return colors;
}

void PowerLineReconstructor::publishCurveMarkers(const std::vector<ReconstructedPowerLine>& power_lines) {
    visualization_msgs::MarkerArray marker_array;
    std::vector<Eigen::Vector3f> colors = generateColorPalette(power_lines.size());
    
    for (size_t i = 0; i < power_lines.size(); ++i) {
        const auto& line = power_lines[i];
        const auto& color = colors[i];
        
        if (line.fitted_curve_points.size() < 2) continue;
        
        // 创建线条标记
        visualization_msgs::Marker line_marker;
        line_marker.header.frame_id = frame_id_;
        line_marker.header.stamp = ros::Time::now();
        line_marker.ns = "power_line_curves";
        line_marker.id = static_cast<int>(i);
        line_marker.type = visualization_msgs::Marker::LINE_STRIP;
        line_marker.action = visualization_msgs::Marker::ADD;
        line_marker.pose.orientation.w = 1.0;
        
        // 设置线条属性
        line_marker.scale.x = 0.05;  // 线宽
        line_marker.color = eigenToColorRGBA(color);
        line_marker.lifetime = ros::Duration(visualization_duration_);
        
        // 添加曲线点
        for (const auto& curve_pt : line.fitted_curve_points) {
            geometry_msgs::Point pt;
            pt.x = curve_pt[0];
            pt.y = curve_pt[1];
            pt.z = curve_pt[2];
            line_marker.points.push_back(pt);
        }
        
        marker_array.markers.push_back(line_marker);
        
        // 添加文本标记显示电力线信息
        visualization_msgs::Marker text_marker;
        text_marker.header.frame_id = frame_id_;
        text_marker.header.stamp = ros::Time::now();
        text_marker.ns = "power_line_info";
        text_marker.id = static_cast<int>(i);
        text_marker.type = visualization_msgs::Marker::TEXT_VIEW_FACING;
        text_marker.action = visualization_msgs::Marker::ADD;
        
        // 文本位置（曲线中点）
        if (!line.fitted_curve_points.empty()) {
            Eigen::Vector3f mid_point = line.fitted_curve_points[line.fitted_curve_points.size() / 2];
            text_marker.pose.position.x = mid_point[0];
            text_marker.pose.position.y = mid_point[1];
            text_marker.pose.position.z = mid_point[2] + 0.5;  // 稍微抬高
        }
        text_marker.pose.orientation.w = 1.0;
        
        // 设置文本属性
        text_marker.scale.z = 0.3;  // 文本大小
        text_marker.color.r = 1.0;
        text_marker.color.g = 1.0;
        text_marker.color.b = 1.0;
        text_marker.color.a = 1.0;
        text_marker.lifetime = ros::Duration(visualization_duration_);
        
        // 文本内容
        std::ostringstream ss;
        ss << "Line " << line.line_id << "\n";
        ss << "Length: " << std::fixed << std::setprecision(1) << line.total_length << "m\n";
        ss << "Segments: " << line.segment_indices.size();
        text_marker.text = ss.str();
        
        marker_array.markers.push_back(text_marker);
    }
    
    curve_markers_pub_.publish(marker_array);
}

void PowerLineReconstructor::publishSegmentInfoMarkers(const std::vector<PowerLineSegment>& segments) {
    visualization_msgs::MarkerArray marker_array;
    std::vector<Eigen::Vector3f> colors = generateColorPalette(segments.size());
    
    for (size_t i = 0; i < segments.size(); ++i) {
        const auto& segment = segments[i];
        const auto& color = colors[i];
        
        // 创建箭头标记显示片段方向
        visualization_msgs::Marker arrow_marker;
        arrow_marker.header.frame_id = frame_id_;
        arrow_marker.header.stamp = ros::Time::now();
        arrow_marker.ns = "segment_directions";
        arrow_marker.id = static_cast<int>(i);
        arrow_marker.type = visualization_msgs::Marker::ARROW;
        arrow_marker.action = visualization_msgs::Marker::ADD;
        
        // 箭头起点和终点
        geometry_msgs::Point start_pt, end_pt;
        start_pt.x = segment.start_point[0];
        start_pt.y = segment.start_point[1];
        start_pt.z = segment.start_point[2];
        end_pt.x = segment.end_point[0];
        end_pt.y = segment.end_point[1];
        end_pt.z = segment.end_point[2];
        
        arrow_marker.points.push_back(start_pt);
        arrow_marker.points.push_back(end_pt);
        
        // 设置箭头属性
        arrow_marker.scale.x = 0.02;  // 箭头杆直径
        arrow_marker.scale.y = 0.04;  // 箭头头部直径
        arrow_marker.scale.z = 0.06;  // 箭头头部长度
        arrow_marker.color = eigenToColorRGBA(color);
        arrow_marker.lifetime = ros::Duration(visualization_duration_);
        
        marker_array.markers.push_back(arrow_marker);
        
        // 创建球体标记显示片段中心
        visualization_msgs::Marker sphere_marker;
        sphere_marker.header.frame_id = frame_id_;
        sphere_marker.header.stamp = ros::Time::now();
        sphere_marker.ns = "segment_centers";
        sphere_marker.id = static_cast<int>(i);
        sphere_marker.type = visualization_msgs::Marker::SPHERE;
        sphere_marker.action = visualization_msgs::Marker::ADD;
        
        sphere_marker.pose.position.x = segment.center[0];
        sphere_marker.pose.position.y = segment.center[1];
        sphere_marker.pose.position.z = segment.center[2];
        sphere_marker.pose.orientation.w = 1.0;
        
        sphere_marker.scale.x = 0.1;
        sphere_marker.scale.y = 0.1;
        sphere_marker.scale.z = 0.1;
        sphere_marker.color = eigenToColorRGBA(color);
        sphere_marker.lifetime = ros::Duration(visualization_duration_);
        
        marker_array.markers.push_back(sphere_marker);
        
        // 创建文本标记显示片段信息
        visualization_msgs::Marker text_marker;
        text_marker.header.frame_id = frame_id_;
        text_marker.header.stamp = ros::Time::now();
        text_marker.ns = "segment_text";
        text_marker.id = static_cast<int>(i);
        text_marker.type = visualization_msgs::Marker::TEXT_VIEW_FACING;
        text_marker.action = visualization_msgs::Marker::ADD;
        
        text_marker.pose.position.x = segment.center[0];
        text_marker.pose.position.y = segment.center[1];
        text_marker.pose.position.z = segment.center[2] + 0.2;
        text_marker.pose.orientation.w = 1.0;
        
        text_marker.scale.z = 0.15;
        text_marker.color.r = 1.0;
        text_marker.color.g = 1.0;
        text_marker.color.b = 1.0;
        text_marker.color.a = 1.0;
        text_marker.lifetime = ros::Duration(visualization_duration_);
        
        std::ostringstream ss;
        ss << "S" << i << "\n";
        ss << std::fixed << std::setprecision(1) << segment.length << "m\n";
        ss << segment.points->size() << "pts";
        text_marker.text = ss.str();
        
        marker_array.markers.push_back(text_marker);
    }
    
    segment_info_pub_.publish(marker_array);
}

std_msgs::ColorRGBA PowerLineReconstructor::eigenToColorRGBA(const Eigen::Vector3f& color, float alpha) {
    std_msgs::ColorRGBA rgba;
    rgba.r = color[0];
    rgba.g = color[1];
    rgba.b = color[2];
    rgba.a = alpha;
    return rgba;
}

void PowerLineReconstructor::visualizeSeparationResults(const pcl::PointCloud<pcl::PointXYZI>::Ptr& input_cloud,
                                                       const std::vector<PowerLineSegment>& segments) {
    try {
        ROS_INFO("=== 发布片段分离结果到RViz ===");
        
        // 1. 发布原始点云（白色）
        sensor_msgs::PointCloud2 original_msg;
        pcl::toROSMsg(*input_cloud, original_msg);
        original_msg.header.frame_id = frame_id_;
        original_msg.header.stamp = ros::Time::now();
        original_cloud_pub_.publish(original_msg);
        
        // 2. 创建带颜色的分离片段点云
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr colored_segments(new pcl::PointCloud<pcl::PointXYZRGB>());
        std::vector<Eigen::Vector3f> colors = generateColorPalette(segments.size());
        
        for (size_t i = 0; i < segments.size(); ++i) {
            const auto& segment = segments[i];
            const auto& color = colors[i];
            
            for (const auto& pt : segment.points->points) {
                pcl::PointXYZRGB colored_pt;
                colored_pt.x = pt.x;
                colored_pt.y = pt.y;
                colored_pt.z = pt.z;
                colored_pt.r = static_cast<uint8_t>(color[0] * 255);
                colored_pt.g = static_cast<uint8_t>(color[1] * 255);
                colored_pt.b = static_cast<uint8_t>(color[2] * 255);
                colored_segments->push_back(colored_pt);
            }
        }
        
        // 发布带颜色的分离片段
        sensor_msgs::PointCloud2 segments_msg;
        pcl::toROSMsg(*colored_segments, segments_msg);
        segments_msg.header.frame_id = frame_id_;
        segments_msg.header.stamp = ros::Time::now();
        separated_segments_pub_.publish(segments_msg);
        
        // 3. 发布片段信息标记
        publishSegmentInfoMarkers(segments);
        
        // 4. 显示统计信息
        ROS_INFO("输入点云总点数: %zu", input_cloud->size());
        ROS_INFO("分离后片段数量: %zu", segments.size());
        
        int total_separated_points = 0;
        for (size_t i = 0; i < segments.size(); ++i) {
            int points_count = segments[i].points->size();
            total_separated_points += points_count;
            ROS_INFO("片段 %zu: %d 个点", i, points_count);
        }
        ROS_INFO("分离后总点数: %d", total_separated_points);
        
        if (total_separated_points < static_cast<int>(input_cloud->size())) {
            int lost_points = input_cloud->size() - total_separated_points;
            ROS_WARN("注意：有 %d 个点在分离过程中被过滤掉（可能因为片段太小）", lost_points);
        }
        
        ROS_INFO("片段分离可视化已发布到以下话题：");
        ROS_INFO("  - 原始点云: %s", original_cloud_pub_.getTopic().c_str());
        ROS_INFO("  - 分离片段: %s", separated_segments_pub_.getTopic().c_str());
        ROS_INFO("  - 片段信息: %s", segment_info_pub_.getTopic().c_str());
        ROS_INFO("请在RViz中添加对应的显示项目以查看结果");
        
        // 等待一段时间确保消息发布
        ros::Duration(0.1).sleep();
        
    } catch (const std::exception& e) {
        ROS_ERROR("片段分离可视化发布失败: %s", e.what());
    }
}
void PowerLineReconstructor::publishSegmentEndpoints(const std::vector<PowerLineSegment>& segments) {
    // 1. 发布端点点云
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr endpoints_cloud(new pcl::PointCloud<pcl::PointXYZRGB>());
    
    for (size_t i = 0; i < segments.size(); ++i) {
        const auto& segment = segments[i];
        
        // 起始点 - 绿色
        pcl::PointXYZRGB start_pt;
        start_pt.x = segment.start_point[0];
        start_pt.y = segment.start_point[1];
        start_pt.z = segment.start_point[2];
        start_pt.r = 0;
        start_pt.g = 255;
        start_pt.b = 0;
        endpoints_cloud->push_back(start_pt);
        
        // 终点 - 红色
        pcl::PointXYZRGB end_pt;
        end_pt.x = segment.end_point[0];
        end_pt.y = segment.end_point[1];
        end_pt.z = segment.end_point[2];
        end_pt.r = 255;
        end_pt.g = 0;
        end_pt.b = 0;
        endpoints_cloud->push_back(end_pt);
    }
    
    // 发布端点点云
    sensor_msgs::PointCloud2 endpoints_msg;
    pcl::toROSMsg(*endpoints_cloud, endpoints_msg);
    endpoints_msg.header.frame_id = frame_id_;
    endpoints_msg.header.stamp = ros::Time::now();
    segment_endpoints_pub_.publish(endpoints_msg);
    
    // 2. 发布连线标记
    visualization_msgs::MarkerArray marker_array;
    std::vector<Eigen::Vector3f> colors = generateColorPalette(segments.size());
    
    for (size_t i = 0; i < segments.size(); ++i) {
        const auto& segment = segments[i];
        const auto& color = colors[i];
        
        // 创建连线标记
        visualization_msgs::Marker line_marker;
        line_marker.header.frame_id = frame_id_;
        line_marker.header.stamp = ros::Time::now();
        line_marker.ns = "segment_endpoint_lines";
        line_marker.id = static_cast<int>(i);
        line_marker.type = visualization_msgs::Marker::LINE_STRIP;
        line_marker.action = visualization_msgs::Marker::ADD;
        line_marker.pose.orientation.w = 1.0;
        
        // 设置线条属性
        line_marker.scale.x = 0.03;  // 线宽，稍细一些
        line_marker.color = eigenToColorRGBA(color, 0.8f);  // 稍透明
        line_marker.lifetime = ros::Duration(visualization_duration_);
        
        // 添加起点和终点
        geometry_msgs::Point start_point, end_point;
        start_point.x = segment.start_point[0];
        start_point.y = segment.start_point[1];
        start_point.z = segment.start_point[2];
        end_point.x = segment.end_point[0];
        end_point.y = segment.end_point[1];
        end_point.z = segment.end_point[2];
        
        line_marker.points.push_back(start_point);
        line_marker.points.push_back(end_point);
        
        marker_array.markers.push_back(line_marker);
        
        // 添加片段ID文本标记（在线的中点）
        visualization_msgs::Marker text_marker;
        text_marker.header.frame_id = frame_id_;
        text_marker.header.stamp = ros::Time::now();
        text_marker.ns = "segment_endpoint_text";
        text_marker.id = static_cast<int>(i);
        text_marker.type = visualization_msgs::Marker::TEXT_VIEW_FACING;
        text_marker.action = visualization_msgs::Marker::ADD;
        
        // 文本位置（线段中点）
        Eigen::Vector3f mid_point = (segment.start_point + segment.end_point) / 2.0f;
        text_marker.pose.position.x = mid_point[0];
        text_marker.pose.position.y = mid_point[1];
        text_marker.pose.position.z = mid_point[2] + 0.1;  // 稍微抬高
        text_marker.pose.orientation.w = 1.0;
        
        // 设置文本属性
        text_marker.scale.z = 0.2;  // 文本大小
        text_marker.color.r = 1.0;
        text_marker.color.g = 1.0;
        text_marker.color.b = 1.0;
        text_marker.color.a = 1.0;
        text_marker.lifetime = ros::Duration(visualization_duration_);
        
        // 文本内容
        std::ostringstream ss;
        ss << "S" << i << "\n" << std::fixed << std::setprecision(2) << segment.length << "m";
        text_marker.text = ss.str();
        
        marker_array.markers.push_back(text_marker);
    }
    
    // 发布连线标记到专门的话题
    segment_endpoint_lines_pub_.publish(marker_array);
    
    ROS_INFO("已发布 %zu 个片段的端点和连线（绿色=起点，红色=终点，彩色线条=片段范围）", segments.size());
}