#include "building_edge_filter.h"
#include <chrono>
#include <sstream>
#include <iomanip>

BuildingEdgeFilter::BuildingEdgeFilter(ros::NodeHandle& nh) : nh_(nh),
debug_cloud((new pcl::PointCloud<pcl::PointXYZRGB>())) {
    // 从ROS参数服务器读取参数
    nh.param("building_edge_filter/cylinder_radius", cylinder_radius_, 0.3f);
    nh.param("building_edge_filter/end_exclusion_distance", end_exclusion_distance_, 0.5f);
    nh.param("building_edge_filter/min_plane_area", min_plane_area_, 2.0f);
    nh.param("building_edge_filter/min_plane_density", min_plane_density_, 100.0f);
    nh.param("building_edge_filter/plane_distance_threshold", plane_distance_threshold_, 0.05f);
    nh.param("building_edge_filter/min_plane_points", min_plane_points_, 50);
    nh.param("building_edge_filter/coordinate_tolerance", coordinate_tolerance_, 0.01f);
    nh.param("building_edge_filter/enable_debug_output", enable_debug_output_, true);
    nh.param("building_edge_filter/enable_visualization", enable_visualization_, true);
    nh.param("building_edge_filter/frame_id", frame_id_, std::string("base_link"));
    nh.param("building_edge_filter/visualization_duration", visualization_duration_, 10.0);

    // 初始化ROS发布器
    filtered_lines_pub_ = nh_.advertise<sensor_msgs::PointCloud2>("building_edge_filter/filtered_lines", 1);
    filtered_info_pub_ = nh_.advertise<visualization_msgs::MarkerArray>("building_edge_filter/filter_info", 1);

    // 添加这一行：
    cylinder_debug_pub_ = nh_.advertise<sensor_msgs::PointCloud2>("building_edge_filter/cylinder_debug", 1);

    ROS_INFO("BuildingEdgeFilter 初始化参数如下：");
    ROS_INFO("圆柱半径: %.3f m", cylinder_radius_);
    ROS_INFO("头尾排除距离: %.3f m", end_exclusion_distance_);
    ROS_INFO("最小平面面积阈值: %.3f m²", min_plane_area_);
    ROS_INFO("最小平面密度阈值: %.3f pts/m²", min_plane_density_);
    ROS_INFO("平面拟合距离阈值: %.3f m", plane_distance_threshold_);
    ROS_INFO("平面最小点数: %d", min_plane_points_);
    ROS_INFO("坐标匹配容差: %.3f m", coordinate_tolerance_);
    ROS_INFO("启用调试输出: %s", enable_debug_output_ ? "是" : "否");
    ROS_INFO("启用可视化: %s", enable_visualization_ ? "是" : "否");
    ROS_INFO("坐标系ID: %s", frame_id_.c_str());
    ROS_INFO("可视化持续时间: %.1f 秒", visualization_duration_);
}

bool BuildingEdgeFilter::filterBuildingEdges(
    const pcl::PointCloud<pcl::PointXYZI>::Ptr& environment_cloud,
    std::vector<ReconstructedPowerLine>& power_lines,
    pcl::PointCloud<pcl::PointXYZI>::Ptr& output_cloud) {
    
    // 记录开始时间
    auto start = std::chrono::high_resolution_clock::now();

    debug_cloud->clear();
    
    ROS_INFO("开始建筑物边缘过滤，输入电力线数量: %zu", power_lines.size());
    
    // 验证输入电力线的完整性
    validatePowerLinesIntegrity(power_lines, "输入阶段");
    
    // 清空输出点云
    output_cloud->clear();
    
    if (environment_cloud->empty()) {
        ROS_WARN("环境点云为空，跳过过滤");
        return false;
    }

    if (power_lines.empty()) {
        ROS_WARN("电力线列表为空，跳过过滤");
        return false;
    }

    // 保存原始电力线用于可视化
    std::vector<ReconstructedPowerLine> original_lines = power_lines;

    // 统计变量
    int total_lines = power_lines.size();
    int filtered_lines = 0;
    std::vector<bool> valid_flags(total_lines, true);
    std::vector<ReconstructedPowerLine> filtered_out_lines;

    // 逐一检查每条电力线
    for (size_t i = 0; i < power_lines.size(); ++i) {
        // 检查电力线完整性
        if (!power_lines[i].points || power_lines[i].points->empty()) {
            ROS_WARN("电力线 %d 点云为空，标记为无效", power_lines[i].line_id);
            valid_flags[i] = false;
            filtered_lines++;
            continue;
        }
        
        ROS_INFO("检查电力线 %d (点数: %zu, 长度: %.2fm)", 
                power_lines[i].line_id, power_lines[i].points->size(), power_lines[i].total_length);
        
        bool is_valid = checkSinglePowerLine(environment_cloud, power_lines[i]);
        valid_flags[i] = is_valid;
        
        if (!is_valid) {
            filtered_lines++;
            filtered_out_lines.push_back(power_lines[i]);
            ROS_INFO("电力线 %d 被识别为建筑物边缘，已过滤 (原点数: %zu)", 
                    power_lines[i].line_id, power_lines[i].points->size());
        } else {
            ROS_INFO("电力线 %d 判定为有效电力线 (点数: %zu)", 
                    power_lines[i].line_id, power_lines[i].points->size());
        }
    }

    // 发布调试点云
    sensor_msgs::PointCloud2 debug_msg;
    pcl::toROSMsg(*debug_cloud, debug_msg);
    debug_msg.header.frame_id = frame_id_;
    debug_msg.header.stamp = ros::Time::now();
    cylinder_debug_pub_.publish(debug_msg);
    
    ROS_INFO("区域外的点云共%zu 圆柱区域调试点云已发布到话题: %s (橙色显示)", 
            debug_cloud->size(), cylinder_debug_pub_.getTopic().c_str());

    // 移除被过滤的电力线（保持原有电力线完整性）
    std::vector<ReconstructedPowerLine> valid_lines;
    for (size_t i = 0; i < valid_flags.size(); ++i) {
        if (valid_flags[i]) {
            valid_lines.push_back(power_lines[i]);
            ROS_INFO("保留电力线 %d (点数: %zu)", 
                    power_lines[i].line_id, power_lines[i].points->size());
        } else {
            ROS_INFO("移除电力线 %d (点数: %zu)", 
                    power_lines[i].line_id, power_lines[i].points->size());
        }
    }
    
    // 更新power_lines数组
    power_lines = valid_lines;
    
    // 验证过滤后电力线的完整性
    validatePowerLinesIntegrity(power_lines, "过滤后");

    // 合并所有有效电力线的点云到输出点云
    mergePointClouds(power_lines, output_cloud);

    // 可视化结果
    if (enable_visualization_) {
        visualizeFilterResults(power_lines, filtered_out_lines);
    }

    // 记录结束时间并输出统计信息
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    
    ROS_INFO("=== 建筑物边缘过滤完成 ===");
    ROS_INFO("处理时间: %.3f 秒", duration.count());
    ROS_INFO("原始电力线数量: %d", total_lines);
    ROS_INFO("过滤掉的电力线数量: %d", filtered_lines);
    ROS_INFO("保留的电力线数量: %zu", power_lines.size());
    ROS_INFO("输出点云总点数: %zu", output_cloud->size());

    return true;
}

bool BuildingEdgeFilter::checkSinglePowerLine(
    const pcl::PointCloud<pcl::PointXYZI>::Ptr& environment_cloud,
    const ReconstructedPowerLine& power_line) {
    
    if (power_line.points->empty()) {
        ROS_WARN("电力线 %d 点云为空，跳过检查", power_line.line_id);
        return true;
    }

    int original_points = power_line.points->size();
    ROS_INFO("检查电力线 %d，原始点数: %d", power_line.line_id, original_points);

    // 创建圆柱形区域点云
    pcl::PointCloud<pcl::PointXYZI>::Ptr cylinder_cloud = 
        createCylinderRegion(environment_cloud, power_line);

    if (cylinder_cloud->empty()) {
        if (enable_debug_output_) {
            ROS_DEBUG("电力线 %d 圆柱区域无点云，认为是有效电力线", power_line.line_id);
        }
        return true;
    }

    ROS_INFO("电力线 %d 圆柱区域找到 %zu 个点", power_line.line_id, cylinder_cloud->size());

    // 移除电力线自身的点
    removePowerLinePoints(cylinder_cloud, power_line);

    ROS_INFO("电力线 %d 移除自身点后剩余 %zu 个点", power_line.line_id, cylinder_cloud->size());

    // 发布圆柱区域调试点云
    if (enable_visualization_ && !cylinder_cloud->empty()) {
        
        
        for (const auto& pt : cylinder_cloud->points) {
            pcl::PointXYZRGB colored_pt;
            colored_pt.x = pt.x;
            colored_pt.y = pt.y;
            colored_pt.z = pt.z;
            // 使用橙色显示圆柱区域剩余点云
            colored_pt.r = 255;
            colored_pt.g = 165;
            colored_pt.b = 0;
            debug_cloud->push_back(colored_pt);
        }
        
        
    }

    if (cylinder_cloud->empty()) {
        if (enable_debug_output_) {
            ROS_DEBUG("电力线 %d 移除自身点后圆柱区域无点云，认为是有效电力线", power_line.line_id);
        }
        return true;
    }

    // 检测大片平面
    float plane_area = 0.0f;
    float plane_density = 0.0f;
    bool has_large_plane = detectLargePlane(cylinder_cloud, plane_area, plane_density);

    // 输出调试信息
    if (enable_debug_output_) {
        ROS_INFO("电力线 %d 检测结果：", power_line.line_id);
        ROS_INFO("  - 原始点数: %d", original_points);
        ROS_INFO("  - 圆柱区域点数: %zu", cylinder_cloud->size());
        ROS_INFO("  - 检测到大片平面: %s", has_large_plane ? "是" : "否");
        ROS_INFO("  - 平面面积: %.3f m²", plane_area);
        ROS_INFO("  - 平面密度: %.3f pts/m²", plane_density);
    }

    // 重要：确认电力线自身的点云没有被修改
    if (static_cast<int>(power_line.points->size()) != original_points) {
        ROS_ERROR("警告！电力线 %d 的点云被意外修改！原始: %d, 当前: %zu", 
                 power_line.line_id, original_points, power_line.points->size());
    }

    // 判断是否为建筑物边缘
    if (has_large_plane) {
        ROS_INFO("电力线 %d 附近检测到大片平面（面积: %.3f m²，密度: %.3f pts/m²），判定为建筑物边缘", 
                power_line.line_id, plane_area, plane_density);
        return false;  // 是建筑物边缘
    }

    ROS_INFO("电力线 %d 判定为有效电力线", power_line.line_id);
    return true;  // 是有效电力线
}

pcl::PointCloud<pcl::PointXYZI>::Ptr BuildingEdgeFilter::createCylinderRegion(
    const pcl::PointCloud<pcl::PointXYZI>::Ptr& environment_cloud,
    const ReconstructedPowerLine& power_line) {
    
    pcl::PointCloud<pcl::PointXYZI>::Ptr cylinder_cloud(new pcl::PointCloud<pcl::PointXYZI>);

    // 遍历环境点云中的所有点
    for (const auto& point : environment_cloud->points) {
        // 检查点是否在电力线的中间部分
        if (!isInMiddleSection(point, power_line)) {
            continue;
        }

        // 计算点到电力线轴线的距离
        float distance_to_axis = calculateDistanceToAxis(point, power_line);

        // 如果距离在圆柱半径内，加入圆柱区域
        if (distance_to_axis <= cylinder_radius_) {
            cylinder_cloud->push_back(point);
        }
    }

    return cylinder_cloud;
}

void BuildingEdgeFilter::removePowerLinePoints(
    pcl::PointCloud<pcl::PointXYZI>::Ptr& cylinder_cloud,
    const ReconstructedPowerLine& power_line) {
    
    if (cylinder_cloud->empty() || power_line.points->empty()) {
        return;
    }

    // 创建临时点云存储要保留的点
    pcl::PointCloud<pcl::PointXYZI>::Ptr filtered_cloud(new pcl::PointCloud<pcl::PointXYZI>);

    // 遍历圆柱区域中的每个点
    for (const auto& cyl_point : cylinder_cloud->points) {
        bool is_powerline_point = false;

        // 检查是否与电力线中的任何点匹配
        for (const auto& pl_point : power_line.points->points) {
            if (arePointsClose(cyl_point, pl_point, coordinate_tolerance_)) {
                is_powerline_point = true;
                break;
            }
        }

        // 如果不是电力线点，保留
        if (!is_powerline_point) {
            filtered_cloud->push_back(cyl_point);
        }
    }

    // 更新圆柱区域点云
    *cylinder_cloud = *filtered_cloud;
}

bool BuildingEdgeFilter::detectLargePlane(
    const pcl::PointCloud<pcl::PointXYZI>::Ptr& neighborhood_cloud,
    float& plane_area,
    float& plane_density) {
    
    plane_area = 0.0f;
    plane_density = 0.0f;

    if (neighborhood_cloud->size() < static_cast<size_t>(min_plane_points_)) {
        return false;
    }

    // 使用RANSAC进行平面拟合
    pcl::SACSegmentation<pcl::PointXYZI> seg;
    pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
    pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);

    seg.setOptimizeCoefficients(true);
    seg.setModelType(pcl::SACMODEL_PLANE);
    seg.setMethodType(pcl::SAC_RANSAC);
    seg.setDistanceThreshold(plane_distance_threshold_);
    seg.setInputCloud(neighborhood_cloud);
    seg.segment(*inliers, *coefficients);

    if (inliers->indices.size() < static_cast<size_t>(min_plane_points_)) {
        return false;
    }

    // 提取平面点
    pcl::ExtractIndices<pcl::PointXYZI> extract;
    pcl::PointCloud<pcl::PointXYZI>::Ptr plane_cloud(new pcl::PointCloud<pcl::PointXYZI>);
    extract.setInputCloud(neighborhood_cloud);
    extract.setIndices(inliers);
    extract.setNegative(false);
    extract.filter(*plane_cloud);

    if (plane_cloud->empty()) {
        return false;
    }

    // 计算平面面积
    Eigen::Vector4f plane_coeffs(coefficients->values[0], coefficients->values[1],
                                coefficients->values[2], coefficients->values[3]);
    plane_area = calculatePlaneArea(plane_cloud, plane_coeffs);

    // 计算平面密度
    if (plane_area > 0.0f) {
        plane_density = static_cast<float>(plane_cloud->size()) / plane_area;
    }

    // 判断是否为大片平面
    return (plane_area >= min_plane_area_ && plane_density >= min_plane_density_);
}

float BuildingEdgeFilter::calculateDistanceToAxis(
    const pcl::PointXYZI& point,
    const ReconstructedPowerLine& power_line) {
    
    if (power_line.points->empty()) {
        return std::numeric_limits<float>::max();
    }

    // 计算电力线的中心点
    Eigen::Vector4f centroid;
    pcl::compute3DCentroid(*power_line.points, centroid);
    Eigen::Vector3f center = centroid.head<3>();

    // 点到电力线中心的向量
    Eigen::Vector3f point_vec(point.x, point.y, point.z);
    Eigen::Vector3f to_point = point_vec - center;

    // 计算到轴线的距离（点到直线的距离）
    Eigen::Vector3f main_dir = power_line.main_direction.normalized();
    float projection = to_point.dot(main_dir);
    Eigen::Vector3f perpendicular = to_point - projection * main_dir;

    return perpendicular.norm();
}

bool BuildingEdgeFilter::isInMiddleSection(
    const pcl::PointXYZI& point,
    const ReconstructedPowerLine& power_line) {
    
    if (power_line.points->empty()) {
        return false;
    }

    // 计算电力线的中心点
    Eigen::Vector4f centroid;
    pcl::compute3DCentroid(*power_line.points, centroid);
    Eigen::Vector3f center = centroid.head<3>();

    // 点到电力线中心的向量
    Eigen::Vector3f point_vec(point.x, point.y, point.z);
    Eigen::Vector3f to_point = point_vec - center;

    // 计算沿主方向的投影
    Eigen::Vector3f main_dir = power_line.main_direction.normalized();
    float projection = to_point.dot(main_dir);

    // 计算电力线的投影范围
    float min_proj = std::numeric_limits<float>::max();
    float max_proj = std::numeric_limits<float>::lowest();

    for (const auto& pl_point : power_line.points->points) {
        Eigen::Vector3f pl_vec(pl_point.x, pl_point.y, pl_point.z);
        float pl_proj = (pl_vec - center).dot(main_dir);
        min_proj = std::min(min_proj, pl_proj);
        max_proj = std::max(max_proj, pl_proj);
    }

    // 排除头尾区域
    float effective_min = min_proj + end_exclusion_distance_;
    float effective_max = max_proj - end_exclusion_distance_;

    // 检查点是否在有效范围内
    return (projection >= effective_min && projection <= effective_max);
}

float BuildingEdgeFilter::calculatePlaneArea(
    const pcl::PointCloud<pcl::PointXYZI>::Ptr& plane_points,
    const Eigen::Vector4f& plane_coefficients) {
    
    if (plane_points->size() < 3) {
        return 0.0f;
    }

    // 将点投影到平面上，然后计算凸包面积
    // 简化方法：使用点的边界框估算面积
    Eigen::Vector4f min_pt, max_pt;
    pcl::getMinMax3D(*plane_points, min_pt, max_pt);

    // 计算三个方向的跨度
    float dx = max_pt[0] - min_pt[0];
    float dy = max_pt[1] - min_pt[1];
    float dz = max_pt[2] - min_pt[2];

    // 根据平面法向量确定主要面积
    Eigen::Vector3f normal(plane_coefficients[0], plane_coefficients[1], plane_coefficients[2]);
    normal = normal.normalized();

    float area_xy = dx * dy * std::abs(normal[2]);
    float area_xz = dx * dz * std::abs(normal[1]);
    float area_yz = dy * dz * std::abs(normal[0]);

    return std::max({area_xy, area_xz, area_yz});
}

bool BuildingEdgeFilter::arePointsClose(
    const pcl::PointXYZI& p1,
    const pcl::PointXYZI& p2,
    float tolerance) {
    
    float dx = p1.x - p2.x;
    float dy = p1.y - p2.y;
    float dz = p1.z - p2.z;
    float distance = std::sqrt(dx*dx + dy*dy + dz*dz);

    return distance <= tolerance;
}

void BuildingEdgeFilter::visualizeFilterResults(
    const std::vector<ReconstructedPowerLine>& valid_lines,
    const std::vector<ReconstructedPowerLine>& filtered_lines) {
    
    try {
        ROS_INFO("=== 发布建筑物边缘过滤结果到RViz ===");
        
        // 1. 创建带颜色的过滤后电力线点云
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr colored_lines(new pcl::PointCloud<pcl::PointXYZRGB>());
        
        // 有效电力线 - 绿色
        for (const auto& line : valid_lines) {
            for (const auto& pt : line.points->points) {
                pcl::PointXYZRGB colored_pt;
                colored_pt.x = pt.x;
                colored_pt.y = pt.y;
                colored_pt.z = pt.z;
                colored_pt.r = 0;
                colored_pt.g = 255;
                colored_pt.b = 0;
                colored_lines->push_back(colored_pt);
            }
        }
        
        // 被过滤的电力线 - 红色
        for (const auto& line : filtered_lines) {
            for (const auto& pt : line.points->points) {
                pcl::PointXYZRGB colored_pt;
                colored_pt.x = pt.x;
                colored_pt.y = pt.y;
                colored_pt.z = pt.z;
                colored_pt.r = 255;
                colored_pt.g = 0;
                colored_pt.b = 0;
                colored_lines->push_back(colored_pt);
            }
        }
        
        // 发布过滤结果点云
        sensor_msgs::PointCloud2 lines_msg;
        pcl::toROSMsg(*colored_lines, lines_msg);
        lines_msg.header.frame_id = frame_id_;
        lines_msg.header.stamp = ros::Time::now();
        filtered_lines_pub_.publish(lines_msg);
        
        // 2. 发布过滤信息标记
        visualization_msgs::MarkerArray marker_array;
        
        // 有效电力线标记
        for (size_t i = 0; i < valid_lines.size(); ++i) {
            const auto& line = valid_lines[i];
            
            // 文本标记
            visualization_msgs::Marker text_marker;
            text_marker.header.frame_id = frame_id_;
            text_marker.header.stamp = ros::Time::now();
            text_marker.ns = "valid_lines";
            text_marker.id = static_cast<int>(i);
            text_marker.type = visualization_msgs::Marker::TEXT_VIEW_FACING;
            text_marker.action = visualization_msgs::Marker::ADD;
            
            // 计算中心点
            if (!line.points->empty()) {
                Eigen::Vector4f centroid;
                pcl::compute3DCentroid(*line.points, centroid);
                text_marker.pose.position.x = centroid[0];
                text_marker.pose.position.y = centroid[1];
                text_marker.pose.position.z = centroid[2] + 0.5;
            }
            text_marker.pose.orientation.w = 1.0;
            
            text_marker.scale.z = 0.3;
            text_marker.color.r = 0.0;
            text_marker.color.g = 1.0;
            text_marker.color.b = 0.0;
            text_marker.color.a = 1.0;
            text_marker.lifetime = ros::Duration(visualization_duration_);
            
            std::ostringstream ss;
            ss << "有效电力线 " << line.line_id << "\n";
            ss << "长度: " << std::fixed << std::setprecision(1) << line.total_length << "m";
            text_marker.text = ss.str();
            
            marker_array.markers.push_back(text_marker);
        }
        
        // 被过滤电力线标记
        for (size_t i = 0; i < filtered_lines.size(); ++i) {
            const auto& line = filtered_lines[i];
            
            // 文本标记
            visualization_msgs::Marker text_marker;
            text_marker.header.frame_id = frame_id_;
            text_marker.header.stamp = ros::Time::now();
            text_marker.ns = "filtered_lines";
            text_marker.id = static_cast<int>(i);
            text_marker.type = visualization_msgs::Marker::TEXT_VIEW_FACING;
            text_marker.action = visualization_msgs::Marker::ADD;
            
            // 计算中心点
            if (!line.points->empty()) {
                Eigen::Vector4f centroid;
                pcl::compute3DCentroid(*line.points, centroid);
                text_marker.pose.position.x = centroid[0];
                text_marker.pose.position.y = centroid[1];
                text_marker.pose.position.z = centroid[2] + 0.5;
            }
            text_marker.pose.orientation.w = 1.0;
            
            text_marker.scale.z = 0.3;
            text_marker.color.r = 1.0;
            text_marker.color.g = 0.0;
            text_marker.color.b = 0.0;
            text_marker.color.a = 1.0;
            text_marker.lifetime = ros::Duration(visualization_duration_);
            
            std::ostringstream ss;
            ss << "已过滤 " << line.line_id << "\n";
            ss << "建筑物边缘";
            text_marker.text = ss.str();
            
            marker_array.markers.push_back(text_marker);
        }
        
        filtered_info_pub_.publish(marker_array);
        
        ROS_INFO("可视化已发布到以下话题：");
        ROS_INFO("  - 过滤结果点云: %s", filtered_lines_pub_.getTopic().c_str());
        ROS_INFO("  - 过滤信息标记: %s", filtered_info_pub_.getTopic().c_str());
        ROS_INFO("  - 绿色: 有效电力线, 红色: 被过滤的建筑物边缘");
        
        ros::Duration(0.1).sleep();
        
    } catch (const std::exception& e) {
        ROS_ERROR("建筑物边缘过滤可视化发布失败: %s", e.what());
    }
}

std::vector<Eigen::Vector3f> BuildingEdgeFilter::generateColorPalette(int num_colors) {
    std::vector<Eigen::Vector3f> colors;
    
    for (int i = 0; i < num_colors; ++i) {
        float hue = static_cast<float>(i) / num_colors * 360.0f;
        float saturation = 0.8f;
        float value = 0.9f;
        
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

std_msgs::ColorRGBA BuildingEdgeFilter::eigenToColorRGBA(const Eigen::Vector3f& color, float alpha) {
    std_msgs::ColorRGBA rgba;
    rgba.r = color[0];
    rgba.g = color[1];
    rgba.b = color[2];
    rgba.a = alpha;
    return rgba;
}

void BuildingEdgeFilter::mergePointClouds(
    const std::vector<ReconstructedPowerLine>& power_lines,
    pcl::PointCloud<pcl::PointXYZI>::Ptr& output_cloud) {
    
    output_cloud->clear();
    int total_merged_points = 0;
    
    ROS_INFO("开始合并 %zu 条有效电力线的点云", power_lines.size());
    
    for (size_t i = 0; i < power_lines.size(); ++i) {
        const auto& line = power_lines[i];
        
        if (!line.points || line.points->empty()) {
            ROS_WARN("电力线 %d 点云为空，跳过合并", line.line_id);
            continue;
        }
        
        int points_before = output_cloud->size();
        *output_cloud += *line.points;
        int points_after = output_cloud->size();
        int added_points = points_after - points_before;
        
        total_merged_points += added_points;
        
        ROS_INFO("电力线 %d: 合并了 %d 个点 (原始点数: %zu)", 
                line.line_id, added_points, line.points->size());
        
        // 检查点数是否匹配
        if (added_points != static_cast<int>(line.points->size())) {
            ROS_WARN("电力线 %d 点数不匹配！期望: %zu, 实际添加: %d", 
                    line.line_id, line.points->size(), added_points);
        }
    }
    
    output_cloud->width = output_cloud->size();
    output_cloud->height = 1;
    output_cloud->is_dense = true;
    
    ROS_INFO("点云合并完成！");
    ROS_INFO("  - 有效电力线数量: %zu", power_lines.size());
    ROS_INFO("  - 合并总点数: %d", total_merged_points);
    ROS_INFO("  - 输出点云总点数: %zu", output_cloud->size());
    
    // 验证点数一致性
    if (total_merged_points != static_cast<int>(output_cloud->size())) {
        
    ROS_INFO("合并计数: %d, 实际点数: %zu", 
                 total_merged_points, output_cloud->size());
    }
}

void BuildingEdgeFilter::validatePowerLinesIntegrity(
    const std::vector<ReconstructedPowerLine>& power_lines,
    const std::string& stage_name) {
    
    ROS_INFO("=== %s 电力线完整性验证 ===", stage_name.c_str());
    
    int total_points = 0;
    int valid_lines = 0;
    int empty_lines = 0;
    
    for (size_t i = 0; i < power_lines.size(); ++i) {
        const auto& line = power_lines[i];
        
        if (!line.points) {
            ROS_ERROR("电力线 %d: 点云指针为空！", line.line_id);
            empty_lines++;
            continue;
        }
        
        if (line.points->empty()) {
            ROS_WARN("电力线 %d: 点云为空", line.line_id);
            empty_lines++;
            continue;
        }
        
        valid_lines++;
        total_points += line.points->size();
        
        ROS_INFO("电力线 %d: 点数=%zu, 长度=%.2fm", 
                line.line_id, line.points->size(), line.total_length);
    }
    
    ROS_INFO("验证结果：");
    ROS_INFO("  - 总电力线数: %zu", power_lines.size());
    ROS_INFO("  - 有效电力线数: %d", valid_lines);
    ROS_INFO("  - 空电力线数: %d", empty_lines);
    ROS_INFO("  - 总点数: %d", total_points);
    ROS_INFO("=== 验证完成 ===");
}