#include "advanced_obstacle_analyzer.h"
#include <pcl/filters/extract_indices.h>
#include <pcl/search/kdtree.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/common/pca.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <algorithm>
#include <set>
#include <cmath>
#include <sstream>
#include <std_msgs/ColorRGBA.h>

AdvancedObstacleAnalyzer::AdvancedObstacleAnalyzer(ros::NodeHandle& nh, const std::string& frame_id) 
    : nh_(nh), frame_id_(frame_id), last_num_obstacles_(0), last_num_powerlines_(0) {
    
    loadParameters();
    
    // 初始化发布器
    obstacle_markers_pub_ = nh_.advertise<visualization_msgs::MarkerArray>(
    "advanced_obstacle/obstacle_markers", 1);
    powerline_info_pub_ = nh_.advertise<visualization_msgs::MarkerArray>(
        "advanced_obstacle/powerline_info", 1);
    merged_warning_pub_ = nh_.advertise<sensor_msgs::PointCloud2>(
        "advanced_obstacle/merged_warning_cloud", 1);
    end_obstacle_pub_ = nh_.advertise<sensor_msgs::PointCloud2>(
        "advanced_obstacle/end_obstacle_cloud", 1);
    warning_radius_pub_ = nh_.advertise<visualization_msgs::MarkerArray>(
        "advanced_obstacle/warning_radius", 1);
    
    ROS_INFO("[AdvancedObstacleAnalyzer] Initialized with parameters:");
    ROS_INFO("  Cluster: tolerance=%.2f, min_size=%d, max_size=%d", 
             cluster_tolerance_, cluster_min_size_, cluster_max_size_);
    ROS_INFO("  Warning levels: L1=%.1fm, L2=%.1fm", 
             warning_level1_radius_, warning_level2_radius_);
    ROS_INFO("  Powerline: clean_radius=%.2f, merge_distance=%.1f", 
             powerline_clean_radius_, powerline_merge_distance_);
    ROS_INFO("  Frame ID: %s", frame_id_.c_str());
}

void AdvancedObstacleAnalyzer::loadParameters() {
    // 聚类参数
    nh_.param("advanced_obstacle/cluster_tolerance", cluster_tolerance_, 0.3);
    nh_.param("advanced_obstacle/cluster_min_size", cluster_min_size_, 50);
    nh_.param("advanced_obstacle/cluster_max_size", cluster_max_size_, 50000);
    
    // 电力线清理参数
    nh_.param("advanced_obstacle/powerline_clean_radius", powerline_clean_radius_, 0.5);
    nh_.param("advanced_obstacle/powerline_merge_distance", powerline_merge_distance_, 2.0);
    
    // 分层预警参数
    nh_.param("advanced_obstacle/warning_level1_radius", warning_level1_radius_, 3.0);
    nh_.param("advanced_obstacle/warning_level2_radius", warning_level2_radius_, 8.0);
    
    // 端点过滤参数
    nh_.param("advanced_obstacle/end_filter_distance", end_filter_distance_, 5.0);
    nh_.param("advanced_obstacle/end_filter_ratio", end_filter_ratio_, 0.8);
    
    // 可视化参数
    nh_.param("advanced_obstacle/obstacle_box_alpha", obstacle_box_alpha_, 0.6);
    nh_.param("advanced_obstacle/show_powerline_distance", show_powerline_distance_, true);
    nh_.param("advanced_obstacle/show_obstacle_distance", show_obstacle_distance_, true);

    nh_.param("advanced_obstacle/min_warning_cloud_size", min_warning_cloud_size_, 10);
    nh_.param("advanced_obstacle/warning_corridor_width", warning_corridor_width_, 100.0);   // 100米宽度
}

void AdvancedObstacleAnalyzer::analyzeObstacles(
    const std::vector<ReconstructedPowerLine>& power_lines,
    const pcl::PointCloud<pcl::PointXYZI>::Ptr& env_cloud,
    std::vector<AdvancedObstacleBox>& obstacle_results,
    LayeredWarningCloud& warning_cloud) {
    
    if (power_lines.empty() || !env_cloud || env_cloud->empty()) {
        ROS_WARN("[AdvancedObstacleAnalyzer] Invalid input data");
        return;
    }
    
    // 1. 清理环境点云中的电力线点
    pcl::PointCloud<pcl::PointXYZI>::Ptr cleaned_cloud(new pcl::PointCloud<pcl::PointXYZI>);
    cleanEnvironmentCloud(power_lines, env_cloud, cleaned_cloud);
    
    // 2. 计算分层预警点云
    computeLayeredWarning(cleaned_cloud, power_lines, warning_cloud);
    
    // 3. 欧式聚类
    std::vector<pcl::PointIndices> cluster_indices;
    euclideanClustering(cleaned_cloud, cluster_indices);
    
    // 4. 处理每个障碍物聚类
    obstacle_results.clear();
    for (const auto& indices : cluster_indices) {
        if (indices.indices.empty()) continue;
        
        pcl::PointCloud<pcl::PointXYZI>::Ptr cluster_cloud(new pcl::PointCloud<pcl::PointXYZI>);
        pcl::copyPointCloud(*cleaned_cloud, indices, *cluster_cloud);
        
        // 5. 过滤端点障碍物
        if (isEndObstacle(cluster_cloud, power_lines)) {
            continue;
        }
        
        // 6. 计算高级OBB
        AdvancedObstacleBox obb;
        computeAdvancedOBB(cluster_cloud, power_lines, obb);
        
        obstacle_results.push_back(obb);
    }
    
    ROS_INFO("[AdvancedObstacleAnalyzer] Found %zu obstacles after filtering", 
             obstacle_results.size());

    // 在函数最后添加自动可视化发布
    publishAllVisualization(power_lines, obstacle_results, warning_cloud);
}

void AdvancedObstacleAnalyzer::cleanEnvironmentCloud(
    const std::vector<ReconstructedPowerLine>& power_lines,
    const pcl::PointCloud<pcl::PointXYZI>::Ptr& env_cloud,
    pcl::PointCloud<pcl::PointXYZI>::Ptr& cleaned_cloud) {
    
    cleaned_cloud->clear();
    cleaned_cloud->header = env_cloud->header;
    
    // 构建所有电力线点的KDTree
    pcl::PointCloud<pcl::PointXYZI>::Ptr all_powerline_points(new pcl::PointCloud<pcl::PointXYZI>);
    for (const auto& line : power_lines) {
        *all_powerline_points += *(line.points);
    }
    
    if (all_powerline_points->empty()) {
        *cleaned_cloud = *env_cloud;
        return;
    }
    
    pcl::KdTreeFLANN<pcl::PointXYZI> kdtree;
    kdtree.setInputCloud(all_powerline_points);
    
    // 过滤环境点云
    for (const auto& pt : env_cloud->points) {
        std::vector<int> pointIdxRadiusSearch;
        std::vector<float> pointRadiusSquaredDistance;
        
        int found = kdtree.radiusSearch(pt, powerline_clean_radius_, 
                                       pointIdxRadiusSearch, pointRadiusSquaredDistance);
        
        if (found == 0) {
            cleaned_cloud->push_back(pt);
        }
    }
    
    ROS_INFO("[AdvancedObstacleAnalyzer] Cleaned cloud: %zu -> %zu points", 
             env_cloud->size(), cleaned_cloud->size());
}

void AdvancedObstacleAnalyzer::euclideanClustering(
    const pcl::PointCloud<pcl::PointXYZI>::Ptr& input_cloud,
    std::vector<pcl::PointIndices>& cluster_indices) {
    
    pcl::search::KdTree<pcl::PointXYZI>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZI>);
    tree->setInputCloud(input_cloud);
    
    pcl::EuclideanClusterExtraction<pcl::PointXYZI> ec;
    ec.setClusterTolerance(cluster_tolerance_);
    ec.setMinClusterSize(cluster_min_size_);
    ec.setMaxClusterSize(cluster_max_size_);
    ec.setSearchMethod(tree);
    ec.setInputCloud(input_cloud);
    ec.extract(cluster_indices);
    
    ROS_INFO("[AdvancedObstacleAnalyzer] Found %zu clusters", cluster_indices.size());
}

void AdvancedObstacleAnalyzer::computeAdvancedOBB(
    const pcl::PointCloud<pcl::PointXYZI>::Ptr& cluster_cloud,
    const std::vector<ReconstructedPowerLine>& power_lines,
    AdvancedObstacleBox& obb) {
    
    // 使用PCA计算OBB
    pcl::PCA<pcl::PointXYZI> pca;
    pca.setInputCloud(cluster_cloud);
    Eigen::Vector4f mean = pca.getMean();
    Eigen::Matrix3f eigenvectors = pca.getEigenVectors();
    
    // 坐标变换到主轴系
    Eigen::Matrix4f tf = Eigen::Matrix4f::Identity();
    tf.block<3,3>(0,0) = eigenvectors.transpose();
    tf.block<3,1>(0,3) = -eigenvectors.transpose() * mean.head<3>();
    
    float minx=1e9, maxx=-1e9, miny=1e9, maxy=-1e9, minz=1e9, maxz=-1e9;
    for (const auto& pt : cluster_cloud->points) {
        Eigen::Vector4f ptv(pt.x, pt.y, pt.z, 1.0);
        Eigen::Vector4f local = tf * ptv;
        minx = std::min(minx, local[0]);
        maxx = std::max(maxx, local[0]);
        miny = std::min(miny, local[1]);
        maxy = std::max(maxy, local[1]);
        minz = std::min(minz, local[2]);
        maxz = std::max(maxz, local[2]);
    }
    
    // 计算OBB属性
    Eigen::Vector3f mean_local((minx+maxx)/2, (miny+maxy)/2, (minz+maxz)/2);
    Eigen::Vector3f size(maxx-minx, maxy-miny, maxz-minz);
    Eigen::Vector3f position = eigenvectors * mean_local + mean.head<3>();
    
    obb.position = position;
    obb.size = size;
    obb.rotation = eigenvectors;
    
    // 计算到电力线的最短距离
    obb.min_distance_to_powerline = computeMinDistanceToPowerlines(
        cluster_cloud, power_lines, obb.closest_powerline_id);
    
    // 确定预警级别
    if (obb.min_distance_to_powerline <= warning_level1_radius_) {
        obb.warning_level = 1;
    } else if (obb.min_distance_to_powerline <= warning_level2_radius_) {
        obb.warning_level = 2;
    } else {
        obb.warning_level = 3;
    }
    
    obb.is_end_obstacle = false;
}

bool AdvancedObstacleAnalyzer::isEndObstacle(
    const pcl::PointCloud<pcl::PointXYZI>::Ptr& cluster_cloud,
    const std::vector<ReconstructedPowerLine>& power_lines) {
    
    for (const auto& line : power_lines) {
        if (line.fitted_curve_points.size() < 2) continue;
        
        // 检查聚类是否靠近电力线端点
        Eigen::Vector3f start_point = line.fitted_curve_points.front();
        Eigen::Vector3f end_point = line.fitted_curve_points.back();
        
        int near_start = 0, near_end = 0;
        for (const auto& pt : cluster_cloud->points) {
            Eigen::Vector3f point(pt.x, pt.y, pt.z);
            
            float dist_to_start = (point - start_point).norm();
            float dist_to_end = (point - end_point).norm();
            
            if (dist_to_start < end_filter_distance_) near_start++;
            if (dist_to_end < end_filter_distance_) near_end++;
        }
        
        float ratio_start = static_cast<float>(near_start) / cluster_cloud->size();
        float ratio_end = static_cast<float>(near_end) / cluster_cloud->size();
        
        if (ratio_start > end_filter_ratio_ || ratio_end > end_filter_ratio_) {
            return true;
        }
    }
    
    return false;
}

void AdvancedObstacleAnalyzer::computeLayeredWarning(
    const pcl::PointCloud<pcl::PointXYZI>::Ptr& cleaned_cloud,
    const std::vector<ReconstructedPowerLine>& power_lines,
    LayeredWarningCloud& warning_cloud) {
    
    warning_cloud.merged_warning_cloud->clear();
    warning_cloud.end_obstacle_cloud->clear();
    
    // 检查输入点云大小
    if (cleaned_cloud->size() < min_warning_cloud_size_) {
        ROS_WARN("[AdvancedObstacleAnalyzer] Input cloud too small (%zu < %d), skipping warning analysis",
                 cleaned_cloud->size(), min_warning_cloud_size_);
        return;
    }
    
    int corridor_points = 0;
    int end_obstacle_points = 0;
    
    for (const auto& pt : cleaned_cloud->points) {
        Eigen::Vector3f point(pt.x, pt.y, pt.z);
        
        // 检查是否为端点附近的障碍物（电线杆等）
        if (isNearPowerlineEnd(point, power_lines)) {
            warning_cloud.end_obstacle_cloud->push_back(pt);
            end_obstacle_points++;
            continue;  // 跳过，不加入预警点云
        }
        
        // 检查点是否在电力线通道内
        float min_distance_to_axis;
        if (!isPointInPowerlineCorridor(point, power_lines, min_distance_to_axis)) {
            continue;  // 不在通道内，跳过
        }
        
        corridor_points++;
        
        // 创建带颜色的点
        pcl::PointXYZRGB colored_pt;
        colored_pt.x = pt.x;
        colored_pt.y = pt.y;
        colored_pt.z = pt.z;

        
        // 根据到电力线轴线的距离分层并设置颜色
        if (min_distance_to_axis <= warning_level1_radius_) {
            // 红色：危险
            colored_pt.r = 255; colored_pt.g = 0; colored_pt.b = 0;
        } else if (min_distance_to_axis <= warning_level2_radius_) {
            // 黄色：警告
            colored_pt.r = 255; colored_pt.g = 255; colored_pt.b = 0;
        } else {
            // 绿色：安全
            colored_pt.r = 0; colored_pt.g = 255; colored_pt.b = 0;
        }
        
        warning_cloud.merged_warning_cloud->push_back(colored_pt);
    }
    
    ROS_INFO("[AdvancedObstacleAnalyzer] Warning analysis: Corridor=%d, EndObstacles=%d, Total=%zu", 
             corridor_points, end_obstacle_points, cleaned_cloud->size());
}

float AdvancedObstacleAnalyzer::computeMinDistanceToPowerlines(
    const pcl::PointCloud<pcl::PointXYZI>::Ptr& cluster_cloud,
    const std::vector<ReconstructedPowerLine>& power_lines,
    int& closest_line_id) {
    
    float min_dist = std::numeric_limits<float>::max();
    closest_line_id = -1;
    
    for (size_t line_idx = 0; line_idx < power_lines.size(); ++line_idx) {
        const auto& line = power_lines[line_idx];
        
        for (const auto& pt : cluster_cloud->points) {
            Eigen::Vector3f point(pt.x, pt.y, pt.z);
            float dist = pointToSplineDistance(point, line.fitted_curve_points);
            
            if (dist < min_dist) {
                min_dist = dist;
                closest_line_id = line.line_id;
            }
        }
    }
    
    return min_dist;
}

float AdvancedObstacleAnalyzer::pointToSplineDistance(
    const Eigen::Vector3f& point,
    const std::vector<Eigen::Vector3f>& spline_points) {
    
    if (spline_points.empty()) return std::numeric_limits<float>::max();
    
    float min_dist = std::numeric_limits<float>::max();
    
    // 计算到样条曲线上所有点的最短距离
    for (const auto& spline_pt : spline_points) {
        float dist = (point - spline_pt).norm();
        min_dist = std::min(min_dist, dist);
    }
    
    return min_dist;
}

void AdvancedObstacleAnalyzer::publishObstacleMarkers(
    const std::vector<AdvancedObstacleBox>& obstacles,
    ros::Publisher& marker_pub,
    const std::string& frame_id) {
    
    visualization_msgs::MarkerArray marker_array;
    
    for (size_t i = 0; i < obstacles.size(); ++i) {
        const auto& obs = obstacles[i];
        
        // 障碍物盒子
        visualization_msgs::Marker box_marker;
        box_marker.header.frame_id = frame_id;
        box_marker.header.stamp = ros::Time::now();
        box_marker.ns = "advanced_obstacle_box";
        box_marker.id = i;
        box_marker.type = visualization_msgs::Marker::CUBE;
        box_marker.action = visualization_msgs::Marker::ADD;
        
        box_marker.pose.position.x = obs.position.x();
        box_marker.pose.position.y = obs.position.y();
        box_marker.pose.position.z = obs.position.z();
        
        Eigen::Quaternionf q(obs.rotation);
        box_marker.pose.orientation.x = q.x();
        box_marker.pose.orientation.y = q.y();
        box_marker.pose.orientation.z = q.z();
        box_marker.pose.orientation.w = q.w();
        
        box_marker.scale.x = obs.size.x();
        box_marker.scale.y = obs.size.y();
        box_marker.scale.z = obs.size.z();
        
        // 根据预警级别设置颜色
        Eigen::Vector3f color = getColorByWarningLevel(obs.warning_level);
        box_marker.color.r = color.x();
        box_marker.color.g = color.y();
        box_marker.color.b = color.z();
        box_marker.color.a = obstacle_box_alpha_;
        
        marker_array.markers.push_back(box_marker);
        
        // 距离文本
        if (show_obstacle_distance_) {
            visualization_msgs::Marker text_marker = box_marker;
            text_marker.type = visualization_msgs::Marker::TEXT_VIEW_FACING;
            text_marker.ns = "advanced_obstacle_text";
            text_marker.id = 2000 + i;
            text_marker.pose.position.z += 0.5 * obs.size.z() + 0.2;
            
            std::ostringstream oss;
            oss << getWarningLevelText(obs.warning_level) << "\n"
                << std::fixed << std::setprecision(2) << obs.min_distance_to_powerline << "m\n"
                << "Line:" << obs.closest_powerline_id;
            text_marker.text = oss.str();
            
            text_marker.scale.x = text_marker.scale.y = 0.1;
            text_marker.scale.z = 0.3;
            text_marker.color.r = text_marker.color.g = text_marker.color.b = 1.0;
            text_marker.color.a = 1.0;
            
            marker_array.markers.push_back(text_marker);
        }
    }
    
    // 清理多余的标记
    for (size_t j = obstacles.size(); j < last_num_obstacles_; ++j) {
        visualization_msgs::Marker del_marker;
        del_marker.header.frame_id = frame_id;
        del_marker.header.stamp = ros::Time::now();
        del_marker.ns = "advanced_obstacle_box";
        del_marker.id = j;
        del_marker.action = visualization_msgs::Marker::DELETE;
        marker_array.markers.push_back(del_marker);
        
        if (show_obstacle_distance_) {
            del_marker.ns = "advanced_obstacle_text";
            del_marker.id = 2000 + j;
            marker_array.markers.push_back(del_marker);
        }
    }
    
    marker_pub.publish(marker_array);
    last_num_obstacles_ = obstacles.size();
}

// 替换原来的publishWarningClouds函数
void AdvancedObstacleAnalyzer::publishWarningClouds(
    const LayeredWarningCloud& warning_cloud,
    ros::Publisher& merged_pub,
    ros::Publisher& end_obstacle_pub,
    ros::Publisher& unused_pub,  // 保持兼容性，但不使用
    const std::string& frame_id) {
    
    sensor_msgs::PointCloud2 cloud_msg;
    
    // 发布合并的预警点云
    if (!warning_cloud.merged_warning_cloud->empty()) {
        warning_cloud.merged_warning_cloud->header.frame_id = frame_id;
        pcl::toROSMsg(*warning_cloud.merged_warning_cloud, cloud_msg);
        merged_pub.publish(cloud_msg);
    }
    
    // 发布端点障碍物点云
    if (!warning_cloud.end_obstacle_cloud->empty()) {
        warning_cloud.end_obstacle_cloud->header.frame_id = frame_id;
        pcl::toROSMsg(*warning_cloud.end_obstacle_cloud, cloud_msg);
        end_obstacle_pub.publish(cloud_msg);
    }
}

void AdvancedObstacleAnalyzer::publishPowerlineInfo(
    const std::vector<ReconstructedPowerLine>& power_lines,
    ros::Publisher& marker_pub,
    const std::string& frame_id) {
    
    if (!show_powerline_distance_) return;
    
    visualization_msgs::MarkerArray marker_array;
    
    for (size_t i = 0; i < power_lines.size(); ++i) {
        const auto& line = power_lines[i];
        
        // 电力线样条曲线可视化
        visualization_msgs::Marker line_marker;
        line_marker.header.frame_id = frame_id;
        line_marker.header.stamp = ros::Time::now();
        line_marker.ns = "powerline_spline";
        line_marker.id = i;
        line_marker.type = visualization_msgs::Marker::LINE_STRIP;
        line_marker.action = visualization_msgs::Marker::ADD;
        
        line_marker.scale.x = 0.1;
        line_marker.color.r = 0.0;
        line_marker.color.g = 1.0;
        line_marker.color.b = 1.0;
        line_marker.color.a = 0.8;
        
        for (const auto& pt : line.fitted_curve_points) {
            geometry_msgs::Point gm_pt;
            gm_pt.x = pt.x();
            gm_pt.y = pt.y();
            gm_pt.z = pt.z();
            line_marker.points.push_back(gm_pt);
        }
        
        marker_array.markers.push_back(line_marker);
        
        // 电力线信息文本
        if (!line.fitted_curve_points.empty()) {
            visualization_msgs::Marker text_marker;
            text_marker.header = line_marker.header;
            text_marker.ns = "powerline_info";
            text_marker.id = 3000 + i;
            text_marker.type = visualization_msgs::Marker::TEXT_VIEW_FACING;
            text_marker.action = visualization_msgs::Marker::ADD;
            
            // 在电力线中点显示信息
            size_t mid_idx = line.fitted_curve_points.size() / 2;
            const auto& mid_pt = line.fitted_curve_points[mid_idx];
            text_marker.pose.position.x = mid_pt.x();
            text_marker.pose.position.y = mid_pt.y();
            text_marker.pose.position.z = mid_pt.z() + 1.0;
            
            std::ostringstream oss;
            oss << "Line " << line.line_id << "\n"
                << "Length: " << std::fixed << std::setprecision(1) << line.total_length << "m";
            text_marker.text = oss.str();
            
            text_marker.scale.z = 0.5;
            text_marker.color.r = text_marker.color.g = text_marker.color.b = 1.0;
            text_marker.color.a = 1.0;
            
            marker_array.markers.push_back(text_marker);
        }
    }
    
    // 清理多余的电力线标记
    for (size_t j = power_lines.size(); j < last_num_powerlines_; ++j) {
        visualization_msgs::Marker del_marker;
        del_marker.header.frame_id = frame_id;
        del_marker.header.stamp = ros::Time::now();
        del_marker.ns = "powerline_spline";
        del_marker.id = j;
        del_marker.action = visualization_msgs::Marker::DELETE;
        marker_array.markers.push_back(del_marker);
        
        del_marker.ns = "powerline_info";
        del_marker.id = 3000 + j;
        marker_array.markers.push_back(del_marker);
    }
    
    marker_pub.publish(marker_array);
    last_num_powerlines_ = power_lines.size();
}

Eigen::Vector3f AdvancedObstacleAnalyzer::getColorByWarningLevel(int level) {
    switch (level) {
        case 1: return Eigen::Vector3f(1.0, 0.0, 0.0);  // 红色：危险
        case 2: return Eigen::Vector3f(1.0, 1.0, 0.0);  // 黄色：警告
        case 3: return Eigen::Vector3f(0.0, 1.0, 0.0);  // 绿色：安全
        default: return Eigen::Vector3f(0.5, 0.5, 0.5); // 灰色：未知
    }
}

std::string AdvancedObstacleAnalyzer::getWarningLevelText(int level) {
    switch (level) {
        case 1: return "DANGER";
        case 2: return "WARNING";
        case 3: return "SAFE";
        default: return "UNKNOWN";
    }
}


void AdvancedObstacleAnalyzer::publishAllVisualization(
    const std::vector<ReconstructedPowerLine>& power_lines,
    const std::vector<AdvancedObstacleBox>& obstacles,
    const LayeredWarningCloud& warning_cloud) {
    
    // 发布障碍物标记
    publishObstacleMarkers(obstacles, obstacle_markers_pub_, frame_id_);
    
    // 发布电力线信息
    publishPowerlineInfo(power_lines, powerline_info_pub_, frame_id_);
    
    // 发布预警半径框架
    publishWarningRadius(power_lines, warning_radius_pub_, frame_id_);
    
    // 发布分层预警点云（现在只需要两个发布器）
    publishWarningClouds(warning_cloud, 
                        merged_warning_pub_, 
                        end_obstacle_pub_, 
                        merged_warning_pub_,  // 传入相同的pub保持兼容
                        frame_id_);
}


// bool AdvancedObstacleAnalyzer::isNearPowerlineEnd(
//     const Eigen::Vector3f& point,
//     const std::vector<ReconstructedPowerLine>& power_lines,
//     double end_threshold) {

//     for (const auto& line : power_lines) {
//         if (line.fitted_curve_points.size() < 2) continue;
        
//         // 检查到起点和终点的距离
//         Eigen::Vector3f start_point = line.fitted_curve_points.front();
//         Eigen::Vector3f end_point = line.fitted_curve_points.back();
        
//         float dist_to_start = (point - start_point).norm();
//         float dist_to_end = (point - end_point).norm();
        
//         if (dist_to_start < end_threshold || dist_to_end < end_threshold) {
//             return true;
//         }
//     }

//     return false;
// }
bool AdvancedObstacleAnalyzer::isNearPowerlineEnd(
    const Eigen::Vector3f& point,
    const std::vector<ReconstructedPowerLine>& power_lines,
    double end_threshold) {
    
    // 使用更小的端点阈值，默认5米
    double actual_threshold = 2.0;
    
    for (const auto& line : power_lines) {
        if (line.fitted_curve_points.size() < 2) continue;
        
        // 取电力线的前10%和后10%作为端点区域
        size_t front_range = std::max(1, static_cast<int>(line.fitted_curve_points.size() * 0.1));
        size_t back_range = std::max(1, static_cast<int>(line.fitted_curve_points.size() * 0.1));
        
        // 检查到起始段的距离
        float min_dist_to_start = std::numeric_limits<float>::max();
        for (size_t i = 0; i < front_range; ++i) {
            float dist = (point - line.fitted_curve_points[i]).norm();
            min_dist_to_start = std::min(min_dist_to_start, dist);
        }
        
        // 检查到结束段的距离
        float min_dist_to_end = std::numeric_limits<float>::max();
        for (size_t i = line.fitted_curve_points.size() - back_range; 
             i < line.fitted_curve_points.size(); ++i) {
            float dist = (point - line.fitted_curve_points[i]).norm();
            min_dist_to_end = std::min(min_dist_to_end, dist);
        }
        
        // 如果距离端点很近，并且高度差不大（避免空中的点被误判）
        if ((min_dist_to_start < actual_threshold || min_dist_to_end < actual_threshold)) {
            // 额外检查：确保点的高度与电力线端点高度相近（±5米内）
            Eigen::Vector3f start_point = line.fitted_curve_points.front();
            Eigen::Vector3f end_point = line.fitted_curve_points.back();
            
            float height_diff_start = std::abs(point.z() - start_point.z());
            float height_diff_end = std::abs(point.z() - end_point.z());
            
            if (min_dist_to_start < actual_threshold && height_diff_start < 5.0) {
                return true;
            }
            if (min_dist_to_end < actual_threshold && height_diff_end < 5.0) {
                return true;
            }
        }
    }
    
    return false;
}
   
void AdvancedObstacleAnalyzer::publishWarningRadius(
    const std::vector<ReconstructedPowerLine>& power_lines,
    ros::Publisher& marker_pub,
    const std::string& frame_id) {
    
    visualization_msgs::MarkerArray marker_array;
    
    for (size_t line_idx = 0; line_idx < power_lines.size(); ++line_idx) {
        const auto& line = power_lines[line_idx];
        
        if (line.fitted_curve_points.size() < 2) continue;
        
        // 计算电力线的中心点
        Eigen::Vector3f start_pt = line.fitted_curve_points.front();
        Eigen::Vector3f end_pt = line.fitted_curve_points.back();
        Eigen::Vector3f center = (start_pt + end_pt) / 2.0f;
        
        // 计算电力线的主方向和长度
        Eigen::Vector3f direction = (end_pt - start_pt).normalized();
        float line_length = (end_pt - start_pt).norm();
        
        // 计算旋转矩阵（将Z轴对齐到电力线方向）
        Eigen::Vector3f z_axis = direction;
        Eigen::Vector3f x_axis = z_axis.cross(Eigen::Vector3f::UnitZ()).normalized();
        if (x_axis.norm() < 0.1) {  // 如果电力线垂直，选择其他轴
            x_axis = z_axis.cross(Eigen::Vector3f::UnitX()).normalized();
        }
        Eigen::Vector3f y_axis = z_axis.cross(x_axis).normalized();
        
        Eigen::Matrix3f rotation;
        rotation.col(0) = x_axis;
        rotation.col(1) = y_axis;
        rotation.col(2) = z_axis;
        
        Eigen::Quaternionf q(rotation);
        
        // 第一层预警框架（红色，3米半径）
        visualization_msgs::Marker box1;
        box1.header.frame_id = frame_id;
        box1.header.stamp = ros::Time::now();
        box1.ns = "warning_box_level1";
        box1.id = line_idx;
        box1.type = visualization_msgs::Marker::CUBE;
        box1.action = visualization_msgs::Marker::ADD;
        
        box1.pose.position.x = center.x();
        box1.pose.position.y = center.y();
        box1.pose.position.z = center.z();
        box1.pose.orientation.x = q.x();
        box1.pose.orientation.y = q.y();
        box1.pose.orientation.z = q.z();
        box1.pose.orientation.w = q.w();
        
        // 尺寸：长度=电力线长度，宽高=预警半径*2
        box1.scale.x = warning_level1_radius_ * 2;  // 宽度
        box1.scale.y = warning_level1_radius_ * 2;  // 高度
        box1.scale.z = line_length;                 // 长度（沿电力线方向）
        
        box1.color.r = 1.0; box1.color.g = 0.0; box1.color.b = 0.0;
        box1.color.a = 0.15;  // 更低的透明度，只显示框架
        
        marker_array.markers.push_back(box1);
        
        // 第二层预警框架（黄色，8米半径）
        visualization_msgs::Marker box2 = box1;
        box2.ns = "warning_box_level2";
        box2.id = line_idx + 100;  // 不同的ID
        
        box2.scale.x = warning_level2_radius_ * 2;  // 宽度
        box2.scale.y = warning_level2_radius_ * 2;  // 高度
        box2.scale.z = line_length;                 // 长度
        
        box2.color.r = 1.0; box2.color.g = 1.0; box2.color.b = 0.0;
        box2.color.a = 0.1;   // 外层更透明
        
        marker_array.markers.push_back(box2);
        
        // 添加框架边线（使框架更明显）
        for (int level = 1; level <= 2; ++level) {
            float radius = (level == 1) ? warning_level1_radius_ : warning_level2_radius_;
            
            visualization_msgs::Marker frame_lines;
            frame_lines.header.frame_id = frame_id;
            frame_lines.header.stamp = ros::Time::now();
            frame_lines.ns = (level == 1) ? "warning_frame_level1" : "warning_frame_level2";
            frame_lines.id = line_idx;
            frame_lines.type = visualization_msgs::Marker::LINE_LIST;
            frame_lines.action = visualization_msgs::Marker::ADD;
            
            frame_lines.scale.x = 0.05;  // 线宽
            
            if (level == 1) {
                frame_lines.color.r = 1.0; frame_lines.color.g = 0.0; frame_lines.color.b = 0.0;
            } else {
                frame_lines.color.r = 1.0; frame_lines.color.g = 1.0; frame_lines.color.b = 0.0;
            }
            frame_lines.color.a = 0.8;
            
            // 计算框架的8个顶点
            std::vector<Eigen::Vector3f> corners;
            for (int i = 0; i < 8; ++i) {
                float x = (i & 1) ? radius : -radius;
                float y = (i & 2) ? radius : -radius;
                float z = (i & 4) ? line_length/2 : -line_length/2;
                
                Eigen::Vector3f local_corner(x, y, z);
                Eigen::Vector3f world_corner = rotation * local_corner + center;
                corners.push_back(world_corner);
            }
            
            // 添加12条边线
            int edges[12][2] = {
                {0,1}, {1,3}, {3,2}, {2,0},  // 底面
                {4,5}, {5,7}, {7,6}, {6,4},  // 顶面
                {0,4}, {1,5}, {2,6}, {3,7}   // 竖直边
            };
            
            for (int e = 0; e < 12; ++e) {
                geometry_msgs::Point p1, p2;
                p1.x = corners[edges[e][0]].x();
                p1.y = corners[edges[e][0]].y();
                p1.z = corners[edges[e][0]].z();
                p2.x = corners[edges[e][1]].x();
                p2.y = corners[edges[e][1]].y();
                p2.z = corners[edges[e][1]].z();
                
                frame_lines.points.push_back(p1);
                frame_lines.points.push_back(p2);
            }
            
            marker_array.markers.push_back(frame_lines);
        }
    }
    
    // 清理旧的标记
    for (size_t j = power_lines.size(); j < last_num_powerlines_; ++j) {
        // 清理旧的盒子和线框
        for (const auto& ns : {"warning_box_level1", "warning_box_level2", 
                              "warning_frame_level1", "warning_frame_level2"}) {
            visualization_msgs::Marker del_marker;
            del_marker.header.frame_id = frame_id;
            del_marker.header.stamp = ros::Time::now();
            del_marker.ns = ns;
            del_marker.id = j;
            del_marker.action = visualization_msgs::Marker::DELETE;
            marker_array.markers.push_back(del_marker);
        }
    }
    
    marker_pub.publish(marker_array);
}


bool AdvancedObstacleAnalyzer::isPointInPowerlineCorridor(
    const Eigen::Vector3f& point,
    const std::vector<ReconstructedPowerLine>& power_lines,
    float& min_distance_to_axis) {
    
    min_distance_to_axis = std::numeric_limits<float>::max();
    bool in_any_corridor = false;
    
    for (const auto& line : power_lines) {
        if (line.fitted_curve_points.size() < 2) continue;
        
        Eigen::Vector3f start_pt = line.fitted_curve_points.front();
        Eigen::Vector3f end_pt = line.fitted_curve_points.back();
        
        // 计算点到电力线轴线的距离
        float dist_to_axis = pointToLineSegmentDistance(point, start_pt, end_pt);
        min_distance_to_axis = std::min(min_distance_to_axis, dist_to_axis);
        
        // 检查点是否在电力线长度范围内（沿电力线方向的投影）
        Eigen::Vector3f line_direction = (end_pt - start_pt).normalized();
        Eigen::Vector3f to_point = point - start_pt;
        
        // 投影到电力线方向上
        float projection = to_point.dot(line_direction);
        float line_length = (end_pt - start_pt).norm();
        
        // 检查是否在电力线长度范围内（允许两端稍微延伸）
        bool within_length_range = (projection >= -line_length * 0.1) && 
                                  (projection <= line_length * 1.1);  // 两端延伸10%
        
        // 如果在电力线长度范围内，就认为在预警范围内（半径不限制）
        if (within_length_range) {
            in_any_corridor = true;
        }
    }
    
    return in_any_corridor;
}

float AdvancedObstacleAnalyzer::pointToLineSegmentDistance(
    const Eigen::Vector3f& point,
    const Eigen::Vector3f& line_start,
    const Eigen::Vector3f& line_end) {
    
    Eigen::Vector3f line_vec = line_end - line_start;
    Eigen::Vector3f point_vec = point - line_start;
    
    float line_length_sq = line_vec.squaredNorm();
    if (line_length_sq < 1e-6) {
        return (point - line_start).norm();  // 线段退化为点
    }
    
    float t = std::max(0.0f, std::min(1.0f, point_vec.dot(line_vec) / line_length_sq));
    Eigen::Vector3f projection = line_start + t * line_vec;
    
    return (point - projection).norm();
}