#include "roi_manager.h"
#include <chrono>
#include <algorithm>
#include <omp.h>
#include <pcl/common/common.h>
#include <pcl/filters/extract_indices.h>

ROIManager::ROIManager(ros::NodeHandle& nh) : nh_(nh) {
    // 读取参数
    loadParameters();
    
    // 初始化ROS发布器
    roi_boxes_pub_ = nh_.advertise<visualization_msgs::MarkerArray>(
        "roi_manager/roi_boxes", 1);
    roi_stats_pub_ = nh_.advertise<visualization_msgs::Marker>(
        "roi_manager/statistics", 1);
    
    ROS_INFO("ROIManager 初始化完成");
    ROS_INFO("参数配置:");
    ROS_INFO("  扩展系数: %.2f", expansion_factor_);
    ROS_INFO("  最小扩展: %.3f m", min_expansion_);
    ROS_INFO("  最大扩展: %.3f m", max_expansion_);
    ROS_INFO("  置信度阈值: %.2f", confidence_threshold_);
    ROS_INFO("  启用合并: %s", enable_merge_ ? "是" : "否");
    ROS_INFO("  最大ROI数: %d", max_roi_count_);
    ROS_INFO("  启用可视化: %s", enable_visualization_ ? "是" : "否");
}

ROIManager::~ROIManager() {
    clearHistory();
    ROS_INFO("ROIManager 析构完成");
}

void ROIManager::loadParameters() {
    // 扩展参数
    nh_.param("roi_manager/expansion_factor", expansion_factor_, 0.3f);
    nh_.param("roi_manager/min_expansion", min_expansion_, 0.5f);
    nh_.param("roi_manager/max_expansion", max_expansion_, 3.0f);
    
    // 过滤参数
    nh_.param("roi_manager/confidence_threshold", confidence_threshold_, 0.3f);
    nh_.param("roi_manager/min_points_per_roi", min_points_per_roi_, 100);
    nh_.param("roi_manager/min_roi_volume", min_roi_volume_, 0.1f);
    nh_.param("roi_manager/max_roi_volume", max_roi_volume_, 50.0f);
    
    // 合并参数
    nh_.param("roi_manager/enable_merge", enable_merge_, true);
    nh_.param("roi_manager/merge_overlap_threshold", merge_overlap_threshold_, 0.1f);
    nh_.param("roi_manager/max_roi_count", max_roi_count_, 10);
    
    // 空间边界参数
    nh_.param("roi_manager/x_min", x_min_, -50.0f);
    nh_.param("roi_manager/x_max", x_max_, 50.0f);
    nh_.param("roi_manager/y_min", y_min_, -50.0f);
    nh_.param("roi_manager/y_max", y_max_, 50.0f);
    nh_.param("roi_manager/z_min", z_min_, -5.0f);
    nh_.param("roi_manager/z_max", z_max_, 40.0f);
    
    // 可视化参数
    nh_.param("roi_manager/enable_visualization", enable_visualization_, true);
    nh_.param("roi_manager/visualization_duration", visualization_duration_, 5.0f);
    nh_.param("roi_manager/frame_id", frame_id_, std::string("base_link"));
    
    // 性能参数
    nh_.param("roi_manager/enable_parallel_processing", enable_parallel_processing_, true);
    nh_.param("roi_manager/max_processing_threads", max_processing_threads_, 4);
}

bool ROIManager::extractROI(const pcl::PointCloud<pcl::PointXYZI>::Ptr& full_cloud,
                           const std::vector<LineROIInfo>& line_rois,
                           std::vector<ROIResult>& roi_results,
                           pcl::PointCloud<pcl::PointXYZI>::Ptr& total_cropped_cloud) {
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    ROS_INFO("开始ROI提取，输入点云: %zu 点，电力线ROI: %zu 个", 
             full_cloud->points.size(), line_rois.size());
    
    // 验证输入数据
    if (!validateInputData(full_cloud, line_rois)) {
        ROS_ERROR("输入数据验证失败");
        return false;
    }
    
    // 清空输出容器
    roi_results.clear();
    total_cropped_cloud = pcl::PointCloud<pcl::PointXYZI>::Ptr(new pcl::PointCloud<pcl::PointXYZI>);
    
    try {
        // 1. 生成初始ROI长方体
        std::vector<ROIBox> initial_boxes = generateInitialROIBoxes(line_rois);
        ROS_DEBUG("生成了 %zu 个初始ROI长方体", initial_boxes.size());
        
        if (initial_boxes.empty()) {
            ROS_WARN("没有生成有效的ROI长方体");
            return true; // 不是错误，只是没有有效ROI
        }
        
        // 2. 过滤无效ROI
        std::vector<ROIBox> filtered_boxes = filterROIBoxes(initial_boxes);
        ROS_DEBUG("过滤后剩余 %zu 个ROI长方体", filtered_boxes.size());
        
        // 3. 合并重叠的ROI长方体
        std::vector<ROIBox> final_boxes;
        if (enable_merge_) {
            final_boxes = mergeOverlappingBoxes(filtered_boxes);
            ROS_DEBUG("合并后剩余 %zu 个ROI长方体", final_boxes.size());
        } else {
            final_boxes = filtered_boxes;
        }
        
        // 4. 限制ROI数量
        if (final_boxes.size() > static_cast<size_t>(max_roi_count_)) {
            // 按置信度排序，保留前max_roi_count_个
            std::sort(final_boxes.begin(), final_boxes.end(),
                     [](const ROIBox& a, const ROIBox& b) {
                         return a.confidence > b.confidence;
                     });
            final_boxes.resize(max_roi_count_);
            ROS_WARN("ROI数量超限，保留置信度最高的 %d 个", max_roi_count_);
        }
        
        // 5. 裁剪点云
        std::vector<pcl::PointCloud<pcl::PointXYZI>::Ptr> cropped_clouds;
        
        if (enable_parallel_processing_) {
            // 并行处理
            roi_results.resize(final_boxes.size());
            cropped_clouds.resize(final_boxes.size());
            
            #pragma omp parallel for num_threads(max_processing_threads_)
            for (size_t i = 0; i < final_boxes.size(); ++i) {
                ROIResult result;
                result.roi_box = final_boxes[i];
                result.cropped_cloud = cropPointCloud(full_cloud, final_boxes[i]);
                result.point_count = result.cropped_cloud->points.size();
                result.calculateDensity();
                
                roi_results[i] = result;
                cropped_clouds[i] = result.cropped_cloud;
            }
        } else {
            // 串行处理
            for (const auto& box : final_boxes) {
                ROIResult result;
                result.roi_box = box;
                result.cropped_cloud = cropPointCloud(full_cloud, box);
                result.point_count = result.cropped_cloud->points.size();
                result.calculateDensity();
                
                roi_results.push_back(result);
                cropped_clouds.push_back(result.cropped_cloud);
            }
        }
        
        // 6. 合并所有裁剪的点云为总点云
        total_cropped_cloud = combinePointClouds(cropped_clouds);
        
        auto end_time = std::chrono::high_resolution_clock::now();
        float processing_time = std::chrono::duration<float, std::milli>(end_time - start_time).count();
        
        // 7. 计算统计信息
        calculateStatistics(full_cloud, roi_results, processing_time);
        
        ROS_INFO("ROI提取完成");
        ROS_INFO("  处理时间: %.2f ms", processing_time);
        ROS_INFO("  最终ROI数: %zu", roi_results.size());
        ROS_INFO("  总输出点: %zu (压缩比: %.1f%%)", 
                 total_cropped_cloud->points.size(),
                 last_statistics_.compression_ratio * 100.0f);
        
        // 8. 可视化
        if (enable_visualization_) {
            visualizeROIBoxes(final_boxes);
        }
        
        return true;
        
    } catch (const std::exception& e) {
        ROS_ERROR("ROI提取过程中发生异常: %s", e.what());
        return false;
    }
}

pcl::PointCloud<pcl::PointXYZI>::Ptr ROIManager::extractROIForLine(
    const pcl::PointCloud<pcl::PointXYZI>::Ptr& full_cloud,
    const LineROIInfo& line_roi) {
    
    if (!line_roi.is_active || line_roi.high_prob_regions.empty()) {
        return pcl::PointCloud<pcl::PointXYZI>::Ptr(new pcl::PointCloud<pcl::PointXYZI>);
    }
    
    if (line_roi.confidence < confidence_threshold_) {
        return pcl::PointCloud<pcl::PointXYZI>::Ptr(new pcl::PointCloud<pcl::PointXYZI>);
    }
    
    // 创建单个ROI长方体
    ROIBox roi_box = createROIBoxFromRegions(line_roi);
    
    if (!validateROIBox(roi_box)) {
        return pcl::PointCloud<pcl::PointXYZI>::Ptr(new pcl::PointCloud<pcl::PointXYZI>);
    }
    
    // 裁剪点云
    return cropPointCloud(full_cloud, roi_box);
}

void ROIManager::clearHistory() {
    last_statistics_ = ROIStatistics();
}

// ==================== ROI生成函数 ====================

std::vector<ROIBox> ROIManager::generateInitialROIBoxes(const std::vector<LineROIInfo>& line_rois) {
    std::vector<ROIBox> roi_boxes;
    
    for (const auto& line_roi : line_rois) {
        // 过滤不活跃或置信度低的电力线
        if (!line_roi.is_active || line_roi.confidence < confidence_threshold_) {
            ROS_DEBUG("跳过电力线 %d (活跃: %s, 置信度: %.3f)", 
                     line_roi.line_id, line_roi.is_active ? "是" : "否", line_roi.confidence);
            continue;
        }
        
        // 过滤没有高概率区域的电力线
        if (line_roi.high_prob_regions.empty()) {
            ROS_DEBUG("跳过电力线 %d (无高概率区域)", line_roi.line_id);
            continue;
        }
        
        ROIBox roi_box = createROIBoxFromRegions(line_roi);
        
        if (validateROIBox(roi_box)) {
            roi_boxes.push_back(roi_box);
            ROS_DEBUG("为电力线 %d 创建ROI: 中心(%.2f,%.2f,%.2f), 尺寸(%.2f,%.2f,%.2f)", 
                     roi_box.primary_line_id,
                     roi_box.getCenter().x(), roi_box.getCenter().y(), roi_box.getCenter().z(),
                     roi_box.getSize().x(), roi_box.getSize().y(), roi_box.getSize().z());
        }
    }
    
    return roi_boxes;
}

ROIBox ROIManager::createROIBoxFromRegions(const LineROIInfo& line_roi) {
    ROIBox roi_box;
    roi_box.primary_line_id = line_roi.line_id;
    roi_box.confidence = line_roi.confidence;
    roi_box.merged_line_ids.push_back(line_roi.line_id);
    
    if (line_roi.high_prob_regions.empty()) {
        return roi_box; // 返回无效的ROI
    }
    
    // 计算包围盒
    Eigen::Vector3f min_point = line_roi.high_prob_regions[0];
    Eigen::Vector3f max_point = line_roi.high_prob_regions[0];
    
    for (const auto& point : line_roi.high_prob_regions) {
        min_point = min_point.cwiseMin(point);
        max_point = max_point.cwiseMax(point);
    }
    
    roi_box.min_point = min_point;
    roi_box.max_point = max_point;
    
    // 扩展长方体
    expandROIBox(roi_box);
    
    return roi_box;
}

void ROIManager::expandROIBox(ROIBox& roi_box) {
    Eigen::Vector3f size = roi_box.max_point - roi_box.min_point;
    Eigen::Vector3f center = roi_box.getCenter();
    
    // 计算自适应扩展
    float adaptive_factor = calculateExpansionFactor({roi_box.min_point, roi_box.max_point});
    
    // 计算扩展量
    Eigen::Vector3f expansion = size * adaptive_factor;
    
    // 限制扩展量在合理范围内
    for (int i = 0; i < 3; ++i) {
        expansion[i] = std::max(expansion[i], min_expansion_);
        expansion[i] = std::min(expansion[i], max_expansion_);
    }
    
    // 应用扩展
    roi_box.min_point = center - (size + expansion) * 0.5f;
    roi_box.max_point = center + (size + expansion) * 0.5f;
    
    // 限制在边界内
    roi_box.min_point.x() = std::max(roi_box.min_point.x(), x_min_);
    roi_box.min_point.y() = std::max(roi_box.min_point.y(), y_min_);
    roi_box.min_point.z() = std::max(roi_box.min_point.z(), z_min_);
    
    roi_box.max_point.x() = std::min(roi_box.max_point.x(), x_max_);
    roi_box.max_point.y() = std::min(roi_box.max_point.y(), y_max_);
    roi_box.max_point.z() = std::min(roi_box.max_point.z(), z_max_);
}

float ROIManager::calculateExpansionFactor(const std::vector<Eigen::Vector3f>& regions) {
    if (regions.size() < 2) return expansion_factor_;
    
    // 计算区域体积
    Eigen::Vector3f size = regions[1] - regions[0];
    float volume = size.x() * size.y() * size.z();
    
    // 基础扩展系数 + 根据体积调整
    float volume_factor = std::max(0.1f, 2.0f / (1.0f + volume)); // 体积越大，额外扩展越少
    
    return expansion_factor_ + volume_factor * 0.2f; // 最多额外20%扩展
}

// ==================== 重叠检测与合并函数 ====================

std::vector<ROIBox> ROIManager::mergeOverlappingBoxes(std::vector<ROIBox>& initial_boxes) {
    if (initial_boxes.size() <= 1) {
        return initial_boxes;
    }
    
    bool has_merge = true;
    
    while (has_merge) {
        has_merge = false;
        
        for (size_t i = 0; i < initial_boxes.size(); ++i) {
            for (size_t j = i + 1; j < initial_boxes.size(); ++j) {
                if (isOverlapping(initial_boxes[i], initial_boxes[j])) {
                    // 合并j到i，删除j
                    ROS_DEBUG("合并ROI %d 和 %d", 
                             initial_boxes[i].primary_line_id, initial_boxes[j].primary_line_id);
                    
                    initial_boxes[i] = mergeBoxes(initial_boxes[i], initial_boxes[j]);
                    initial_boxes.erase(initial_boxes.begin() + j);
                    has_merge = true;
                    break;
                }
            }
            if (has_merge) break;
        }
    }
    
    return initial_boxes;
}

bool ROIManager::isOverlapping(const ROIBox& box1, const ROIBox& box2) {
    // 计算重叠体积比例
    float overlap_ratio = calculateOverlapRatio(box1, box2);
    return overlap_ratio > merge_overlap_threshold_;
}

ROIBox ROIManager::mergeBoxes(const ROIBox& box1, const ROIBox& box2) {
    ROIBox merged_box;
    
    // 合并边界
    merged_box.min_point = box1.min_point.cwiseMin(box2.min_point);
    merged_box.max_point = box1.max_point.cwiseMax(box2.max_point);
    
    // 选择置信度更高的作为主ID
    if (box1.confidence >= box2.confidence) {
        merged_box.primary_line_id = box1.primary_line_id;
        merged_box.confidence = box1.confidence;
    } else {
        merged_box.primary_line_id = box2.primary_line_id;
        merged_box.confidence = box2.confidence;
    }
    
    // 合并电力线ID列表
    merged_box.merged_line_ids = box1.merged_line_ids;
    merged_box.merged_line_ids.insert(merged_box.merged_line_ids.end(),
                                     box2.merged_line_ids.begin(), box2.merged_line_ids.end());
    
    // 去重
    std::sort(merged_box.merged_line_ids.begin(), merged_box.merged_line_ids.end());
    merged_box.merged_line_ids.erase(
        std::unique(merged_box.merged_line_ids.begin(), merged_box.merged_line_ids.end()),
        merged_box.merged_line_ids.end());
    
    merged_box.is_merged = true;
    merged_box.creation_time = std::min(box1.creation_time, box2.creation_time);
    
    return merged_box;
}

float ROIManager::calculateOverlapRatio(const ROIBox& box1, const ROIBox& box2) {
    // 计算相交长方体
    Eigen::Vector3f intersect_min = box1.min_point.cwiseMax(box2.min_point);
    Eigen::Vector3f intersect_max = box1.max_point.cwiseMin(box2.max_point);
    
    // 检查是否有重叠
    if ((intersect_max.array() <= intersect_min.array()).any()) {
        return 0.0f; // 没有重叠
    }
    
    // 计算重叠体积
    Eigen::Vector3f intersect_size = intersect_max - intersect_min;
    float intersect_volume = intersect_size.x() * intersect_size.y() * intersect_size.z();
    
    // 计算两个长方体的体积
    float volume1 = box1.getVolume();
    float volume2 = box2.getVolume();
    
    // 返回重叠体积与较小长方体体积的比例
    float min_volume = std::min(volume1, volume2);
    return (min_volume > 0.001f) ? intersect_volume / min_volume : 0.0f;
}

// ==================== 点云裁剪函数 ====================

pcl::PointCloud<pcl::PointXYZI>::Ptr ROIManager::cropPointCloud(
    const pcl::PointCloud<pcl::PointXYZI>::Ptr& full_cloud,
    const ROIBox& roi_box) {
    
    pcl::PointCloud<pcl::PointXYZI>::Ptr cropped_cloud(new pcl::PointCloud<pcl::PointXYZI>);
    
    try {
        // 使用PCL的CropBox滤波器
        pcl::CropBox<pcl::PointXYZI> crop_filter;
        crop_filter.setInputCloud(full_cloud);
        
        // 设置裁剪范围
        Eigen::Vector4f min_point(roi_box.min_point.x(), roi_box.min_point.y(), 
                                 roi_box.min_point.z(), 1.0);
        Eigen::Vector4f max_point(roi_box.max_point.x(), roi_box.max_point.y(), 
                                 roi_box.max_point.z(), 1.0);
        
        crop_filter.setMin(min_point);
        crop_filter.setMax(max_point);
        crop_filter.filter(*cropped_cloud);
        
        ROS_DEBUG("ROI %d 裁剪点云: %zu -> %zu 点", 
                 roi_box.primary_line_id, full_cloud->points.size(), cropped_cloud->points.size());
        
    } catch (const std::exception& e) {
        ROS_ERROR("裁剪点云时发生异常: %s", e.what());
        // 返回空点云
        cropped_cloud = pcl::PointCloud<pcl::PointXYZI>::Ptr(new pcl::PointCloud<pcl::PointXYZI>);
    }
    
    return cropped_cloud;
}

pcl::PointCloud<pcl::PointXYZI>::Ptr ROIManager::combinePointClouds(
    const std::vector<pcl::PointCloud<pcl::PointXYZI>::Ptr>& clouds) {
    
    pcl::PointCloud<pcl::PointXYZI>::Ptr combined_cloud(new pcl::PointCloud<pcl::PointXYZI>);
    
    for (const auto& cloud : clouds) {
        if (cloud && !cloud->points.empty()) {
            *combined_cloud += *cloud;
        }
    }
    
    combined_cloud->width = combined_cloud->points.size();
    combined_cloud->height = 1;
    combined_cloud->is_dense = true;
    
    return combined_cloud;
}

// ==================== 验证与过滤函数 ====================

std::vector<ROIBox> ROIManager::filterROIBoxes(const std::vector<ROIBox>& roi_boxes) {
    std::vector<ROIBox> filtered_boxes;
    
    for (const auto& box : roi_boxes) {
        if (validateROIBox(box)) {
            filtered_boxes.push_back(box);
        } else {
            ROS_DEBUG("过滤掉无效ROI: 电力线ID %d", box.primary_line_id);
        }
    }
    
    return filtered_boxes;
}

bool ROIManager::validateROIBox(const ROIBox& roi_box) {
    // 检查基本有效性
    if (roi_box.primary_line_id < 0) return false;
    
    // 检查尺寸
    Eigen::Vector3f size = roi_box.getSize();
    if ((size.array() <= 0.0f).any()) return false;
    
    // 检查体积
    float volume = roi_box.getVolume();
    if (volume < min_roi_volume_ || volume > max_roi_volume_) return false;
    
    // 检查是否在边界内
    if (!isInBounds(roi_box.min_point) || !isInBounds(roi_box.max_point)) return false;
    
    // 检查置信度
    if (roi_box.confidence < confidence_threshold_) return false;
    
    return true;
}

bool ROIManager::isInBounds(const Eigen::Vector3f& point) {
    return (point.x() >= x_min_ && point.x() <= x_max_ &&
            point.y() >= y_min_ && point.y() <= y_max_ &&
            point.z() >= z_min_ && point.z() <= z_max_);
}

bool ROIManager::validateInputData(const pcl::PointCloud<pcl::PointXYZI>::Ptr& full_cloud,
                                  const std::vector<LineROIInfo>& line_rois) {
    if (!full_cloud || full_cloud->points.empty()) {
        ROS_ERROR("输入点云为空");
        return false;
    }
    
    if (line_rois.empty()) {
        ROS_WARN("电力线ROI列表为空");
        return false;
    }
    
    // 检查活跃的ROI数量
    int active_count = 0;
    for (const auto& roi : line_rois) {
        if (roi.is_active && roi.confidence >= confidence_threshold_) {
            active_count++;
        }
    }
    
    if (active_count == 0) {
        ROS_WARN("没有活跃且置信度足够的ROI");
        return false;
    }
    
    return true;
}

// ==================== 统计计算函数 ====================

void ROIManager::calculateStatistics(const pcl::PointCloud<pcl::PointXYZI>::Ptr& input_cloud,
                                     const std::vector<ROIResult>& roi_results,
                                     float processing_time_ms) {
    last_statistics_ = ROIStatistics();
    
    last_statistics_.total_roi_count = roi_results.size();
    last_statistics_.total_input_points = input_cloud->points.size();
    last_statistics_.processing_time_ms = processing_time_ms;
    
    int merged_count = 0;
    int total_output_points = 0;
    
    for (const auto& result : roi_results) {
        if (result.roi_box.is_merged) {
            merged_count++;
        }
        total_output_points += result.point_count;
    }
    
    last_statistics_.merged_roi_count = merged_count;
    last_statistics_.total_output_points = total_output_points;
    
    if (last_statistics_.total_input_points > 0) {
        last_statistics_.compression_ratio = 
            static_cast<float>(total_output_points) / last_statistics_.total_input_points;
    }
}

// ==================== 参数设置接口 ====================

void ROIManager::setExpansionParameters(float expansion_factor, float min_expansion) {
    expansion_factor_ = expansion_factor;
    min_expansion_ = min_expansion;
    ROS_INFO("更新扩展参数: 系数=%.2f, 最小扩展=%.3f", expansion_factor_, min_expansion_);
}

// ==================== 可视化函数 ====================

void ROIManager::visualizeROIBoxes(const std::vector<ROIBox>& roi_boxes) {
    if (!enable_visualization_) return;
    
    publishROIBoxesVisualization(roi_boxes);
    publishROIStatistics();
}

void ROIManager::publishROIBoxesVisualization(const std::vector<ROIBox>& roi_boxes) {
    visualization_msgs::MarkerArray marker_array;
    
    // 清除之前的标记
    visualization_msgs::Marker delete_marker;
    delete_marker.header.frame_id = frame_id_;
    delete_marker.header.stamp = ros::Time::now();
    delete_marker.ns = "roi_boxes";
    delete_marker.action = visualization_msgs::Marker::DELETEALL;
    marker_array.markers.push_back(delete_marker);
    
    // 创建新的ROI长方体标记
    for (size_t i = 0; i < roi_boxes.size(); ++i) {
        visualization_msgs::Marker marker = createBoxMarker(roi_boxes[i], i);
        marker_array.markers.push_back(marker);
    }
    
    roi_boxes_pub_.publish(marker_array);
    
    ROS_DEBUG("发布了 %zu 个ROI长方体可视化标记", roi_boxes.size());
}

void ROIManager::publishROIStatistics() {
    visualization_msgs::Marker text_marker;
    text_marker.header.frame_id = frame_id_;
    text_marker.header.stamp = ros::Time::now();
    text_marker.ns = "roi_statistics";
    text_marker.id = 0;
    text_marker.type = visualization_msgs::Marker::TEXT_VIEW_FACING;
    text_marker.action = visualization_msgs::Marker::ADD;
    
    // 位置设置
    text_marker.pose.position.x = 0.0;
    text_marker.pose.position.y = 0.0;
    text_marker.pose.position.z = z_max_ + 5.0f;
    text_marker.pose.orientation.w = 1.0;
    
    // 文本属性
    text_marker.scale.z = 1.0;
    text_marker.color.r = 1.0;
    text_marker.color.g = 1.0;
    text_marker.color.b = 1.0;
    text_marker.color.a = 1.0;
    text_marker.lifetime = ros::Duration(visualization_duration_);
    
    // 构建文本内容
    std::ostringstream ss;
    ss << "ROI管理器统计信息\n";
    ss << "ROI数量: " << last_statistics_.total_roi_count << "\n";
    ss << "合并ROI: " << last_statistics_.merged_roi_count << "\n";
    ss << "输入点: " << last_statistics_.total_input_points << "\n";
    ss << "输出点: " << last_statistics_.total_output_points << "\n";
    ss << "压缩比: " << std::fixed << std::setprecision(1) 
       << last_statistics_.compression_ratio * 100.0f << "%\n";
    ss << "处理时间: " << std::fixed << std::setprecision(2) 
       << last_statistics_.processing_time_ms << " ms";
    
    text_marker.text = ss.str();
    
    roi_stats_pub_.publish(text_marker);
}

visualization_msgs::Marker ROIManager::createBoxMarker(const ROIBox& roi_box, int marker_id) {
    visualization_msgs::Marker marker;
    marker.header.frame_id = frame_id_;
    marker.header.stamp = ros::Time::now();
    marker.ns = "roi_boxes";
    marker.id = marker_id;
    marker.type = visualization_msgs::Marker::CUBE;
    marker.action = visualization_msgs::Marker::ADD;
    
    // 设置位置（长方体中心）
    Eigen::Vector3f center = roi_box.getCenter();
    marker.pose.position.x = center.x();
    marker.pose.position.y = center.y();
    marker.pose.position.z = center.z();
    marker.pose.orientation.w = 1.0;
    
    // 设置尺寸
    Eigen::Vector3f size = roi_box.getSize();
    marker.scale.x = size.x();
    marker.scale.y = size.y();
    marker.scale.z = size.z();
    
    // 设置颜色
    marker.color = getROIBoxColor(roi_box);
    
    // 生命周期
    marker.lifetime = ros::Duration(visualization_duration_);
    
    return marker;
}

std_msgs::ColorRGBA ROIManager::getROIBoxColor(const ROIBox& roi_box) {
    std_msgs::ColorRGBA color;
    
    if (roi_box.is_merged) {
        // 合并的ROI：橙色
        color.r = 1.0f;
        color.g = 0.5f;
        color.b = 0.0f;
        color.a = 0.3f;
    } else {
        // 单独的ROI：根据置信度着色
        if (roi_box.confidence > 0.7f) {
            // 高置信度：绿色
            color.r = 0.0f;
            color.g = 1.0f;
            color.b = 0.0f;
        } else if (roi_box.confidence > 0.5f) {
            // 中置信度：黄色
            color.r = 1.0f;
            color.g = 1.0f;
            color.b = 0.0f;
        } else {
            // 低置信度：红色
            color.r = 1.0f;
            color.g = 0.0f;
            color.b = 0.0f;
        }
        color.a = 0.2f + roi_box.confidence * 0.3f; // 透明度基于置信度
    }
    
    return color;
}