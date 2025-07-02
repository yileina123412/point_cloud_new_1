// ================= ROS集成版本的预处理器实现 =================
// src/preprocessing/multilevel_preprocessor.cpp
#include "powerline_extraction/preprocessing/multilevel_preprocessor.h"
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/crop_box.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/segmentation/progressive_morphological_filter.h>
#include <pcl/common/common.h>
#include <iostream>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <iomanip>

namespace powerline_extraction {

MultiLevelPreprocessor::MultiLevelPreprocessor(ros::NodeHandle& nh) 
    : nh_(nh), private_nh_("~") {
    ROS_INFO("[PreProcessor] Initializing MultiLevel Preprocessor...");
    // 从ROS参数服务器加载配置
    loadROSParameters();
    ROS_INFO("[PreProcessor] MultiLevel Preprocessor initialized successfully");
}

MultiLevelPreprocessor::MultiLevelPreprocessor(ros::NodeHandle& nh, const PreprocessConfig& config) 
    : nh_(nh), private_nh_("~"), config_(config) {
    ROS_INFO("[PreProcessor] Initializing MultiLevel Preprocessor with custom config...");
    // 仍然尝试从ROS参数覆盖配置（可选）
    loadROSParameters();
    ROS_INFO("[PreProcessor] MultiLevel Preprocessor initialized with custom config");
}

void MultiLevelPreprocessor::loadROSParameters() {
    ROS_INFO("[PreProcessor] Loading parameters from ROS parameter server...");
    
    // 加载各模块参数
    loadDownsamplingParams();
    loadCroppingParams();
    loadCSFParams();
    loadIntensityParams();
    loadElevationParams();
    loadGeneralParams();
    
    // 输出所有参数信息
    printROSParameters();
    
    ROS_INFO("[PreProcessor] All parameters loaded successfully");
}

void MultiLevelPreprocessor::loadDownsamplingParams() {
    // 降采样参数
    private_nh_.param("preprocessing/downsampling/enable", config_.downsampling.enable, true);
    private_nh_.param("preprocessing/downsampling/leaf_size", config_.downsampling.leaf_size, 0.1);
    
    ROS_INFO("[PreProcessor] Downsampling params - Enable: %s, Leaf size: %.3f", 
             config_.downsampling.enable ? "true" : "false", 
             config_.downsampling.leaf_size);
}

void MultiLevelPreprocessor::loadCroppingParams() {
    // 点云裁剪参数
    private_nh_.param("preprocessing/cropping/enable", config_.cropping.enable, true);
    private_nh_.param("preprocessing/cropping/cube_size", config_.cropping.cube_size, 70.0);
    private_nh_.param("preprocessing/cropping/center_x", config_.cropping.center_x, 0.0);
    private_nh_.param("preprocessing/cropping/center_y", config_.cropping.center_y, 0.0);
    private_nh_.param("preprocessing/cropping/center_z", config_.cropping.center_z, 0.0);
    
    ROS_INFO("[PreProcessor] Cropping params - Enable: %s, Cube size: %.1f", 
             config_.cropping.enable ? "true" : "false", 
             config_.cropping.cube_size);
    ROS_INFO("[PreProcessor] Cropping center: (%.1f, %.1f, %.1f)", 
             config_.cropping.center_x, config_.cropping.center_y, config_.cropping.center_z);
}

void MultiLevelPreprocessor::loadCSFParams() {
    // CSF地面滤波参数
    private_nh_.param("preprocessing/csf/enable", config_.csf.enable, false);
    private_nh_.param("preprocessing/csf/classification_threshold", config_.csf.classification_threshold, 0.4);
    private_nh_.param("preprocessing/csf/max_iterations", config_.csf.max_iterations, 500);
    private_nh_.param("preprocessing/csf/cloth_resolution", config_.csf.cloth_resolution, 0.1);
    private_nh_.param("preprocessing/csf/rigidness", config_.csf.rigidness, 3);
    private_nh_.param("preprocessing/csf/time_step", config_.csf.time_step, 1.65);
    
    ROS_INFO("[PreProcessor] CSF params - Enable: %s, Threshold: %.2f, Iterations: %d", 
             config_.csf.enable ? "true" : "false", 
             config_.csf.classification_threshold, 
             config_.csf.max_iterations);
}

void MultiLevelPreprocessor::loadIntensityParams() {
    // 强度滤波参数
    private_nh_.param("preprocessing/intensity/enable", config_.intensity.enable, false);
    private_nh_.param("preprocessing/intensity/iqr_factor", config_.intensity.iqr_factor, 1.5);
    private_nh_.param("preprocessing/intensity/min_intensity", config_.intensity.min_intensity, 8200.0);
    private_nh_.param("preprocessing/intensity/max_intensity", config_.intensity.max_intensity, 9000.0);
    private_nh_.param("preprocessing/intensity/use_statistical_filtering", config_.intensity.use_statistical_filtering, true);
    
    ROS_INFO("[PreProcessor] Intensity params - Enable: %s, Statistical: %s", 
             config_.intensity.enable ? "true" : "false",
             config_.intensity.use_statistical_filtering ? "true" : "false");
    if (config_.intensity.use_statistical_filtering) {
        ROS_INFO("[PreProcessor] IQR factor: %.2f", config_.intensity.iqr_factor);
    } else {
        ROS_INFO("[PreProcessor] Intensity range: [%.0f, %.0f]", 
                 config_.intensity.min_intensity, config_.intensity.max_intensity);
    }
}

void MultiLevelPreprocessor::loadElevationParams() {
    // 高程滤波参数
    private_nh_.param("preprocessing/elevation/enable", config_.elevation.enable, false);
    private_nh_.param("preprocessing/elevation/min_elevation_threshold", config_.elevation.min_elevation_threshold, 10.0);
    private_nh_.param("preprocessing/elevation/max_elevation_threshold", config_.elevation.max_elevation_threshold, 80.0);
    private_nh_.param("preprocessing/elevation/relative_to_ground", config_.elevation.relative_to_ground, false);
    
    ROS_INFO("[PreProcessor] Elevation params - Enable: %s, Range: [%.1f, %.1f]", 
             config_.elevation.enable ? "true" : "false",
             config_.elevation.min_elevation_threshold, 
             config_.elevation.max_elevation_threshold);
}

void MultiLevelPreprocessor::loadGeneralParams() {
    // 通用参数
    private_nh_.param("preprocessing/build_kdtree", config_.build_kdtree, true);
    private_nh_.param("preprocessing/verbose", config_.verbose, false);
    
    // ROS话题参数
    private_nh_.param("lidar_topic", config_.lidar_topic, std::string("/livox/lidar"));
    private_nh_.param("output_topic", config_.output_topic, std::string("/powerline/preprocessed_cloud"));
    private_nh_.param("target_frame", config_.frame_id, std::string("map"));
    
    ROS_INFO("[PreProcessor] General params - Build KDTree: %s, Verbose: %s", 
             config_.build_kdtree ? "true" : "false",
             config_.verbose ? "true" : "false");
    ROS_INFO("[PreProcessor] ROS Topics - Input: %s, Output: %s, Frame: %s", 
             config_.lidar_topic.c_str(), 
             config_.output_topic.c_str(), 
             config_.frame_id.c_str());
}

void MultiLevelPreprocessor::printROSParameters() {
    ROS_INFO("================== PreProcessor Configuration ==================");
    ROS_INFO("Downsampling: %s (leaf_size: %.3f)", 
             config_.downsampling.enable ? "ON" : "OFF", config_.downsampling.leaf_size);
    ROS_INFO("Cropping: %s (cube_size: %.1f, center: [%.1f,%.1f,%.1f])", 
             config_.cropping.enable ? "ON" : "OFF", config_.cropping.cube_size,
             config_.cropping.center_x, config_.cropping.center_y, config_.cropping.center_z);
    ROS_INFO("Ground Filtering (CSF): %s (threshold: %.2f, iterations: %d)", 
             config_.csf.enable ? "ON" : "OFF", 
             config_.csf.classification_threshold, config_.csf.max_iterations);
    ROS_INFO("Intensity Filtering: %s (%s, factor/range: %.2f)", 
             config_.intensity.enable ? "ON" : "OFF",
             config_.intensity.use_statistical_filtering ? "Statistical" : "Fixed",
             config_.intensity.use_statistical_filtering ? config_.intensity.iqr_factor : 
                (config_.intensity.max_intensity - config_.intensity.min_intensity));
    ROS_INFO("Elevation Filtering: %s (range: [%.1f, %.1f])", 
             config_.elevation.enable ? "ON" : "OFF",
             config_.elevation.min_elevation_threshold, 
             config_.elevation.max_elevation_threshold);
    ROS_INFO("Topics: Input=%s, Output=%s, Frame=%s", 
             config_.lidar_topic.c_str(), 
             config_.output_topic.c_str(), 
             config_.frame_id.c_str());
    ROS_INFO("=============================================================");
}

void MultiLevelPreprocessor::setConfig(const PreprocessConfig& config) {
    config_ = config;
    ROS_INFO("[PreProcessor] Configuration updated manually");
}

MultiLevelPreprocessor::PreprocessConfig MultiLevelPreprocessor::getConfig() const {
    return config_;
}

PreprocessingResult MultiLevelPreprocessor::process(const PointCloudPtr& input_cloud) {
    PreprocessingResult result;
    auto start_time = std::chrono::high_resolution_clock::now();
    
    try {
        // 验证输入点云
        if (!validatePointCloud(input_cloud)) {
            result.statistics.success = false;
            result.statistics.error_message = "Invalid input point cloud";
            ROS_ERROR("[PreProcessor] Invalid input point cloud");
            return result;
        }
        
        if (config_.verbose) {
            ROS_INFO("[PreProcessor] Starting preprocessing pipeline...");
            ROS_INFO("[PreProcessor] Input cloud size: %zu points", input_cloud->size());
        }
        
        // 统计信息记录
        std::vector<std::pair<std::string, size_t>> step_sizes;
        std::vector<std::pair<std::string, double>> step_times;
        
        PointCloudPtr current_cloud = input_cloud;
        step_sizes.emplace_back("input", current_cloud->size());
        
        // 1. 降采样
        if (config_.downsampling.enable) {
            auto step_start = std::chrono::high_resolution_clock::now();
            current_cloud = applyDownsampling(current_cloud);
            auto step_end = std::chrono::high_resolution_clock::now();
            
            double step_time = std::chrono::duration<double>(step_end - step_start).count();
            step_sizes.emplace_back("downsampling", current_cloud->size());
            step_times.emplace_back("downsampling", step_time);
            
            if (config_.verbose) {
                printProcessingInfo("Downsampling", step_sizes[step_sizes.size()-2].second, 
                                  current_cloud->size(), step_time);
            }
        }
        
        // 2. 点云裁剪
        if (config_.cropping.enable) {
            auto step_start = std::chrono::high_resolution_clock::now();
            current_cloud = applyCropping(current_cloud);
            auto step_end = std::chrono::high_resolution_clock::now();
            
            double step_time = std::chrono::duration<double>(step_end - step_start).count();
            step_sizes.emplace_back("cropping", current_cloud->size());
            step_times.emplace_back("cropping", step_time);
            
            if (config_.verbose) {
                printProcessingInfo("Cropping", step_sizes[step_sizes.size()-2].second, 
                                  current_cloud->size(), step_time);
            }
        }
        
        // 3. CSF地面滤波
        if (config_.csf.enable) {
            auto step_start = std::chrono::high_resolution_clock::now();
            current_cloud = applyCSFFilter(current_cloud);
            auto step_end = std::chrono::high_resolution_clock::now();
            
            double step_time = std::chrono::duration<double>(step_end - step_start).count();
            step_sizes.emplace_back("ground_filtering", current_cloud->size());
            step_times.emplace_back("ground_filtering", step_time);
            
            if (config_.verbose) {
                printProcessingInfo("Ground Filtering", step_sizes[step_sizes.size()-2].second, 
                                  current_cloud->size(), step_time);
            }
        }
        
        // 4. 强度滤波
        if (config_.intensity.enable) {
            auto step_start = std::chrono::high_resolution_clock::now();
            current_cloud = applyIntensityFilter(current_cloud);
            auto step_end = std::chrono::high_resolution_clock::now();
            
            double step_time = std::chrono::duration<double>(step_end - step_start).count();
            step_sizes.emplace_back("intensity_filtering", current_cloud->size());
            step_times.emplace_back("intensity_filtering", step_time);
            
            if (config_.verbose) {
                printProcessingInfo("Intensity Filtering", step_sizes[step_sizes.size()-2].second, 
                                  current_cloud->size(), step_time);
            }
        }
        
        // 5. 高程滤波
        if (config_.elevation.enable) {
            auto step_start = std::chrono::high_resolution_clock::now();
            current_cloud = applyElevationFilter(current_cloud);
            auto step_end = std::chrono::high_resolution_clock::now();
            
            double step_time = std::chrono::duration<double>(step_end - step_start).count();
            step_sizes.emplace_back("elevation_filtering", current_cloud->size());
            step_times.emplace_back("elevation_filtering", step_time);
            
            if (config_.verbose) {
                printProcessingInfo("Elevation Filtering", step_sizes[step_sizes.size()-2].second, 
                                  current_cloud->size(), step_time);
            }
        }
        
        // 6. 构建KD树
        if (config_.build_kdtree) {
            auto step_start = std::chrono::high_resolution_clock::now();
            result.kdtree_index = buildKDTreeIndex(current_cloud);
            auto step_end = std::chrono::high_resolution_clock::now();
            
            double step_time = std::chrono::duration<double>(step_end - step_start).count();
            step_times.emplace_back("kdtree_building", step_time);
            
            if (config_.verbose) {
                ROS_INFO("[PreProcessor] KD-Tree built in %.3f seconds", step_time);
            }
        }
        
        // 设置结果
        result.filtered_cloud = current_cloud;
        
        // 计算总体统计
        auto end_time = std::chrono::high_resolution_clock::now();
        result.statistics = computeFinalStatistics(input_cloud, current_cloud, 
                                                 step_sizes, step_times);
        result.statistics.processing_time_seconds = 
            std::chrono::duration<double>(end_time - start_time).count();
        result.statistics.success = true;
        
        if (config_.verbose) {
            ROS_INFO("[PreProcessor] Preprocessing completed successfully!");
            ROS_INFO("[PreProcessor] Final cloud size: %zu points", current_cloud->size());
            ROS_INFO("[PreProcessor] Reduction ratio: %.1f%%", 
                     result.statistics.reduction_ratio * 100);
            ROS_INFO("[PreProcessor] Total processing time: %.3f seconds", 
                     result.statistics.processing_time_seconds);
        }
        
        // 使用ROS_INFO输出关键统计信息
        ROS_INFO("[PreProcessor] Processing complete: %zu -> %zu points (%.1f%% reduction) in %.3fs",
                 result.statistics.input_points,
                 result.statistics.output_points,
                 result.statistics.reduction_ratio * 100,
                 result.statistics.processing_time_seconds);
        
    } catch (const std::exception& e) {
        result.statistics.success = false;
        result.statistics.error_message = e.what();
        ROS_ERROR("[PreProcessor] Error during preprocessing: %s", e.what());
    }
    
    return result;
}

PointCloudPtr MultiLevelPreprocessor::applyDownsampling(const PointCloudPtr& cloud) {
    PointCloudPtr downsampled_cloud(new PointCloud);
    
    pcl::VoxelGrid<pcl::PointXYZI> voxel_filter;
    voxel_filter.setInputCloud(cloud);
    voxel_filter.setLeafSize(config_.downsampling.leaf_size, 
                           config_.downsampling.leaf_size, 
                           config_.downsampling.leaf_size);
    voxel_filter.filter(*downsampled_cloud);
    
    return downsampled_cloud;
}

PointCloudPtr MultiLevelPreprocessor::applyCropping(const PointCloudPtr& cloud) {
    PointCloudPtr cropped_cloud(new PointCloud);
    
    pcl::CropBox<pcl::PointXYZI> crop_filter;
    crop_filter.setInputCloud(cloud);
    
    double half_size = config_.cropping.cube_size / 2.0;
    Eigen::Vector4f min_point(config_.cropping.center_x - half_size,
                             config_.cropping.center_y - half_size,
                             config_.cropping.center_z - half_size, 1.0);
    Eigen::Vector4f max_point(config_.cropping.center_x + half_size,
                             config_.cropping.center_y + half_size,
                             config_.cropping.center_z + half_size, 1.0);
    
    crop_filter.setMin(min_point);
    crop_filter.setMax(max_point);
    crop_filter.filter(*cropped_cloud);
    
    return cropped_cloud;
}

PointCloudPtr MultiLevelPreprocessor::applyCSFFilter(const PointCloudPtr& cloud) {
    // 简化的地面滤波实现（使用Progressive Morphological Filter作为CSF的替代）
    PointCloudPtr non_ground_cloud(new PointCloud);
    
    try {
        pcl::ProgressiveMorphologicalFilter<pcl::PointXYZI> pmf;
        pmf.setInputCloud(cloud);
        pmf.setMaxWindowSize(20);
        pmf.setSlope(1.0f);
        pmf.setInitialDistance(0.5f);
        pmf.setMaxDistance(3.0f);
        
        pcl::PointIndices::Ptr ground_indices(new pcl::PointIndices);
        pmf.extract(ground_indices->indices);
        
        // 提取非地面点
        pcl::ExtractIndices<pcl::PointXYZI> extract;
        extract.setInputCloud(cloud);
        extract.setIndices(ground_indices);
        extract.setNegative(true);  // 提取非地面点
        extract.filter(*non_ground_cloud);
        
        ROS_DEBUG("[PreProcessor] Ground filtering: %zu -> %zu points", 
                  cloud->size(), non_ground_cloud->size());
        
    } catch (const std::exception& e) {
        ROS_WARN("[PreProcessor] Ground filtering failed, using original cloud: %s", e.what());
        *non_ground_cloud = *cloud;
    }
    
    return non_ground_cloud;
}

PointCloudPtr MultiLevelPreprocessor::applyIntensityFilter(const PointCloudPtr& cloud) {
    PointCloudPtr intensity_filtered_cloud(new PointCloud);
    
    if (config_.intensity.use_statistical_filtering) {
        // 统计方法：基于IQR的异常值检测
        auto intensity_stats = computeIntensityStatistics(cloud);
        
        for (const auto& point : cloud->points) {
            if (point.intensity >= intensity_stats.lower_bound && 
                point.intensity <= intensity_stats.upper_bound) {
                intensity_filtered_cloud->points.push_back(point);
            }
        }
        
        ROS_DEBUG("[PreProcessor] Intensity filtering (statistical): %zu -> %zu points", 
                  cloud->size(), intensity_filtered_cloud->size());
    } else {
        // 固定阈值方法
        for (const auto& point : cloud->points) {
            if (point.intensity >= config_.intensity.min_intensity && 
                point.intensity <= config_.intensity.max_intensity) {
                intensity_filtered_cloud->points.push_back(point);
            }
        }
        
        ROS_DEBUG("[PreProcessor] Intensity filtering (fixed): %zu -> %zu points", 
                  cloud->size(), intensity_filtered_cloud->size());
    }
    
    intensity_filtered_cloud->width = intensity_filtered_cloud->points.size();
    intensity_filtered_cloud->height = 1;
    intensity_filtered_cloud->is_dense = false;
    
    return intensity_filtered_cloud;
}

PointCloudPtr MultiLevelPreprocessor::applyElevationFilter(const PointCloudPtr& cloud) {
    PointCloudPtr elevation_filtered_cloud(new PointCloud);
    
    pcl::PassThrough<pcl::PointXYZI> pass_filter;
    pass_filter.setInputCloud(cloud);
    pass_filter.setFilterFieldName("z");
    pass_filter.setFilterLimits(config_.elevation.min_elevation_threshold, 
                               config_.elevation.max_elevation_threshold);
    pass_filter.filter(*elevation_filtered_cloud);
    
    ROS_DEBUG("[PreProcessor] Elevation filtering: %zu -> %zu points", 
              cloud->size(), elevation_filtered_cloud->size());
    
    return elevation_filtered_cloud;
}

KdTreePtr MultiLevelPreprocessor::buildKDTreeIndex(const PointCloudPtr& cloud) {
    KdTreePtr kdtree(new pcl::search::KdTree<pcl::PointXYZI>);
    kdtree->setInputCloud(cloud);
    ROS_DEBUG("[PreProcessor] KD-Tree index built for %zu points", cloud->size());
    return kdtree;
}

IntensityStatistics MultiLevelPreprocessor::computeIntensityStatistics(const PointCloudPtr& cloud) {
    IntensityStatistics stats;
    
    if (cloud->empty()) {
        ROS_WARN("[PreProcessor] Empty cloud for intensity statistics");
        return stats;
    }
    
    std::vector<float> intensities;
    intensities.reserve(cloud->size());
    
    for (const auto& point : cloud->points) {
        intensities.push_back(point.intensity);
    }
    
    std::sort(intensities.begin(), intensities.end());
    
    // 基本统计
    stats.min_intensity = intensities.front();
    stats.max_intensity = intensities.back();
    stats.mean_intensity = std::accumulate(intensities.begin(), intensities.end(), 0.0f) / intensities.size();
    
    // 计算标准差
    float variance = 0.0f;
    for (float intensity : intensities) {
        variance += (intensity - stats.mean_intensity) * (intensity - stats.mean_intensity);
    }
    stats.std_intensity = std::sqrt(variance / intensities.size());
    
    // 四分位数
    size_t q1_idx = intensities.size() / 4;
    size_t q3_idx = 3 * intensities.size() / 4;
    
    stats.q1_intensity = intensities[q1_idx];
    stats.q3_intensity = intensities[q3_idx];
    stats.iqr_intensity = stats.q3_intensity - stats.q1_intensity;
    
    // 异常值边界
    stats.lower_bound = stats.q1_intensity - config_.intensity.iqr_factor * stats.iqr_intensity;
    stats.upper_bound = stats.q3_intensity + config_.intensity.iqr_factor * stats.iqr_intensity;
    
    ROS_DEBUG("[PreProcessor] Intensity stats - Mean: %.1f, Std: %.1f, IQR: [%.1f, %.1f]",
              stats.mean_intensity, stats.std_intensity, stats.q1_intensity, stats.q3_intensity);
    
    return stats;
}

bool MultiLevelPreprocessor::validatePointCloud(const PointCloudPtr& cloud) {
    if (!cloud) {
        ROS_ERROR("[PreProcessor] Null point cloud pointer");
        return false;
    }
    
    if (cloud->empty()) {
        ROS_ERROR("[PreProcessor] Empty point cloud");
        return false;
    }
    
    if (cloud->size() < 100) {
        ROS_WARN("[PreProcessor] Very small point cloud (< 100 points): %zu", cloud->size());
    }
    
    return true;
}

void MultiLevelPreprocessor::printProcessingInfo(const std::string& step, 
                                               size_t input_size, 
                                               size_t output_size, 
                                               double time_elapsed) {
    double reduction = (1.0 - static_cast<double>(output_size) / input_size) * 100.0;
    
    if (config_.verbose) {
        ROS_INFO("[PreProcessor] %s: %zu -> %zu points (%.1f%% reduction) in %.3fs", 
                 step.c_str(), input_size, output_size, reduction, time_elapsed);
    } else {
        ROS_DEBUG("[PreProcessor] %s: %zu -> %zu points (%.1f%% reduction) in %.3fs", 
                  step.c_str(), input_size, output_size, reduction, time_elapsed);
    }
}

PreprocessingStatistics MultiLevelPreprocessor::computeFinalStatistics(
    const PointCloudPtr& input_cloud,
    const PointCloudPtr& output_cloud,
    const std::vector<std::pair<std::string, size_t>>& step_sizes,
    const std::vector<std::pair<std::string, double>>& step_times) {
    
    PreprocessingStatistics stats;
    
    stats.input_points = input_cloud->size();
    stats.output_points = output_cloud->size();
    stats.reduction_ratio = 1.0 - static_cast<double>(stats.output_points) / stats.input_points;
    
    // 填充各步骤统计
    for (const auto& step : step_sizes) {
        if (step.first == "downsampling") stats.points_after_downsampling = step.second;
        else if (step.first == "cropping") stats.points_after_cropping = step.second;
        else if (step.first == "ground_filtering") stats.points_after_ground_filtering = step.second;
        else if (step.first == "intensity_filtering") stats.points_after_intensity_filtering = step.second;
        else if (step.first == "elevation_filtering") stats.points_after_elevation_filtering = step.second;
    }
    
    // 填充时间统计
    for (const auto& step : step_times) {
        if (step.first == "downsampling") stats.downsampling_time = step.second;
        else if (step.first == "cropping") stats.cropping_time = step.second;
        else if (step.first == "ground_filtering") stats.ground_filtering_time = step.second;
        else if (step.first == "intensity_filtering") stats.intensity_filtering_time = step.second;
        else if (step.first == "elevation_filtering") stats.elevation_filtering_time = step.second;
        else if (step.first == "kdtree_building") stats.kdtree_building_time = step.second;
    }
    
    return stats;
}

} // namespace powerline_extraction