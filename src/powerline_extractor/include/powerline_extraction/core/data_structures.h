
// ================= 1. 数据结构定义 =================
// include/powerline_extraction/core/data_structures.h
#ifndef POWERLINE_EXTRACTION_DATA_STRUCTURES_H
#define POWERLINE_EXTRACTION_DATA_STRUCTURES_H

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/search/kdtree.h>
#include <vector>
#include <string>
#include <chrono>

namespace powerline_extraction {

// 基础点云类型定义
using PointCloud = pcl::PointCloud<pcl::PointXYZI>;
using PointCloudPtr = pcl::PointCloud<pcl::PointXYZI>::Ptr;
using KdTreePtr = pcl::search::KdTree<pcl::PointXYZI>::Ptr;

// 预处理统计信息
struct PreprocessingStatistics {
    size_t input_points = 0;
    size_t output_points = 0;
    double reduction_ratio = 0.0;
    double processing_time_seconds = 0.0;
    
    // 各步骤统计
    size_t points_after_downsampling = 0;
    size_t points_after_cropping = 0;
    size_t points_after_ground_filtering = 0;
    size_t points_after_intensity_filtering = 0;
    size_t points_after_elevation_filtering = 0;
    
    // 处理时间
    double downsampling_time = 0.0;
    double cropping_time = 0.0;
    double ground_filtering_time = 0.0;
    double intensity_filtering_time = 0.0;
    double elevation_filtering_time = 0.0;
    double kdtree_building_time = 0.0;
    
    std::string error_message;
    bool success = true;
};

// 预处理结果
struct PreprocessingResult {
    PointCloudPtr filtered_cloud;
    KdTreePtr kdtree_index;
    PreprocessingStatistics statistics;
    
    PreprocessingResult() {
        filtered_cloud = PointCloudPtr(new PointCloud);
        kdtree_index = KdTreePtr(new pcl::search::KdTree<pcl::PointXYZI>);
    }
};

// 强度统计信息
struct IntensityStatistics {
    float min_intensity = 0.0f;
    float max_intensity = 0.0f;
    float mean_intensity = 0.0f;
    float std_intensity = 0.0f;
    float q1_intensity = 0.0f;
    float q3_intensity = 0.0f;
    float iqr_intensity = 0.0f;
    float lower_bound = 0.0f;
    float upper_bound = 0.0f;
};

} // namespace powerline_extraction

#endif // POWERLINE_EXTRACTION_DATA_STRUCTURES_H