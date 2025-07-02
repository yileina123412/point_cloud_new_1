// ================= ROS集成版本的预处理器头文件 =================
// include/powerline_extraction/preprocessing/multilevel_preprocessor.h
/*
输入：
pcl::PointCloud<pcl::PointXYZI>::Ptr input_cloud
输出：
PreprocessingResult {
    PointCloudPtr filtered_cloud;     // 处理后的点云
    KdTreePtr kdtree_index;          // KD树索引
    PreprocessingStatistics statistics; // 详细统计信息
}
*/
#ifndef POWERLINE_EXTRACTION_MULTILEVEL_PREPROCESSOR_H
#define POWERLINE_EXTRACTION_MULTILEVEL_PREPROCESSOR_H

#include "../core/data_structures.h"
#include <ros/ros.h>
#include <memory>

namespace powerline_extraction {

class MultiLevelPreprocessor {
public:
    // 预处理配置结构
    struct PreprocessConfig {
        // 降采样参数
        struct DownsamplingParams {
            bool enable = true;
            double leaf_size = 0.1;  // 体素大小(m)
        } downsampling;
        
        // 点云裁剪参数
        struct CroppingParams {
            bool enable = true;
            double cube_size = 70.0;  // 立方体边长(m)
            double center_x = 0.0;    // 立方体中心坐标
            double center_y = 0.0;
            double center_z = 0.0;
        } cropping;
        
        // CSF地面滤波参数
        struct CSFParams {
            bool enable = false;  // 默认关闭
            double classification_threshold = 0.4;
            int max_iterations = 500;
            double cloth_resolution = 0.1;
            int rigidness = 3;
            double time_step = 1.65;
        } csf;
        
        // 反射强度滤波参数
        struct IntensityParams {
            bool enable = false;  // 默认关闭
            double iqr_factor = 1.5;
            double min_intensity = 8200.0;
            double max_intensity = 9000.0;
            bool use_statistical_filtering = true;  // 使用统计滤波还是固定阈值
        } intensity;
        
        // 高程过滤参数
        struct ElevationParams {
            bool enable = false;  // 默认关闭
            double min_elevation_threshold = 10.0;
            double max_elevation_threshold = 80.0;
            bool relative_to_ground = false;  // 是否相对于地面高度
        } elevation;
        
        // 通用参数
        bool build_kdtree = true;  // 是否构建KD树
        bool verbose = false;      // 是否输出详细信息
        
        // ROS相关参数
        std::string lidar_topic = "/livox/lidar";
        std::string output_topic = "/powerline/preprocessed_cloud";
        std::string frame_id = "map";
    };

    // 构造函数 - 添加ROS NodeHandle参数
    explicit MultiLevelPreprocessor(ros::NodeHandle& nh);
    explicit MultiLevelPreprocessor(ros::NodeHandle& nh, const PreprocessConfig& config);
    
    // 析构函数
    ~MultiLevelPreprocessor() = default;
    
    // 主要接口
    PreprocessingResult process(const PointCloudPtr& input_cloud);
    
    // 配置管理
    void loadROSParameters();  // 从ROS参数服务器加载参数
    void setConfig(const PreprocessConfig& config);
    PreprocessConfig getConfig() const;
    
    // 单步处理接口（用于调试）
    PointCloudPtr applyDownsampling(const PointCloudPtr& cloud);
    PointCloudPtr applyCropping(const PointCloudPtr& cloud);
    PointCloudPtr applyCSFFilter(const PointCloudPtr& cloud);
    PointCloudPtr applyIntensityFilter(const PointCloudPtr& cloud);
    PointCloudPtr applyElevationFilter(const PointCloudPtr& cloud);
    KdTreePtr buildKDTreeIndex(const PointCloudPtr& cloud);
    
    // ROS相关接口
    std::string getLidarTopic() const { return config_.lidar_topic; }
    std::string getOutputTopic() const { return config_.output_topic; }
    std::string getFrameId() const { return config_.frame_id; }
    
private:
    PreprocessConfig config_;
    ros::NodeHandle& nh_;          // ROS节点句柄引用
    ros::NodeHandle private_nh_;   // 私有节点句柄，用于参数读取
    
    // 内部辅助方法
    IntensityStatistics computeIntensityStatistics(const PointCloudPtr& cloud);
    void printProcessingInfo(const std::string& step, 
                           size_t input_size, 
                           size_t output_size, 
                           double time_elapsed);
    bool validatePointCloud(const PointCloudPtr& cloud);
    PreprocessingStatistics computeFinalStatistics(
        const PointCloudPtr& input_cloud,
        const PointCloudPtr& output_cloud,
        const std::vector<std::pair<std::string, size_t>>& step_sizes,
        const std::vector<std::pair<std::string, double>>& step_times);
    
    // ROS参数加载辅助方法
    void loadDownsamplingParams();
    void loadCroppingParams();
    void loadCSFParams();
    void loadIntensityParams();
    void loadElevationParams();
    void loadGeneralParams();
    void printROSParameters();  // 输出加载的参数信息
};

} // namespace powerline_extraction

#endif // POWERLINE_EXTRACTION_MULTILEVEL_PREPROCESSOR_H