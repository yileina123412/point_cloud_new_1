#ifndef POWERLINE_EXTRACTOR_H
#define POWERLINE_EXTRACTOR_H

#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_sensor_msgs/tf2_sensor_msgs.h>
#include <pcl_ros/transforms.h>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/common/common.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/io/pcd_io.h>

#include <Eigen/Dense>
#include <string>
#include <vector>
#include <memory>

#include "point_cloud_preprocessor.h"
#include "power_line_coarse_extractor_s.h"
#include "power_line_fine_extraction.h"
#include "obstacle_analyzer.h"
#include "power_line_reconstruction.h"
#include "building_edge_filter.h"
#include "advanced_obstacle_analyzer.h"
// #include "power_line_probability_map.h"
#include "power_line_probability_map.h"
#include "power_line_tracker.h"
#include "enhanced_power_line_tracker.h"
#include "imu_orientation_estimator.h"



//================= 融合多技术的现代电力线提取完整解决方案 =================
#include "powerline_extraction/core/data_structures.h"
#include "powerline_extraction/preprocessing/multilevel_preprocessor.h"

// 前向声明粗提取器
class PowerlineCoarseExtractor;

class PowerlineExtractor {
public:
    PowerlineExtractor(ros::NodeHandle& nh, ros::NodeHandle& private_nh);
    ~PowerlineExtractor();

private:
    // 回调函数
    void pointCloudCallback(const sensor_msgs::PointCloud2::ConstPtr& msg);
    
    // 参数检查和初始化

    // 在现有私有函数中添加
    void initializeAccumulateCloud();    // 初始化预处理管道

    void loadParameters();
    void initializePublishers();
    void initializeSubscribers();

    void initializeIMUEstimator();       // 初始化IMU姿态估计器

    void initializeFineExtractor();
    
    // 点云变换
    bool transformPointCloud(const sensor_msgs::PointCloud2::ConstPtr& input_msg,
                           sensor_msgs::PointCloud2& transformed_msg);
    
    // 点云水平校正 - 新增函数  imu矫正
    void correctPointCloudOrientation(pcl::PointCloud<pcl::PointXYZI>::Ptr& cloud);
    
    // 发布点云
    void publishPointClouds(const pcl::PointCloud<pcl::PointXYZI>::Ptr& original_cloud,
                          const pcl::PointCloud<pcl::PointXYZI>::Ptr& powerline_cloud,
                          const pcl::PointCloud<pcl::PointXYZI>::Ptr& clustered_cloud,
                          const std_msgs::Header& header);

    void process_first_method(const std_msgs::Header& header);   //第一种方法的执行函数

    void process_second_times(const std_msgs::Header& header);  //第二次及以后

private:
    // ROS 节点句柄
    ros::NodeHandle nh_;
    ros::NodeHandle private_nh_;
    
    // 订阅器和发布器

    ros::Publisher preprocessor_cloud_pub_;    //预处理后的点云
    ros::Publisher extractor_s_cloud_pub_;    //粗提取_s后的点云
    ros::Publisher env_not_power_cloud_pub_;    //粗提取_s后的环境点云点云 
    ros::Publisher fine_extractor_cloud_pub_;      
    ros::Subscriber point_cloud_sub_;
    ros::Publisher original_cloud_pub_;
    ros::Publisher powerline_cloud_pub_;
    ros::Publisher clustered_powerline_cloud_pub_;
    ros::Publisher obb_marker_pub;  //距离可视化
    ros::Publisher powerlines_distance_cloud_pub_;

    ros::Publisher rol_cloud_pub_;

    ros::Publisher second_powerline_preprocessor_cloud_pub_;

    // IMU校正后点云发布器
    ros::Publisher corrected_cloud_pub_;


    
    // TF变换
    std::unique_ptr<tf2_ros::Buffer> tf_buffer_;
    std::unique_ptr<tf2_ros::TransformListener> tf_listener_;

    //================= 初代方案 =================
    //点云数据预处理
    std::unique_ptr<PointCloudPreprocessor> preprocessor_;  //实例化类
    pcl::PointCloud<pcl::PointXYZI>::Ptr preprocessor__output_cloud_;  //输出预处理后的点云
    // pcl::octree::OctreePointCloudChangeDetector<pcl::PointXYZI>& octree_;  //输出的octree     octree = preprocessor.getOctree();

    //粗提取_s
    std::unique_ptr<PowerLineCoarseExtractor> extractor_s_; 
    pcl::PointCloud<pcl::PointXYZI>::Ptr extractor_s__output_cloud_;
    pcl::PointCloud<pcl::PointXYZI>::Ptr env_not_power_cloud_;

    //距离可视化
    std::unique_ptr<ObstacleAnalyzer> analyzer_; 
    std::vector<OrientedBoundingBox> obbs_;

    //精提取器
    std::unique_ptr<PowerLineFineExtractor> fine_extractor_;

    //重构
    std::unique_ptr<PowerLineReconstructor> reconstruction_;
    std::vector<ReconstructedPowerLine> power_lines_;
    pcl::PointCloud<pcl::PointXYZI>::Ptr reconstruction_output_cloud_;

    //过滤
    std::unique_ptr<BuildingEdgeFilter> building_edge_filter_;
    pcl::PointCloud<pcl::PointXYZI>::Ptr building_edge_filter_output_cloud_;

    //预警
    // 发布器
    std::unique_ptr<AdvancedObstacleAnalyzer> obs_analyzer_;
    std::vector<AdvancedObstacleBox> obstacle_results_;
    LayeredWarningCloud warning_cloud_;

    //概率地图
    // std::unique_ptr<PowerLineProbabilityMap> prob_map_;
    std::unique_ptr<PowerLineProbabilityMap> prob_map_;
    std::vector<LineROIInfo> line_rois_;  // 2. 获取概率地图的ROI信息

    //ROI管理器

    pcl::PointCloud<pcl::PointXYZI>::Ptr roi_output_cloud_;
    // 跟踪
    std::unique_ptr<PowerLineTracker> tracker_;
    std::vector<ReconstructedPowerLine> complete_result;

    std::unique_ptr<EnhancedPowerLineTracker> enhanced_tracker_;  //增强版跟踪器
    std::vector<ReconstructedPowerLine> enhanced_complete_result;

    // IMU姿态估计器
    std::unique_ptr<IMUOrientationEstimator> imu_estimator_;
    // IMU相关参数
    bool enable_imu_correction_;  // 是否启用IMU姿态校正

    //================= 融合多技术的现代电力线提取完整解决方案 =================
    std::unique_ptr<powerline_extraction::MultiLevelPreprocessor> multi_preprocessor_;  //多层级预处理
    
    
    // 参数
    std::string lidar_topic_;
    std::string lidar_frame_;
    std::string target_frame_;
    




    
    // 点云对象


    pcl::PointCloud<pcl::PointXYZI>::Ptr original_cloud_;
    pcl::PointCloud<pcl::PointXYZI>::Ptr non_ground_cloud_;  // 保留用于兼容性，但不使用
    pcl::PointCloud<pcl::PointXYZI>::Ptr powerline_cloud_;
    pcl::PointCloud<pcl::PointXYZI>::Ptr clustered_powerline_cloud_;


    pcl::PointCloud<pcl::PointXYZI>::Ptr filtered_pc_;

    pcl::PointCloud<pcl::PointXYZI>::Ptr fine_extract_cloud_;

    // 校正后的点云
    pcl::PointCloud<pcl::PointXYZI>::Ptr corrected_cloud_;
    
    // 处理标志
    bool first_cloud_received_;
    ros::Time last_process_time_;
    double process_frequency_;
    // 状态变量
    bool is_first_frame_;
    int frame_count_;
};

#endif // POWERLINE_EXTRACTOR_H