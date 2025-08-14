#include "powerline_extractor.h"


PowerlineExtractor::PowerlineExtractor(ros::NodeHandle& nh, ros::NodeHandle& private_nh)
    : nh_(nh), 
      private_nh_(private_nh),
      first_cloud_received_(false),
      preprocessor__output_cloud_(new pcl::PointCloud<pcl::PointXYZI>()),
      extractor_s__output_cloud_(new pcl::PointCloud<pcl::PointXYZI>()),
      env_not_power_cloud_(new pcl::PointCloud<pcl::PointXYZI>()),
      original_cloud_(new pcl::PointCloud<pcl::PointXYZI>()),
      non_ground_cloud_(new pcl::PointCloud<pcl::PointXYZI>()),
      powerline_cloud_(new pcl::PointCloud<pcl::PointXYZI>()),
      clustered_powerline_cloud_(new pcl::PointCloud<pcl::PointXYZI>()),
      fine_extract_cloud_(new pcl::PointCloud<pcl::PointXYZI>()),
      filtered_pc_(new pcl::PointCloud<pcl::PointXYZI>()),
      reconstruction_output_cloud_(new pcl::PointCloud<pcl::PointXYZI>()),
      roi_output_cloud_(new pcl::PointCloud<pcl::PointXYZI>()),
      building_edge_filter_output_cloud_(new pcl::PointCloud<pcl::PointXYZI>()),
      corrected_cloud_(new pcl::PointCloud<pcl::PointXYZI>()),  // 新增
      enable_imu_correction_(true),
      prob_map_output_cloud_(new pcl::PointCloud<pcl::PointXYZI>()),
      need_reset_detection_(false){

    //
    is_first_frame_ = 0;
    frame_count_ = 0;
    
    // 初始化TF
    tf_buffer_ = std::make_unique<tf2_ros::Buffer>();
    tf_listener_ = std::make_unique<tf2_ros::TransformListener>(*tf_buffer_);
    
    // 加载参数
    loadParameters();
    //初始化累积点云
    initializeAccumulateCloud();
    
    // 初始化IMU姿态估计器 
    initializeIMUEstimator();
    // 初始化精提取器
    initializeFineExtractor();
    
    // 初始化发布器和订阅器
    initializePublishers();
    initializeSubscribers();
    
    ROS_INFO("PowerlineExtractor initialized successfully");
}

PowerlineExtractor::~PowerlineExtractor() {
    ROS_INFO("PowerlineExtractor destructor called");
}

void PowerlineExtractor::loadParameters() {
    // 雷达相关参数
    private_nh_.param<std::string>("lidar_topic", lidar_topic_, "/livox/lidar");
    private_nh_.param<std::string>("lidar_frame", lidar_frame_, "livox_frame");
    private_nh_.param<std::string>("target_frame", target_frame_, "map");
    

    

    
  
    

    
    // 处理频率
    private_nh_.param<double>("process_frequency", process_frequency_, 2.0);
    
    // 打印参数
    ROS_INFO("=== Powerline Extractor Parameters ===");
    ROS_INFO("Lidar topic: %s", lidar_topic_.c_str());
    ROS_INFO("Lidar frame: %s", lidar_frame_.c_str());
    ROS_INFO("Target frame: %s", target_frame_.c_str());



    ROS_INFO("Process frequency: %.1f Hz", process_frequency_);
}

// 初始化IMU姿态估计器
void PowerlineExtractor::initializeIMUEstimator() {
    try {
        imu_estimator_ = std::make_unique<IMUOrientationEstimator>(nh_, private_nh_);
        ROS_INFO("IMU Orientation Estimator initialized successfully");
    } catch (const std::exception& e) {
        ROS_ERROR("Failed to initialize IMU Orientation Estimator: %s", e.what());
        enable_imu_correction_ = false;
    }
}

void PowerlineExtractor::initializeFineExtractor(){

    fine_extractor_ = std::make_unique<PowerLineFineExtractor>(nh_);
    reconstruction_.reset(new PowerLineReconstructor(nh_));
    building_edge_filter_.reset(new BuildingEdgeFilter(nh_));

    ROS_INFO("Fine extractor initialized successfully");


}

void PowerlineExtractor::initializePublishers() {

    preprocessor_cloud_pub_ = private_nh_.advertise<sensor_msgs::PointCloud2>("preprocessor_cloud", 1);
    powerlines_distance_cloud_pub_ = private_nh_.advertise<visualization_msgs::MarkerArray>("powerlines_distance_cloud", 1);

    extractor_s_cloud_pub_ = private_nh_.advertise<sensor_msgs::PointCloud2>("extractor_s_cloud", 1);
    env_not_power_cloud_pub_ = private_nh_.advertise<sensor_msgs::PointCloud2>("env_not_power_cloud", 1);

    obb_marker_pub = nh_.advertise<visualization_msgs::MarkerArray>("obb_marker", 10);



    original_cloud_pub_ = private_nh_.advertise<sensor_msgs::PointCloud2>("original_cloud", 1);
    powerline_cloud_pub_ = private_nh_.advertise<sensor_msgs::PointCloud2>("powerline_cloud", 1);
    clustered_powerline_cloud_pub_ = private_nh_.advertise<sensor_msgs::PointCloud2>("clustered_powerline_cloud", 1);

    

    fine_extractor_cloud_pub_ = private_nh_.advertise<sensor_msgs::PointCloud2>("fine_extractor_cloud", 1);
    rol_cloud_pub_ = private_nh_.advertise<sensor_msgs::PointCloud2>("rol_cloud", 1);

    // IMU校正后点云发布器
    corrected_cloud_pub_ = private_nh_.advertise<sensor_msgs::PointCloud2>("corrected_cloud", 1);
    ROS_INFO("Publishers initialized");
}
void PowerlineExtractor::initializeAccumulateCloud()
{
    //点云数据预处理
    preprocessor_.reset(new PointCloudPreprocessor(nh_));
    //粗提取_s
    extractor_s_.reset(new PowerLineCoarseExtractor(nh_));

    //可视化距离
    analyzer_.reset(new ObstacleAnalyzer(nh_));
    obs_analyzer_.reset(new AdvancedObstacleAnalyzer(nh_));

    prob_map_.reset(new PowerLineProbabilityMap(nh_));

    enhanced_tracker_.reset(new EnhancedPowerLineTracker(nh_));


    ROS_INFO("Accumulate Cloud initialized");


}
void PowerlineExtractor::initializeSubscribers() {
    point_cloud_sub_ = nh_.subscribe(lidar_topic_, 1, &PowerlineExtractor::pointCloudCallback, this);
    ROS_INFO("Subscribed to point cloud topic: %s", lidar_topic_.c_str());
}


// IMU点云水平校正函数
void PowerlineExtractor::correctPointCloudOrientation(pcl::PointCloud<pcl::PointXYZI>::Ptr& cloud) {
    if (!enable_imu_correction_ || !imu_estimator_ || !imu_estimator_->isIMUDataValid()) {
        ROS_WARN_THROTTLE(10.0, "IMU correction disabled or IMU data not valid");
        *corrected_cloud_ = *cloud;  // 直接复制原始点云
        return;
    }
    
    try {
        // 获取水平校正变换矩阵
        Eigen::Matrix4f transform_matrix = imu_estimator_->getHorizontalTransformMatrix();
        
        // 应用变换到点云
        pcl::transformPointCloud(*cloud, *corrected_cloud_, transform_matrix);
        
        // // 用校正后的点云替换原始点云
        // *cloud = *corrected_cloud_;
        
        // 记录调试信息
        double pitch = imu_estimator_->getCurrentPitch() * 180.0 / M_PI;
        double roll = imu_estimator_->getCurrentRoll() * 180.0 / M_PI;
        ROS_INFO("Applied IMU correction - Pitch: %.2f°, Roll: %.2f°", pitch, roll);
        
    } catch (const std::exception& e) {
        ROS_INFO("Error applying IMU correction: %s", e.what());
        *corrected_cloud_ = *cloud;  // 发生错误时使用原始点云
    }
}

void PowerlineExtractor::process_first_method(const std_msgs::Header& header){

    auto pre_start = std::chrono::high_resolution_clock::now();
    preprocessor_->processPointCloud(original_cloud_);  //0.04
    preprocessor_->publishColoredClusters();
    auto pre_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> pre_duration = pre_end - pre_start;
    ROS_INFO("预处理电力线 执行时间: %f 秒", pre_duration.count());


    preprocessor__output_cloud_ = preprocessor_->getProcessedCloud();
    auto extractor_start = std::chrono::high_resolution_clock::now();
    extractor_s_->extractPowerLinesByPoints(preprocessor__output_cloud_);    //0.3
    env_not_power_cloud_ = extractor_s_->getEnvWithoutPowerCloud();
    // extractor_s_->visualizeParameters(preprocessor_);
    auto extractor_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> extractor_duration = extractor_end - extractor_start;
    ROS_INFO("粗提取电力线 执行时间: %f 秒", extractor_duration.count());

    extractor_s__output_cloud_ = extractor_s_->getExtractedCloud();
    ROS_INFO("extractor_s__output_cloud_ 粗提取_s: %ld",extractor_s__output_cloud_->size());
    
    // 发布结果
    publishPointClouds(original_cloud_, powerline_cloud_, 
                        clustered_powerline_cloud_, header);
    
    // 精提取
    fine_extractor_->extractPowerLines(extractor_s__output_cloud_,fine_extract_cloud_); //0.09s



  
    reconstruction_->reconstructPowerLines(extractor_s__output_cloud_,reconstruction_output_cloud_,power_lines_);//0.2s
   
    // building_edge_filter_->filterBuildingEdges(preprocessor__output_cloud_,power_lines_,building_edge_filter_output_cloud_);
    
    // auto analy_star = std::chrono::high_resolution_clock::now();

    // obs_analyzer_->analyzeObstacles(power_lines_,env_not_power_cloud_,obstacle_results_,warning_cloud_);
    // auto analy_end = std::chrono::high_resolution_clock::now();
    // std::chrono::duration<double> analy_duration = analy_end - analy_star;
    // ROS_INFO("障碍物检测 执行时间: %f 秒", analy_duration.count());

    // //发布距离可视化
    // analyzer_->publishObbMarkers(obbs_, obb_marker_pub, "map");
    // analyzer_->publishPowerlineDistanceMarkers(fine_extract_cloud_,powerlines_distance_cloud_pub_,"map");

}

void PowerlineExtractor::process_second_times(const std_msgs::Header& header){
    auto pre_start = std::chrono::high_resolution_clock::now();
    preprocessor_->processPointCloud(original_cloud_);  //0.04
    preprocessor_->publishColoredClusters();
    auto pre_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> pre_duration = pre_end - pre_start;
    ROS_INFO("预处理电力线 执行时间: %f 秒", pre_duration.count());
    preprocessor__output_cloud_ = preprocessor_->getProcessedCloud();
    // line_rois_ = prob_map_->getAllLineROIs(0.3f, 0.1f);  // 获取所有电力线的ROI信息



    roi_output_cloud_ = prob_map_->processEnvironmentPointCloud(preprocessor__output_cloud_);
    ROS_INFO("roi_output_cloud_ roi提取: %ld",roi_output_cloud_->size());


    
    auto extractor_start = std::chrono::high_resolution_clock::now();
    extractor_s_->extractPowerLinesByPoints(roi_output_cloud_);    //0.3
    env_not_power_cloud_ = extractor_s_->getEnvWithoutPowerCloud();
    // extractor_s_->visualizeParameters(preprocessor_);
    auto extractor_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> extractor_duration = extractor_end - extractor_start;
    ROS_INFO("粗提取电力线 执行时间: %f 秒", extractor_duration.count());

    extractor_s__output_cloud_ = extractor_s_->getExtractedCloud();
    ROS_INFO("extractor_s__output_cloud_ 粗提取_s: %ld",extractor_s__output_cloud_->size());
    
    // 发布结果
    publishPointClouds(original_cloud_, powerline_cloud_, 
                        clustered_powerline_cloud_, header);
    
    // 精提取
    // fine_extractor_->extractPowerLines(extractor_s__output_cloud_,fine_extract_cloud_); //0.09s



  
    reconstruction_->reconstructPowerLines(extractor_s__output_cloud_,reconstruction_output_cloud_,power_lines_);//0.2s

    //跟踪器跟踪

    // complete_result = tracker_->updateTracker(power_lines_, *prob_map_);
    // enhanced_complete_result = enhanced_tracker_->updateTracker(power_lines_, *prob_map_);

    // building_edge_filter_->filterBuildingEdges(preprocessor__output_cloud_,power_lines_,building_edge_filter_output_cloud_);
    prob_map_output_cloud_ = prob_map_->getProbabilityPointCloud();
    
    auto analy_star = std::chrono::high_resolution_clock::now();
    // analyzer_->analyzeObstacles(env_not_power_cloud_, fine_extract_cloud_, obbs_); //0.4
    // 创建增强的电力线列表（原始检测 + 稳定电力线）
    std::vector<ReconstructedPowerLine> enhanced_power_lines;
    reconstruction_->separateCompletePowerLines(prob_map_output_cloud_,enhanced_power_lines);//0.2s
    
    obs_analyzer_->analyzeObstacles(enhanced_power_lines,preprocessor__output_cloud_,obstacle_results_,warning_cloud_);
    // obs_analyzer_->analyzeObstacles(power_lines_,env_not_power_cloud_,prob_map_output_cloud_,obstacle_results_,warning_cloud_);
    auto analy_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> analy_duration = analy_end - analy_star;
    ROS_INFO("障碍物检测 执行时间: %f 秒", analy_duration.count());
    //发布距离可视化
    // analyzer_->publishObbMarkers(obbs_, obb_marker_pub, "map");
    // analyzer_->publishPowerlineDistanceMarkers(fine_extract_cloud_,powerlines_distance_cloud_pub_,"map");

}


void PowerlineExtractor::pointCloudCallback(const sensor_msgs::PointCloud2::ConstPtr& msg) {

    static int run_times = 0;
    // 频率控制
    ros::Time current_time = ros::Time::now();
    if (first_cloud_received_ && 
        (current_time - last_process_time_).toSec() < (1.0 / process_frequency_)) {
        return;
    }
    
    last_process_time_ = current_time;
    first_cloud_received_ = true;
    
    ROS_DEBUG("Received point cloud with %d points", msg->width * msg->height);
    
    try {


        // 检查设备稳定性
    if (enable_imu_correction_ && imu_estimator_ && imu_estimator_->isIMUDataValid()) {
        // 检查是否有移动
        if (imu_estimator_->hasRecentMovement() || !imu_estimator_->isDeviceStable()) {
            if (!need_reset_detection_) {
                ROS_INFO("Device movement detected! Will reset powerline detection when stable.");

                need_reset_detection_ = true;  // 标记需要重置
            }
            ROS_INFO_THROTTLE(2.0, "Device is moving/unstable, skipping point cloud processing");
            return;
        }
        
        // 如果现在稳定了，但之前有移动，需要重置
        if (need_reset_detection_) {
            ROS_WARN("Device stabilized after movement. Resetting powerline detection...");
            run_times = 0;
            need_reset_detection_ = false;
        }
    }
        
        // 坐标变换
        sensor_msgs::PointCloud2 transformed_msg;
        if (!transformPointCloud(msg, transformed_msg)) {
            ROS_WARN("Failed to transform point cloud, skipping this frame");
            return;
        }
        
        // 转换为PCL格式
        pcl::fromROSMsg(transformed_msg, *original_cloud_);
        
        if (original_cloud_->empty()) {
            ROS_WARN("Received empty point cloud");
            return;
        }
        run_times ++;
        ROS_INFO("程序运行的第: %d 轮。",run_times);

        // 新增：应用IMU姿态校正

        correctPointCloudOrientation(original_cloud_);

        if(run_times == 1) is_first_frame_ = 0;
        else is_first_frame_ = 1;
        // ROS_INFO("is_first_frame_ = %d",is_first_frame_);
        if(is_first_frame_ == 0)
        {
            auto start_time = std::chrono::high_resolution_clock::now();
            process_first_method(transformed_msg.header);
            auto end_time = std::chrono::high_resolution_clock::now();
            double all_time = std::chrono::duration<double, std::milli>(end_time - start_time).count();
            ROS_INFO("===================程序运行的第: %d 轮。",run_times);
            ROS_INFO("整个过程总共运行时间: %.3f ms", 
                        all_time);
            is_first_frame_ = 1;
            prob_map_->initializeProbabilityMap(power_lines_);
            prob_map_->resetProbabilityMap();
            prob_map_->initializeProbabilityMap(power_lines_);

            // tracker_->initializeTracker(power_lines_); // 初始化跟踪器
        }
        else{
            if(run_times == 2)
            {
                // enhanced_tracker_->initializeTracker(*prob_map_); // 初始化增强跟踪器
            }
            auto start_time = std::chrono::high_resolution_clock::now();
            process_second_times(transformed_msg.header);
            auto end_time = std::chrono::high_resolution_clock::now();
            double all_time = std::chrono::duration<double, std::milli>(end_time - start_time).count();
            ROS_INFO("=================   second  程序运行的第: %d 轮。",run_times);
            ROS_INFO("整个过程总共运行时间: %.3f ms", 
                        all_time);
            prob_map_->updateProbabilityMap(power_lines_);
            // prob_map_->updateProbabilityMap(complete_result);

        }

        // publishPointClouds(original_cloud_, powerline_cloud_, 
        //                 clustered_powerline_cloud_, transformed_msg.header);

                     
    } catch (const std::exception& e) {
        ROS_ERROR("Error processing point cloud: %s", e.what());
    }
}

bool PowerlineExtractor::transformPointCloud(const sensor_msgs::PointCloud2::ConstPtr& input_msg,
                                            sensor_msgs::PointCloud2& transformed_msg) {
    try {
        // 检查是否需要变换
        if (input_msg->header.frame_id == target_frame_) {
            transformed_msg = *input_msg;
            return true;
        }
        
        // 查找变换
        geometry_msgs::TransformStamped transform_stamped;
        transform_stamped = tf_buffer_->lookupTransform(
            target_frame_, input_msg->header.frame_id, 
            input_msg->header.stamp, ros::Duration(1.0));
        
        // 执行变换
        tf2::doTransform(*input_msg, transformed_msg, transform_stamped);
        transformed_msg.header.frame_id = target_frame_;
        
        return true;
        
    } catch (tf2::TransformException& ex) {
        ROS_WARN("Could not transform point cloud: %s", ex.what());
        return false;
    }
}



void PowerlineExtractor::publishPointClouds(const pcl::PointCloud<pcl::PointXYZI>::Ptr& original_cloud,
                                          const pcl::PointCloud<pcl::PointXYZI>::Ptr& powerline_cloud,
                                          const pcl::PointCloud<pcl::PointXYZI>::Ptr& clustered_cloud,
                                          const std_msgs::Header& header) {
    
    // 发布原始点云
    if (original_cloud_pub_.getNumSubscribers() > 0 && !original_cloud->empty()) {
        sensor_msgs::PointCloud2 original_msg;
        pcl::toROSMsg(*original_cloud, original_msg);
        original_msg.header = header;
        original_cloud_pub_.publish(original_msg);
    }
    // IMU发布校正后的点云
    if (corrected_cloud_pub_.getNumSubscribers() > 0 && !corrected_cloud_->empty()) {
        sensor_msgs::PointCloud2 corrected_msg;
        pcl::toROSMsg(*corrected_cloud_, corrected_msg);
        corrected_msg.header = header;
        corrected_cloud_pub_.publish(corrected_msg);
    }

    if( preprocessor_cloud_pub_.getNumSubscribers() > 0 && !preprocessor__output_cloud_->empty()){
        sensor_msgs::PointCloud2 temp_msg;
        pcl::toROSMsg(*preprocessor__output_cloud_, temp_msg);
        temp_msg.header = header;
        preprocessor_cloud_pub_.publish(temp_msg);
    }

    if( extractor_s_cloud_pub_.getNumSubscribers() > 0 && !extractor_s__output_cloud_->empty()){
        sensor_msgs::PointCloud2 temp_msg;
        pcl::toROSMsg(*extractor_s__output_cloud_, temp_msg);
        temp_msg.header = header;
        extractor_s_cloud_pub_.publish(temp_msg);
    }

    if( rol_cloud_pub_.getNumSubscribers() > 0 && !roi_output_cloud_->empty()){
        sensor_msgs::PointCloud2 temp_msg;
        pcl::toROSMsg(*roi_output_cloud_, temp_msg);
        temp_msg.header = header;
        rol_cloud_pub_.publish(temp_msg);
    }

    if( env_not_power_cloud_pub_.getNumSubscribers() > 0 && !env_not_power_cloud_->empty()){
        sensor_msgs::PointCloud2 temp_msg;
        pcl::toROSMsg(*env_not_power_cloud_, temp_msg);
        temp_msg.header = header;
        env_not_power_cloud_pub_.publish(temp_msg);
    }

    // 发布电力线点云
    if (powerline_cloud_pub_.getNumSubscribers() > 0 && !powerline_cloud->empty()) {
        sensor_msgs::PointCloud2 powerline_msg;
        pcl::toROSMsg(*powerline_cloud, powerline_msg);
        powerline_msg.header = header;
        powerline_cloud_pub_.publish(powerline_msg);
    }
    
    // 发布聚类后的电力线点云
    if (clustered_powerline_cloud_pub_.getNumSubscribers() > 0 && !clustered_cloud->empty()) {
        sensor_msgs::PointCloud2 clustered_msg;
        pcl::toROSMsg(*clustered_cloud, clustered_msg);
        clustered_msg.header = header;
        clustered_powerline_cloud_pub_.publish(clustered_msg);
    }

    if(fine_extractor_cloud_pub_.getNumSubscribers()>0 && !fine_extract_cloud_->empty()){
        sensor_msgs::PointCloud2 fine_extractor_msg;
        pcl::toROSMsg(*fine_extract_cloud_, fine_extractor_msg);
        fine_extractor_msg.header = header;
        fine_extractor_cloud_pub_.publish(fine_extractor_msg);
    }
    

}