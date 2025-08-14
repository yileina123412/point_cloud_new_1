#include "power_line_coarse_extractor_s.h"
#include <Eigen/Dense>
#include <queue>

PowerLineCoarseExtractor::PowerLineCoarseExtractor(ros::NodeHandle& nh) : nh_(nh),extracted_cloud_(new pcl::PointCloud<pcl::PointXYZI>),
env_without_powerline_cloud_(new pcl::PointCloud<pcl::PointXYZI>){
    loadParameters(nh);
    normal_estimation_.setRadiusSearch(0.5);
    kdtree_.reset(new pcl::search::KdTree<pcl::PointXYZI>);
    pub_linearity_ = nh_.advertise<sensor_msgs::PointCloud2>("linearity_cloud", 1);
    pub_curvature_ = nh_.advertise<sensor_msgs::PointCloud2>("curvature_cloud", 1);
    pub_variance_ = nh_.advertise<sensor_msgs::PointCloud2>("variance_cloud", 1);

    pub_qualified_ = nh_.advertise<sensor_msgs::PointCloud2>("qualified_cloud", 1);
    pub_unqualified_ = nh_.advertise<sensor_msgs::PointCloud2>("unqualified_cloud", 1);
}

void PowerLineCoarseExtractor::loadParameters(ros::NodeHandle& nh) {
    // ros::NodeHandle nh("~");
    nh.param("power_line_coarse_extractor_s/linearity_threshold", linearity_threshold_, 0.7);
    nh.param("power_line_coarse_extractor_s/curvature_threshold", curvature_threshold_, 0.1);
    nh.param("power_line_coarse_extractor_s/planarity_threshold", planarity_threshold_, 0.1);
    nh.param("power_line_coarse_extractor_s/use_planarity", use_planarity_, false);
    nh.param("power_line_coarse_extractor_s/cluster_tolerance", cluster_tolerance_, 0.5);
    nh.param("power_line_coarse_extractor_s/min_cluster_size", min_cluster_size_, 10);
    // 新增参数
    nh.param("power_line_coarse_extractor_s/variance_threshold", variance_threshold_, 0.1);
    nh.param("power_line_coarse_extractor_s/search_radius", search_radius_, 0.5);
    // 新增参数
    nh.param("power_line_coarse_extractor_s/min_cluster_length", min_cluster_length_, 5.0); // 单位：米

    // 边沿检查参数
    nh.param("power_line_coarse_extractor_s/edge_check_radius", edge_check_radius_, 0.1);
    nh.param("power_line_coarse_extractor_s/max_unqualified_neighbors", max_unqualified_neighbors_, 3);
    
    
    ROS_INFO("粗提取_s参数加载完成");
    ROS_INFO("min_cluster_length: %.2f",min_cluster_length_);
}



void PowerLineCoarseExtractor::manualClustering(const pcl::PointCloud<pcl::PointXYZI>::Ptr& cloud,
                                          std::vector<pcl::PointIndices>& cluster_indices,
                                          double tolerance, int min_size) {
    if (cloud->empty()) return;

    std::vector<bool> processed(cloud->size(), false);
    kdtree_->setInputCloud(cloud);

    for (size_t i = 0; i < cloud->size(); ++i) {
        if (processed[i]) continue;

        std::queue<int> queue;
        queue.push(i);
        processed[i] = true;
        pcl::PointIndices cluster;
        cluster.indices.push_back(i);

        while (!queue.empty()) {
            int idx = queue.front();
            queue.pop();

            std::vector<int> neighbors;
            std::vector<float> distances;
            kdtree_->radiusSearch(idx, tolerance, neighbors, distances);

            for (int neighbor : neighbors) {
                if (!processed[neighbor]) {
                    processed[neighbor] = true;
                    queue.push(neighbor);
                    cluster.indices.push_back(neighbor);
                }
            }
        }

        if (cluster.indices.size() >= static_cast<size_t>(min_size)) {
            cluster_indices.push_back(cluster);
        }
    }
}

pcl::PointCloud<pcl::PointXYZI>::Ptr PowerLineCoarseExtractor::getExtractedCloud() const {
    return extracted_cloud_;
}
pcl::PointCloud<pcl::PointXYZI>::Ptr PowerLineCoarseExtractor::getEnvWithoutPowerCloud() const
{
    return env_without_powerline_cloud_;
}
// 新增逐点提取函数
void PowerLineCoarseExtractor::extractPowerLinesByPoints(const pcl::PointCloud<pcl::PointXYZI>::Ptr& input_cloud) {
    
    // 获取预处理后的点云
    auto cloud = input_cloud;
    if (cloud->empty()) {
        ROS_WARN("Input point cloud is empty!");
        return;
    }
    
    kdtree_->setInputCloud(cloud);

    // 计算所有点的法向量
    pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
    normal_estimation_.setInputCloud(cloud);
    normal_estimation_.setSearchMethod(kdtree_);
    normal_estimation_.setRadiusSearch(search_radius_);
    normal_estimation_.compute(*normals);
    // env_without_powerline_cloud_->clear();
    // 遍历每个点，筛选电力线点
    pcl::PointCloud<pcl::PointXYZI>::Ptr power_lines(new pcl::PointCloud<pcl::PointXYZI>);
    for (size_t i = 0; i < cloud->points.size(); ++i) {
        if (isPowerLinePoint(cloud, normals, i)) {
            power_lines->points.push_back(cloud->points[i]);
        }
        else
        {
            env_without_powerline_cloud_->points.push_back(cloud->points[i]);
        }
    }
    power_lines->width = power_lines->points.size();
    power_lines->height = 1;
    power_lines->is_dense = true;

    // 在聚类前添加边沿过滤
    filterEdgePoints(power_lines, cloud, normals);

    // 手动聚类
    std::vector<pcl::PointIndices> cluster_indices;
    manualClustering(power_lines, cluster_indices, cluster_tolerance_, min_cluster_size_);

    // 滤除较短的簇
    filterShortClusters(power_lines, cluster_indices, min_cluster_length_);

    // 提取聚类结果
    extracted_cloud_->clear();
    for (const auto& indices : cluster_indices) {
        pcl::PointCloud<pcl::PointXYZI>::Ptr cluster(new pcl::PointCloud<pcl::PointXYZI>);
        for (int idx : indices.indices) {
            cluster->points.push_back(power_lines->points[idx]);
        }
        cluster->width = cluster->points.size();
        cluster->height = 1;
        cluster->is_dense = true;
        *extracted_cloud_ += *cluster;
    }
}


// 新增辅助函数：判断单个点是否为电力线点
bool PowerLineCoarseExtractor::isPowerLinePoint(const pcl::PointCloud<pcl::PointXYZI>::Ptr& cloud,
                                          const pcl::PointCloud<pcl::Normal>::Ptr& normals,
                                          int index) {
    // 获取邻域点
    std::vector<int> neighbors;
    std::vector<float> distances;
    kdtree_->radiusSearch(index, search_radius_, neighbors, distances);

    if (neighbors.size() < 3) return false; // 邻域点数不足

    // 提取邻域点云
    pcl::PointCloud<pcl::PointXYZI>::Ptr local_cloud(new pcl::PointCloud<pcl::PointXYZI>);
    for (int idx : neighbors) {
        local_cloud->points.push_back(cloud->points[idx]);
    }

    // 手动计算 PCA
    Eigen::Vector4f centroid;
    pcl::compute3DCentroid(*local_cloud, centroid);
    Eigen::Matrix3f covariance;
    pcl::computeCovarianceMatrix(*local_cloud, centroid, covariance);
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> eigen_solver(covariance);
    Eigen::Vector3f eigenvalues = eigen_solver.eigenvalues();
    std::sort(eigenvalues.data(), eigenvalues.data() + 3, std::greater<float>());

    if (eigenvalues[0] < 1e-6) return false; // 避免除以零

    float linearity = (eigenvalues[0] - eigenvalues[1]) / eigenvalues[0];
    float curvature = (eigenvalues[2] + eigenvalues[1])/ (eigenvalues[0] + eigenvalues[1] + eigenvalues[2]);

    // 计算法向量一致性
    Eigen::Vector3f mean_normal(0, 0, 0);
    for (int idx : neighbors) {
        mean_normal += Eigen::Vector3f(normals->points[idx].normal_x,
                                       normals->points[idx].normal_y,
                                       normals->points[idx].normal_z);
    }
    mean_normal /= neighbors.size();
    float variance = 0;
    for (int idx : neighbors) {
        Eigen::Vector3f diff = Eigen::Vector3f(normals->points[idx].normal_x,
                                               normals->points[idx].normal_y,
                                               normals->points[idx].normal_z) - mean_normal;
        variance += diff.squaredNorm();
    }
    variance /= neighbors.size();

    // // 原有筛选条件
    // bool basic_criteria = linearity > linearity_threshold_ && curvature < curvature_threshold_;
    // if (!basic_criteria) return false;
    // // 新增垂直方向孤立性检查
    // Eigen::Vector3f main_direction = eigen_solver.eigenvectors().col(2); // 主方向是最大特征值对应的特征向量
    // return checkPerpendicularIsolation(cloud, normals, index, main_direction);

    // 筛选条件
    // return linearity > linearity_threshold_ && curvature < curvature_threshold_ && variance > variance_threshold_;
    return linearity > linearity_threshold_ && curvature < curvature_threshold_;

}

void PowerLineCoarseExtractor::filterShortClusters(const pcl::PointCloud<pcl::PointXYZI>::Ptr& cloud,
    std::vector<pcl::PointIndices>& cluster_indices,
    double min_length) {
std::vector<pcl::PointIndices> filtered_indices;
for (const auto& cluster : cluster_indices) {
if (cluster.indices.size() < 2) continue; // 跳过点数不足的簇

// 计算簇的边界框
pcl::PointXYZI min_pt, max_pt;
min_pt.x = min_pt.y = min_pt.z = std::numeric_limits<float>::max();
max_pt.x = max_pt.y = max_pt.z = -std::numeric_limits<float>::max();
for (int idx : cluster.indices) {
const auto& point = cloud->points[idx];
min_pt.x = std::min(min_pt.x, point.x);
min_pt.y = std::min(min_pt.y, point.y);
min_pt.z = std::min(min_pt.z, point.z);
max_pt.x = std::max(max_pt.x, point.x);
max_pt.y = std::max(max_pt.y, point.y);
max_pt.z = std::max(max_pt.z, point.z);
}
// 计算边界框对角线长度
double length = std::sqrt(std::pow(max_pt.x - min_pt.x, 2) +
std::pow(max_pt.y - min_pt.y, 2) +
std::pow(max_pt.z - min_pt.z, 2));
if (length >= min_length) {
filtered_indices.push_back(cluster);
}
}
cluster_indices = filtered_indices;
}


//  可视化函数实现


void PowerLineCoarseExtractor::visualizeParameters(const std::unique_ptr<PointCloudPreprocessor>& preprocessor_ptr) {
    auto cloud = preprocessor_ptr->getProcessedCloud();
    if (cloud->empty()) {
        ROS_WARN("Input point cloud is empty!");
        return;
    }
    kdtree_->setInputCloud(cloud);

    pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
    normal_estimation_.setInputCloud(cloud);
    normal_estimation_.setSearchMethod(kdtree_);
    normal_estimation_.setRadiusSearch(search_radius_);
    normal_estimation_.compute(*normals);

    std::vector<float> linearity_values(cloud->size(), 0.0f);
    std::vector<float> curvature_values(cloud->size(), 0.0f);
    std::vector<float> variance_values(cloud->size(), 0.0f);

    for (size_t i = 0; i < cloud->points.size(); ++i) {
        std::vector<int> neighbors;
        std::vector<float> distances;
        kdtree_->radiusSearch(i, search_radius_, neighbors, distances);
        if (neighbors.size() < 3) continue;

        pcl::PointCloud<pcl::PointXYZI>::Ptr local_cloud(new pcl::PointCloud<pcl::PointXYZI>);
        for (int idx : neighbors) local_cloud->points.push_back(cloud->points[idx]);

        Eigen::Vector4f centroid;
        pcl::compute3DCentroid(*local_cloud, centroid);
        Eigen::Matrix3f covariance;
        pcl::computeCovarianceMatrix(*local_cloud, centroid, covariance);
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> eigen_solver(covariance);
        Eigen::Vector3f eigenvalues = eigen_solver.eigenvalues();
        std::sort(eigenvalues.data(), eigenvalues.data() + 3, std::greater<float>());
        if (eigenvalues[0] < 1e-6) continue;

        linearity_values[i] = (eigenvalues[0] - eigenvalues[1]) / eigenvalues[0];
        curvature_values[i] = (eigenvalues[2]+eigenvalues[1]) / (eigenvalues[0] + eigenvalues[1] + eigenvalues[2]);

        Eigen::Vector3f mean_normal(0, 0, 0);
        for (int idx : neighbors) {
            mean_normal += Eigen::Vector3f(normals->points[idx].normal_x, normals->points[idx].normal_y, normals->points[idx].normal_z);
        }
        mean_normal /= neighbors.size();
        float variance = 0;
        for (int idx : neighbors) {
            Eigen::Vector3f diff = Eigen::Vector3f(normals->points[idx].normal_x, normals->points[idx].normal_y, normals->points[idx].normal_z) - mean_normal;
            variance += diff.squaredNorm();
        }
        variance /= neighbors.size();
        variance_values[i] = variance;
    }

    auto valueToColor = [](float value, float min_val, float max_val) -> uint32_t {
        float normalized = (value - min_val) / (max_val - min_val);
        normalized = std::max(0.0f, std::min(1.0f, normalized)); // 限制在 0-1 范围内
        uint8_t r = static_cast<uint8_t>(255 * normalized);      // 红色分量
        uint8_t g = 0;                                           // 绿色分量
        uint8_t b = static_cast<uint8_t>(255 * (1 - normalized)); // 蓝色分量
        return (r << 16) | (g << 8) | b;                         // 组合成 RGB 值
    };
    float min_curvature = *std::min_element(curvature_values.begin(), curvature_values.end());
    float max_curvature = *std::max_element(curvature_values.begin(), curvature_values.end());
    if (max_curvature <= min_curvature) max_curvature = min_curvature + 1e-6;
    ROS_INFO("Curvature range: min = %f, max = %f", min_curvature, max_curvature);

    std::vector<float> log_variance_values(cloud->size());
    for (size_t i = 0; i < variance_values.size(); ++i) log_variance_values[i] = std::log1p(variance_values[i]);
    float min_log_variance = *std::min_element(log_variance_values.begin(), log_variance_values.end());
    float max_log_variance = *std::max_element(log_variance_values.begin(), log_variance_values.end());
    if (max_log_variance <= min_log_variance) max_log_variance = min_log_variance + 1e-6;
    ROS_INFO("Log Variance range: min = %f, max = %f", min_log_variance, max_log_variance);

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr linearity_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr curvature_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr variance_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);

    for (size_t i = 0; i < cloud->points.size(); ++i) {
        pcl::PointXYZRGB point;
        point.x = cloud->points[i].x;
        point.y = cloud->points[i].y;
        point.z = cloud->points[i].z;
        
        uint32_t color = valueToColor(linearity_values[i], 0.0f, 1.0f);
        point.rgb = *reinterpret_cast<float*>(&color);
        linearity_cloud->points.push_back(point);

        color = valueToColor(curvature_values[i], min_curvature, max_curvature);
        point.rgb = *reinterpret_cast<float*>(&color);
        curvature_cloud->points.push_back(point);

        color = valueToColor(log_variance_values[i], min_log_variance, max_log_variance);
        point.rgb = *reinterpret_cast<float*>(&color);
        variance_cloud->points.push_back(point);
    }

    linearity_cloud->width = linearity_cloud->points.size();
    linearity_cloud->height = 1;
    curvature_cloud->width = curvature_cloud->points.size();
    curvature_cloud->height = 1;
    variance_cloud->width = variance_cloud->points.size();
    variance_cloud->height = 1;

    

    sensor_msgs::PointCloud2 output;
    pcl::toROSMsg(*linearity_cloud, output);
    output.header.frame_id = "map";
    output.header.stamp = ros::Time::now();
    pub_linearity_.publish(output);

    pcl::toROSMsg(*curvature_cloud, output);
    output.header.frame_id = "map";
    output.header.stamp = ros::Time::now();
    pub_curvature_.publish(output);

    pcl::toROSMsg(*variance_cloud, output);
    output.header.frame_id = "map";
    output.header.stamp = ros::Time::now();
    pub_variance_.publish(output);
}

void PowerLineCoarseExtractor::filterEdgePoints(const pcl::PointCloud<pcl::PointXYZI>::Ptr& input_cloud,
                                               const pcl::PointCloud<pcl::PointXYZI>::Ptr& original_cloud,
                                               const pcl::PointCloud<pcl::Normal>::Ptr& normals) {
    if (input_cloud->empty()) return;
    
    // 为原点云建立kdtree用于邻域搜索
    pcl::search::KdTree<pcl::PointXYZI>::Ptr original_kdtree(new pcl::search::KdTree<pcl::PointXYZI>);
    original_kdtree->setInputCloud(original_cloud);
    
    pcl::PointCloud<pcl::PointXYZI>::Ptr qualified_cloud(new pcl::PointCloud<pcl::PointXYZI>);
    pcl::PointCloud<pcl::PointXYZI>::Ptr unqualified_cloud(new pcl::PointCloud<pcl::PointXYZI>);
    pcl::PointCloud<pcl::PointXYZI>::Ptr final_power_lines(new pcl::PointCloud<pcl::PointXYZI>);
    
    for (size_t i = 0; i < input_cloud->points.size(); ++i) {
        // 在原点云中搜索邻域
        std::vector<int> neighbors;
        std::vector<float> distances;
        original_kdtree->radiusSearch(input_cloud->points[i], edge_check_radius_, neighbors, distances);
        
        // 统计邻域中不合格的点数量
        int unqualified_count = 0;
        for (int neighbor_idx : neighbors) {
            if (!isPowerLinePoint(original_cloud, normals, neighbor_idx)) {
                unqualified_count++;
            }
        }
        
        if (unqualified_count <= max_unqualified_neighbors_) {
            qualified_cloud->points.push_back(input_cloud->points[i]);
            final_power_lines->points.push_back(input_cloud->points[i]);
        } else {
            unqualified_cloud->points.push_back(input_cloud->points[i]);
        }
    }
    
    // 设置点云属性
    qualified_cloud->width = qualified_cloud->points.size();
    qualified_cloud->height = 1;
    qualified_cloud->is_dense = true;
    unqualified_cloud->width = unqualified_cloud->points.size();
    unqualified_cloud->height = 1;
    unqualified_cloud->is_dense = true;
    final_power_lines->width = final_power_lines->points.size();
    final_power_lines->height = 1;
    final_power_lines->is_dense = true;

    // 发布可视化点云
    sensor_msgs::PointCloud2 output;
    pcl::toROSMsg(*qualified_cloud, output);
    output.header.frame_id = "map";
    output.header.stamp = ros::Time::now();
    pub_qualified_.publish(output);

    pcl::toROSMsg(*unqualified_cloud, output);
    output.header.frame_id = "map";
    output.header.stamp = ros::Time::now();
    pub_unqualified_.publish(output);
    
    // 更新输入点云为过滤后的结果
    *input_cloud = *final_power_lines;
}