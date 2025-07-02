#ifndef POWER_LINE_FINE_EXTRACTION_H
#define POWER_LINE_FINE_EXTRACTION_H

#include <ros/ros.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/common/pca.h>
#include <pcl/common/common.h>  
#include <pcl/sample_consensus/ransac.h>
#include <pcl/sample_consensus/sac_model_line.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/search/kdtree.h>     
#include <Eigen/Dense>

class PowerLineFineExtractor {
public:
    // 构造函数：通过ROS节点句柄读取参数
    PowerLineFineExtractor(ros::NodeHandle& nh);

    // 主要处理函数
    void extractPowerLines(const pcl::PointCloud<pcl::PointXYZI>::Ptr& input_cloud,
                           pcl::PointCloud<pcl::PointXYZI>::Ptr& output_cloud);

private:
    // 从YAML文件通过ROS读取的参数
    double line_distance_threshold_;    // 直线内点的距离阈值
    int line_min_points_;               // 形成一条直线的最小点数
    int max_lines_;                     // 检测的最大直线数量
    double vertical_slice_width_;       // 垂直切片宽度
    double parabola_distance_threshold_; // 抛物线内点的距离阈值
    int parabola_min_points_;           // 拟合一条抛物线的最小点数
    double power_line_distance_threshold_; // 最终电力线点距离阈值

    double angle_threshold_;         // 新增：方向夹角阈值
    double distance_threshold_;      // 新增：直线间距离阈值
    int min_parallel_lines_;         // 新增：最小平行直线数量
    // 新增参数
    double min_line_length_; // 最小直线长度（单位：米）

    // 新增密度聚类参数
    double dbscan_epsilon_;        // DBSCAN 邻域半径
    int dbscan_min_points_;        // DBSCAN 最小邻域点数
    int cluster_min_points_;       // 聚类簇的最小点数


    // 辅助方法
    void computePCA(const pcl::PointCloud<pcl::PointXYZI>::Ptr& cloud,
                    Eigen::Matrix3f& eigenvectors,
                    Eigen::Vector3f& eigenvalues,
                    Eigen::Vector3f& centroid);

    void projectToPlane(const pcl::PointCloud<pcl::PointXYZI>::Ptr& cloud,
                        const Eigen::Matrix3f& eigenvectors,
                        const Eigen::Vector3f& centroid,
                        pcl::PointCloud<pcl::PointXY>::Ptr& projected_cloud);


    

    void detectLinesRANSAC(const pcl::PointCloud<pcl::PointXY>::Ptr& projected_cloud,
                        std::vector<Eigen::VectorXf>& line_models);

    void extractVerticalSlice(const pcl::PointCloud<pcl::PointXYZI>::Ptr& cloud,
        const Eigen::VectorXf& line_model,
        const Eigen::Matrix3f& eigenvectors,
        double width,
        pcl::PointCloud<pcl::PointXYZI>::Ptr& slice_cloud,
        const Eigen::Vector3f& centroid);
  

    void fitParabolasRANSAC(const pcl::PointCloud<pcl::PointXYZI>::Ptr& slice_cloud,
                            const Eigen::Vector3f& line_dir,
                            const Eigen::Vector3f& vertical_dir,
                            std::vector<std::vector<int>>& inlier_indices);
    // 新增方法声明
    void filterParallelLines(const std::vector<Eigen::VectorXf>& line_models,
        std::vector<Eigen::VectorXf>& filtered_line_models,
        double angle_threshold, double distance_threshold, int min_parallel_lines);
};

#endif // POWER_LINE_FINE_EXTRACTION_H
