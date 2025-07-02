#include "power_line_fine_extraction.h"
#include <pcl/filters/extract_indices.h>
#include <random>
#include <chrono>  // 添加 chrono 头文件
#include <omp.h>  // 添加 OpenMP 头文件

// thread_local std::mt19937 gen(std::random_device{}());

PowerLineFineExtractor::PowerLineFineExtractor(ros::NodeHandle& nh) {
    // 从ROS参数服务器读取参数（通过launch文件从YAML文件加载）
    nh.param("fineextract/line_distance_threshold", line_distance_threshold_, 0.1);
    nh.param("fineextract/line_min_points", line_min_points_, 100);
    nh.param("fineextract/max_lines", max_lines_, 10);
    nh.param("fineextract/vertical_slice_width", vertical_slice_width_, 0.5);
    nh.param("fineextract/parabola_distance_threshold", parabola_distance_threshold_, 0.1);
    nh.param("fineextract/parabola_min_points", parabola_min_points_, 50);
    nh.param("fineextract/power_line_distance_threshold", power_line_distance_threshold_, 0.1);
    nh.param("fineextract/angle_threshold", angle_threshold_, 0.99);            // 默认 0.99（约 8 度）
    nh.param("fineextract/distance_threshold", distance_threshold_, 1.0);       // 默认 1.0 米
    nh.param("fineextract/min_parallel_lines", min_parallel_lines_, 2);         // 默认 2
     // 新增参数加载
    nh.param("fineextract/min_line_length", min_line_length_, 0.3); // 默认 5.0 米
    nh.param("fineextract/dbscan_epsilon", dbscan_epsilon_, 0.5);       // 默认 0.5 米
    nh.param("fineextract/dbscan_min_points", dbscan_min_points_, 10);  // 默认 10
    nh.param("fineextract/cluster_min_points", cluster_min_points_, 50); // 默认 50

    ROS_INFO("PowerLineFineExtractor 初始化参数如下：");
    ROS_INFO("直线距离阈值: %f", line_distance_threshold_);
    ROS_INFO("直线最小点数: %d", line_min_points_);
    ROS_INFO("最大直线数量: %d", max_lines_);
    ROS_INFO("垂直切片宽度: %f", vertical_slice_width_);
    ROS_INFO("抛物线距离阈值: %f", parabola_distance_threshold_);
    ROS_INFO("抛物线最小点数: %d", parabola_min_points_);
    ROS_INFO("电力线距离阈值: %f", power_line_distance_threshold_);
    ROS_INFO("方向夹角阈值: %f", angle_threshold_);
    ROS_INFO("直线间距离阈值: %f", distance_threshold_);
    ROS_INFO("最小平行直线数量: %d", min_parallel_lines_);
    ROS_INFO("最小直线长度: %f", min_line_length_); // 输出新参数
    // 输出新参数
    ROS_INFO("DBSCAN 邻域半径: %f", dbscan_epsilon_);
    ROS_INFO("DBSCAN 最小邻域点数: %d", dbscan_min_points_);
    ROS_INFO("聚类簇最小点数: %d", cluster_min_points_);
    
}

void PowerLineFineExtractor::extractPowerLines(const pcl::PointCloud<pcl::PointXYZI>::Ptr& input_cloud,
    pcl::PointCloud<pcl::PointXYZI>::Ptr& output_cloud) {

    // 记录开始时间
    auto start = std::chrono::high_resolution_clock::now();
    
    // 计算PCA
    Eigen::Matrix3f eigenvectors;
    Eigen::Vector3f eigenvalues, centroid;
    computePCA(input_cloud, eigenvectors, eigenvalues, centroid);

    // 投影到水平平面
    pcl::PointCloud<pcl::PointXY>::Ptr projected_cloud(new pcl::PointCloud<pcl::PointXY>);
    projectToPlane(input_cloud, eigenvectors, centroid, projected_cloud);

    // 检测直线
    std::vector<Eigen::VectorXf> line_models; // 类型改为 Eigen::VectorXf
    detectLinesRANSAC(projected_cloud, line_models);

     // 新增：过滤平行直线
     std::vector<Eigen::VectorXf> filtered_line_models;
     filterParallelLines(line_models, filtered_line_models, angle_threshold_, distance_threshold_, min_parallel_lines_);

    // 为每条直线提取并拟合抛物线
    output_cloud->clear();
    for (const auto& line_model : filtered_line_models) {
    pcl::PointCloud<pcl::PointXYZI>::Ptr slice_cloud(new pcl::PointCloud<pcl::PointXYZI>);
    extractVerticalSlice(input_cloud, line_model, eigenvectors, vertical_slice_width_, slice_cloud,centroid);

    std::vector<std::vector<int>> inlier_indices;
    Eigen::Vector3f line_dir(line_model[2], line_model[3], 0); // 从 Eigen::VectorXf 中提取方向
    Eigen::Vector3f vertical_dir = eigenvectors.col(2);
    fitParabolasRANSAC(slice_cloud, line_dir, vertical_dir, inlier_indices);

    for (const auto& indices : inlier_indices) {
    for (int idx : indices) {
    output_cloud->push_back(slice_cloud->points[idx]);
    }
    }
    }

    output_cloud->width = output_cloud->size();
    output_cloud->height = 1;
    output_cloud->is_dense = true;


    // 新增：应用密度聚类并过滤小簇
    if (output_cloud->size() > 0) {
        pcl::search::KdTree<pcl::PointXYZI>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZI>());
        tree->setInputCloud(output_cloud);

        std::vector<pcl::PointIndices> cluster_indices;
        pcl::EuclideanClusterExtraction<pcl::PointXYZI> ec;
        ec.setClusterTolerance(dbscan_epsilon_);
        ec.setMinClusterSize(dbscan_min_points_);
        ec.setSearchMethod(tree);
        ec.setInputCloud(output_cloud);
        ec.extract(cluster_indices);

        pcl::PointCloud<pcl::PointXYZI>::Ptr filtered_cloud(new pcl::PointCloud<pcl::PointXYZI>());
        for (const auto& indices : cluster_indices) {
            if (indices.indices.size() >= static_cast<size_t>(cluster_min_points_)) {
                for (int idx : indices.indices) {
                    filtered_cloud->push_back(output_cloud->points[idx]);
                }
            }
        }

        *output_cloud = *filtered_cloud;
        output_cloud->width = output_cloud->size();
        output_cloud->height = 1;
        output_cloud->is_dense = true;
    }

    // 记录结束时间并计算耗时
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    ROS_INFO("精提取电力线 执行时间: %f 秒", duration.count());
}


void PowerLineFineExtractor::computePCA(const pcl::PointCloud<pcl::PointXYZI>::Ptr& cloud,
                                        Eigen::Matrix3f& eigenvectors,
                                        Eigen::Vector3f& eigenvalues,
                                        Eigen::Vector3f& centroid) {
    pcl::PCA<pcl::PointXYZI> pca;
    pca.setInputCloud(cloud);
    eigenvectors = pca.getEigenVectors();
    eigenvalues = pca.getEigenValues();
    centroid = pca.getMean().head<3>();
    // 添加ROS_INFO
    // ROS_INFO("PCA计算完成：");
    // ROS_INFO("  输入点云点数: %zu", cloud->size());
    // ROS_INFO("  特征值: %f, %f, %f", eigenvalues[0], eigenvalues[1], eigenvalues[2]);
    // ROS_INFO("  质心: (%f, %f, %f)", centroid[0], centroid[1], centroid[2]);
}

void PowerLineFineExtractor::projectToPlane(const pcl::PointCloud<pcl::PointXYZI>::Ptr& cloud,
                                            const Eigen::Matrix3f& eigenvectors,
                                            const Eigen::Vector3f& centroid,
                                            pcl::PointCloud<pcl::PointXY>::Ptr& projected_cloud) {
    Eigen::Vector3f e1 = eigenvectors.col(0); // 最大方差
    Eigen::Vector3f e2 = eigenvectors.col(1); // 次大方差
    projected_cloud->clear();
    projected_cloud->resize(cloud->size());  // 预分配内存，避免线程竞争

    // #pragma omp parallel for
    // for (const auto& pt : *cloud) {
    for (size_t i = 0; i < cloud->size(); ++i) {
        const auto& pt = cloud->points[i];
        Eigen::Vector3f p(pt.x - centroid[0], pt.y - centroid[1], pt.z - centroid[2]);
        pcl::PointXY p2d;
        p2d.x = p.dot(e1);
        p2d.y = p.dot(e2);
        projected_cloud->points[i] = p2d;  // 每个线程写入独立位置
    }
    // 添加ROS_INFO
    ROS_INFO("投影后2D点云点数: %zu", projected_cloud->size());
}

void PowerLineFineExtractor::detectLinesRANSAC(const pcl::PointCloud<pcl::PointXY>::Ptr& projected_cloud,
    std::vector<Eigen::VectorXf>& line_models) {

    pcl::PointCloud<pcl::PointXY>::Ptr cloud_copy(new pcl::PointCloud<pcl::PointXY>(*projected_cloud));
    std::random_device rd;
    std::mt19937 gen(rd());
    const int max_iterations = 1000; // RANSAC迭代次数

    while (cloud_copy->size() > static_cast<size_t>(line_min_points_) && line_models.size() < static_cast<size_t>(max_lines_)) {
        std::vector<int> best_inliers;
        Eigen::VectorXf best_coefficients(4); // [x0, y0, dx, dy]
        size_t max_inliers = 0;

        // RANSAC迭代
        for (int iter = 0; iter < max_iterations; ++iter) {
            // 随机选择两个点拟合直线
            std::uniform_int_distribution<> dis(0, cloud_copy->size() - 1);
            int idx1 = dis(gen);
            int idx2;
            do {
                idx2 = dis(gen);
            } while (idx2 == idx1);

            pcl::PointXY p1 = cloud_copy->points[idx1];
            pcl::PointXY p2 = cloud_copy->points[idx2];

            // 计算直线参数
            float dx = p2.x - p1.x;
            float dy = p2.y - p1.y;
            float length = std::sqrt(dx * dx + dy * dy);
            if (length < 1e-6) continue; // 避免除以零

            dx /= length; // 归一化方向向量
            dy /= length;

            Eigen::VectorXf coefficients(4);
            coefficients << p1.x, p1.y, dx, dy;

            // 计算内点
            std::vector<bool> is_inlier(cloud_copy->size(), false);
            // #pragma omp parallel for
            for (size_t i = 0; i < cloud_copy->size(); ++i) {
                pcl::PointXY pt = cloud_copy->points[i];
                // 点到直线的距离
                float distance = fabs((pt.x - p1.x) * dy - (pt.y - p1.y) * dx);
                if (distance < line_distance_threshold_) {
                    is_inlier[i] = true;
                }
            }
            std::vector<int> inliers;
            for (size_t i = 0; i < is_inlier.size(); ++i) {
                if (is_inlier[i]) inliers.push_back(i);
            }

            if (inliers.size() > max_inliers) {
                max_inliers = inliers.size();
                best_inliers = inliers;
                best_coefficients = coefficients;
            }
        }

        

        if (max_inliers < static_cast<size_t>(line_min_points_)) break;

        

        line_models.push_back(best_coefficients);

        // 移除内点
        pcl::ExtractIndices<pcl::PointXY> extract;
        extract.setInputCloud(cloud_copy);
        extract.setIndices(boost::make_shared<std::vector<int>>(best_inliers));
        extract.setNegative(true);
        extract.filter(*cloud_copy);
    }
    // 函数末尾输出总结
    ROS_INFO("直线检测完成，共检测到 %zu 条直线", line_models.size());
    
}


void PowerLineFineExtractor::extractVerticalSlice(const pcl::PointCloud<pcl::PointXYZI>::Ptr& cloud,
    const Eigen::VectorXf& line_model,
    const Eigen::Matrix3f& eigenvectors,
    double width,
    pcl::PointCloud<pcl::PointXYZI>::Ptr& slice_cloud,
    const Eigen::Vector3f& centroid) { // 新增centroid参数
    Eigen::Vector3f e1 = eigenvectors.col(0);
    Eigen::Vector3f e2 = eigenvectors.col(1);
    Eigen::Vector3f e3 = eigenvectors.col(2);

    // 反投影到3D参考点
    Eigen::Vector3f p0_3d = centroid + line_model[0] * e1 + line_model[1] * e2;

    // 3D方向
    Eigen::Vector3f line_dir_3d = line_model[2] * e1 + line_model[3] * e2;

    // 平面法向量
    Eigen::Vector3f vertical_dir = e3;
    Eigen::Vector3f plane_normal = line_dir_3d.cross(vertical_dir).normalized();

    // 调试输出
    // ROS_INFO("3D参考点 p0_3d: (%f, %f, %f)", p0_3d[0], p0_3d[1], p0_3d[2]);
    // ROS_INFO("直线参数: x0=%f, y0=%f, dx=%f, dy=%f", line_model[0], line_model[1], line_model[2], line_model[3]);
    // ROS_INFO("剖面宽度: %f", width);

    //最初的方法
    slice_cloud->clear();
    for (const auto& pt : *cloud) {
        Eigen::Vector3f p(pt.x, pt.y, pt.z);
        double distance = fabs((p - p0_3d).dot(plane_normal));
        if (distance < width / 2.0) {
            slice_cloud->push_back(pt);
        }
    }
    //并行计算
    // std::vector<int> indices;
    // #pragma omp parallel
    // {
    //     std::vector<int> local_indices; // 每个线程的私有索引容器
    //     #pragma omp for nowait
    //     for (size_t i = 0; i < cloud->size(); ++i) {
    //         const auto& pt = cloud->points[i];
    //         Eigen::Vector3f p(pt.x, pt.y, pt.z);
    //         double distance = fabs((p - p0_3d).dot(plane_normal));
    //         if (distance < width / 2.0) {
    //             local_indices.push_back(i);
    //         }
    //     }
    //     #pragma omp critical
    //     {
    //         indices.insert(indices.end(), local_indices.begin(), local_indices.end());
    //     }
    // }
    // pcl::copyPointCloud(*cloud, indices, *slice_cloud);




    ROS_INFO("提取的垂直剖面点云点数: %zu", slice_cloud->size());
}



void PowerLineFineExtractor::fitParabolasRANSAC(const pcl::PointCloud<pcl::PointXYZI>::Ptr& slice_cloud,
                                                const Eigen::Vector3f& line_dir,
                                                const Eigen::Vector3f& vertical_dir,
                                                std::vector<std::vector<int>>& inlier_indices) {
    pcl::PointCloud<pcl::PointXYZI>::Ptr remaining_cloud(new pcl::PointCloud<pcl::PointXYZI>(*slice_cloud));
    std::random_device rd;
    std::mt19937 gen(rd());

    while (remaining_cloud->size() > static_cast<size_t>(parabola_min_points_)) {
        std::vector<int> best_inliers;
        Eigen::Vector3f best_coeffs; // a, b, c for z = a*x^2 + b*x + c
        size_t max_inliers = 0;
        const int max_iterations = 1000;

        for (int iter = 0; iter < max_iterations; ++iter) {
            // 随机选择3个点
            std::vector<int> sample_indices;
            std::uniform_int_distribution<> dis(0, remaining_cloud->size() - 1);
            while (sample_indices.size() < 3) {
                int idx = dis(gen);
                if (std::find(sample_indices.begin(), sample_indices.end(), idx) == sample_indices.end()) {
                    sample_indices.push_back(idx);
                }
            }

            // 计算平面坐标中的x, z
            std::vector<double> x(3), z(3);
            for (int i = 0; i < 3; ++i) {
                Eigen::Vector3f p(remaining_cloud->points[sample_indices[i]].x,
                                  remaining_cloud->points[sample_indices[i]].y,
                                  remaining_cloud->points[sample_indices[i]].z);
                x[i] = p.dot(line_dir.normalized());
                z[i] = p.dot(vertical_dir);
            }

            // 求解抛物线：z = a*x^2 + b*x + c
            Eigen::Matrix3d A;
            Eigen::Vector3d B;
            for (int i = 0; i < 3; ++i) {
                A(i, 0) = x[i] * x[i];
                A(i, 1) = x[i];
                A(i, 2) = 1.0;
                B(i) = z[i];
            }
            Eigen::Vector3d coeffs = A.colPivHouseholderQr().solve(B);

            // 统计内点
            std::vector<int> inliers;
            for (size_t j = 0; j < remaining_cloud->size(); ++j) {
                Eigen::Vector3f p(remaining_cloud->points[j].x,
                                  remaining_cloud->points[j].y,
                                  remaining_cloud->points[j].z);
                double xj = p.dot(line_dir.normalized());
                double zj = p.dot(vertical_dir);
                double z_pred = coeffs[0] * xj * xj + coeffs[1] * xj + coeffs[2];
                if (fabs(zj - z_pred) < parabola_distance_threshold_) {
                    inliers.push_back(j);
                }
            }

            if (inliers.size() > max_inliers) {
                max_inliers = inliers.size();
                best_inliers = inliers;
                best_coeffs = coeffs.cast<float>(); // 显式转换为 float 类型
            }
        }

        if (max_inliers < static_cast<size_t>(parabola_min_points_)) break;

        inlier_indices.push_back(best_inliers);

        // 移除内点
        pcl::ExtractIndices<pcl::PointXYZI> extract;
        extract.setInputCloud(remaining_cloud);
        extract.setIndices(boost::make_shared<std::vector<int>>(best_inliers));
        extract.setNegative(true);
        extract.filter(*remaining_cloud);
    }
      // 函数末尾输出总结
      ROS_INFO("抛物线拟合完成，共拟合 %zu 条抛物线", inlier_indices.size());
}



void PowerLineFineExtractor::filterParallelLines(const std::vector<Eigen::VectorXf>& line_models,
    std::vector<Eigen::VectorXf>& filtered_line_models,
    double angle_threshold, double distance_threshold, int min_parallel_lines) {
    filtered_line_models.clear();

    for (size_t i = 0; i < line_models.size(); ++i) {
    const Eigen::VectorXf& line_i = line_models[i];
    float dx_i = line_i[2], dy_i = line_i[3];  // 方向向量
    float x0_i = line_i[0], y0_i = line_i[1];  // 直线上的点
    int parallel_count = 0;

    for (size_t j = 0; j < line_models.size(); ++j) {
    if (i == j) continue;  // 跳过自身

    const Eigen::VectorXf& line_j = line_models[j];
    float dx_j = line_j[2], dy_j = line_j[3];
    float x0_j = line_j[0], y0_j = line_j[1];

    // 计算方向向量的点积
    float dot = fabs(dx_i * dx_j + dy_i * dy_j);
    if (dot < angle_threshold) continue;  // 方向不相似

    // 计算点到直线的距离
    float dist = fabs((x0_i - x0_j) * dy_j - (y0_i - y0_j) * dx_j);
    if (dist < distance_threshold) {
    parallel_count++;
    }
    }

    // 如果平行直线数量满足条件，保留该直线
    if (parallel_count >= min_parallel_lines) {
    filtered_line_models.push_back(line_i);
    }
    }

    ROS_INFO("过滤后保留 %zu 条直线（平行直线数量 >= %d）", filtered_line_models.size(), min_parallel_lines);
}