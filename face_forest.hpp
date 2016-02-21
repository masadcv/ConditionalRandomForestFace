/*
 * conditional_rf.hpp
 *
 *  Created on: Jul 24, 2012
 *      Author: Matthias Dantone
 */

#ifndef CONDITIONAL_RF_HPP_
#define CONDITIONAL_RF_HPP_

#include "forest.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/objdetect/objdetect.hpp"
#include "thread_pool.hpp"
#include "head_pose_sample.hpp"
#include "multi_part_sample.hpp"

struct Face {
  float headpose;
  cv::Rect bbox;
  std::vector<cv::Point> ffd_cordinates;
};

struct FaceDetectionOption {
  FaceDetectionOption() :
    min_feature_size(30), min_neighbors(1), search_scale_factor(1.3f) {
  };

  //options for face detection
  int min_feature_size;
  int min_neighbors;
  float search_scale_factor;
  std::string path_face_cascade;

};

struct FaceForestOptions {

  FaceDetectionOption face_detection_option;
  HeadPoseEstimatorOption pose_estimator_option;
  MultiPartEstimatorOption multi_part_option;

  ForestParam head_pose_forest_param;
  ForestParam mp_forest_param;
  std::vector<std::string> mp_tree_paths;

};

class FaceForest {
public:

  static void detect_face(const cv::Mat& img,
      cv::CascadeClassifier& face_cascade, FaceDetectionOption option,
      std::vector<cv::Rect>& faces);

  static void estimate_head_pose(const ImageSample& img_sample,
      const cv::Rect& face_bbox, const Forest<HeadPoseSample>& forest,
      HeadPoseEstimatorOption option, float* head_pose, float* variance);

  static void estimate_ffd(const ImageSample& image_sample,
      const cv::Rect face_bbox, const Forest<MPSample>& forest,
      MultiPartEstimatorOption option, std::vector<cv::Point>& ffd_cordinates);

  static void show_results(const cv::Mat img, std::vector<Face>& faces,
      int wait_key = 0);

  FaceForest() :
    trees(0), num_trees(0), is_inizialized(false) {
  }
  ;

  FaceForest(FaceForestOptions option);

  void analize_image(cv::Mat img, std::vector<Face>& faces);

  void analize_face(const cv::Mat img, cv::Rect face_bbox, Face& face,
      bool normalize = true);

  virtual ~FaceForest() {
  };

private:

  void loading_all_trees(std::vector<std::string> urls);

  void get_paths_to_trees(std::string url, std::vector<std::string>& urls);
  bool load_face_cascade(std::string url) {
    if (!face_cascade.load(url)) {
      std::cout << "--(!)Error loading face cascade : " << url << std::endl;
      return false;
    };
    return true;
  }

  FaceForestOptions option_;
  Forest<HeadPoseSample> con_forest;
  Forest<MPSample> forest;
  std::vector<std::vector<Tree<MPSample>*> > trees;
  int num_trees;
  cv::CascadeClassifier face_cascade;
  FeatureChannelFactory fcf;

  bool is_inizialized;
};

#endif /* CONDITIONAL_RF_HPP_ */
