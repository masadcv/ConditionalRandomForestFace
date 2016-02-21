/*
 * FFDSample.h
 *
 *  Created on: Sep 8, 2011
 *      Author: Matthias Dantone
 */

#ifndef MULTIPART_H_
#define MULTIPART_H_

#include "opencv2/core/core.hpp"
#include "image_sample.hpp"
#include "tree_node.hpp"
//#include "mean_shift.hpp"
//#include "cpp/random_forest/utils/face_utils.hpp"

struct MultiPartEstimatorOption {

  MultiPartEstimatorOption() :
      num_parts(10), step_size(3), min_samples(2), min_forground(0.5),
      min_pf(0.25), max_variance(25)
  {};
  int num_parts;
  int step_size;
  int min_samples;
  float min_forground;
  float min_pf;
  float max_variance;
};

class MPLeaf;
class MPSample {

public:

//    typedef ThresholdSplit<SimplePixelFeature> Split;
  typedef ThresholdSplit<SimplePatchFeature> Split;

  typedef MPLeaf Leaf;

  MPSample(const ImageSample* patch, cv::Rect rect, const cv::Rect roi,
      const std::vector<cv::Point> parts, float size, bool label,
      float lamda = 0.125);
  MPSample(const ImageSample* patch, cv::Rect rect, int n_points, float size);
  MPSample(const ImageSample* patch, cv::Rect rect);
  MPSample() :
      num_parts(0) {
  };

  float distToCenter;
  const ImageSample* image;

  std::vector<cv::Point_<int> > part_offsets;
  cv::Rect rect;
  cv::Rect roi;
  cv::Mat dist;

  float size;
  int num_parts;

  cv::Point_<int> patch_offset;
  void show() {

    cv::imshow("X", image->featureChannels[0](rect));
    cv::Mat face = image->featureChannels[0].clone();
    cv::rectangle(face, rect, cv::Scalar(255, 255, 255, 0));
    cv::rectangle(face, roi, cv::Scalar(255, 255, 255, 0));

    if (isPos) {

      int patch_size = (rect.height) / 2.0;
      for (int i = 0; i < (int) part_offsets.size(); i++) {
        int x = rect.x + patch_size + part_offsets[i].x;
        int y = rect.y + patch_size + part_offsets[i].y;
        cv::circle(face, cv::Point_<int>(x, y), 3, cv::Scalar(255, 255, 255, 0));
        std::cout << i << " " << dist.at<float>(0, i) << std::endl;
      }
      int x = rect.x + patch_size + patch_offset.x;
      int y = rect.y + patch_size + patch_offset.y;
      cv::circle(face, cv::Point_<int>(x, y), 3, cv::Scalar(0, 0, 0, 0));
    }
    cv::imshow("Y", face);
    cv::waitKey(0);
  };

  bool isPos;

  int evalTest(const Split& test) const;
  bool eval(const Split& test) const;

  static bool generateSplit(const std::vector<MPSample*>& data,
      boost::mt19937* rng, ForestParam fp, Split& split,
      float split_mode, int depth);

  static double evalSplit(const std::vector<MPSample*>& setA,
      const std::vector<MPSample*>& setB, const std::vector<float>& poppClasses,
      float splitMode, int depth);

  static void optimize(boost::mt19937* rng, const std::vector<MPSample*>& set,
      Split& split, float split_mode, int depth);

  inline static double entropie(const std::vector<MPSample*>& set);

  inline static double entropie_pose(const std::vector<MPSample*>& set);

  inline static double entropie_parts(const std::vector<MPSample*>& set);

  inline static double infoGain(const std::vector<MPSample*>& set);

  static void makeLeaf(MPLeaf& leaf, const std::vector<MPSample*>& set,
      const std::vector<float>& poppClasses, int leaf_id = 0);

  static void calcWeightClasses(std::vector<float>& poppClasses,
      const std::vector<MPSample*>& set);

  static double entropieFaceOrNot(const std::vector<MPSample*>& set);

  static double eval_oob(const std::vector<MPSample*>& data, Split& test) {
    return 0;
  }

  friend class boost::serialization::access;
  template<class Archive>
  void serialize(Archive & ar, const unsigned int version) {
    ar & isPos;
    if (isPos) {
      ar & patch_offset;
      ar & part_offsets;
      ar & dist;
      ar & rect;
      ar & distToCenter;
    }
  }
  virtual ~MPSample() {
  };

};

class MPLeaf {
public:
  MPLeaf() {
    depth = -1;
    save_all = false;
  };

  std::vector<float> maxDists;
  std::vector<float> lamda;

  //number of patches reached the leaf
  int nSamples;

  //vector of the means
  std::vector<cv::Point_<int> > parts_offset;

  // variance of the votes
  std::vector<float> variance;

  //probability of foreground per each point
  std::vector<float> pF;

  cv::Point_<int> patch_offset;

  //probability of face
  float forgound;
  int depth;

  bool save_all;
  //
  std::vector<cv::Point_<int> > offset_sum;
  std::vector<cv::Point_<int> > offset_sum_sq;
  std::vector<float> sum_pf;
  int sum_pos;
  int sum_all;

  friend class boost::serialization::access;
  template<class Archive>
  void serialize(Archive & ar, const unsigned int version) {
    ar & nSamples;
    ar & parts_offset;
    ar & variance;
    ar & pF;
    ar & forgound;
    ar & patch_offset;
    ar & save_all;
  }
};
#endif /* MULTIPART_H_ */
