/*
 * head_pose_sample.h
 *
 *  Created on: May 11, 2012
 *      Author: Matthias Dantone
 */

#ifndef HEAD_POSE_SAMPLE_H_
#define HEAD_POSE_SAMPLE_H_

#include "opencv2/core/core.hpp"
#include "image_sample.hpp"
#include "tree_node.hpp"

struct HeadPoseEstimatorOption {

  HeadPoseEstimatorOption() :
      num_head_pose_labels(5), step_size(4), min_forground_probability(0.5) {
  };
  int num_head_pose_labels;
  int step_size;
  float min_forground_probability;
};

class HeadPoseLeaf;
class HeadPoseSample {
public:

  typedef ThresholdSplit<SimplePatchFeature> Split;
  typedef HeadPoseLeaf Leaf;

  HeadPoseSample(const ImageSample* image_, const cv::Rect roi_,
      cv::Rect rect_, int label_) :
      image(image_), roi(roi_), rect(rect_), label(label_) {

    //if the label is smaller then 0 then neg example.
    isPos = (label >= 0);
  };

  HeadPoseSample(const ImageSample* image_, cv::Rect rect_) :
      image(image_), rect(rect_), label(-1) {
  };

  HeadPoseSample() {
  };

  const ImageSample* image;
  bool isPos;
  cv::Rect roi;
  cv::Rect rect;
  int label;
  void show();

  int evalTest(const Split& test) const;
  bool eval(const Split& test) const;

  static bool generateSplit(const std::vector<HeadPoseSample*>& data,
      boost::mt19937* rng, ForestParam fp, Split& split, float split_mode,
      int depth);

  static double evalSplit(const std::vector<HeadPoseSample*>& setA,
      const std::vector<HeadPoseSample*>& setB,
      const std::vector<float>& poppClasses, float splitMode, int depth);

  static void makeLeaf(HeadPoseLeaf& leaf,
      const std::vector<HeadPoseSample*>& set,
      const std::vector<float>& poppClasses,
      int leaf_id = 0);

  static void calcWeightClasses(std::vector<float>& poppClasses,
      const std::vector<HeadPoseSample*>& set);

  static double entropie(const std::vector<HeadPoseSample*>& set);

  static double gain(const std::vector<HeadPoseSample*>& set,
      int* num_pos_elements);

  static double gain2(const std::vector<HeadPoseSample*>& set,
      int* num_pos_elements);

  static double entropie_pose(const std::vector<HeadPoseSample*>& set);

  virtual ~HeadPoseSample() {
  };
};

class HeadPoseLeaf {
public:
  HeadPoseLeaf() {
    depth = -1;
  };

  //number of patches reached the leaf
  int nSamples;

  std::vector<int> hist_labels;

  float forgound;
  int depth;

  friend class boost::serialization::access;
  template<class Archive>
  void serialize(Archive & ar, const unsigned int version) {
    ar & nSamples;
    ar & forgound;
    ar & depth;
    ar & hist_labels;
  }
};

#endif /* HEAD_POSE_SAMPLE_H_ */
