/*
 * PatchFeatures.h
 *
 *  Created on: May 3, 2011
 *      Author: Matthias Dantone
 */

#ifndef PATCHFEATURES_H_
#define PATCHFEATURES_H_

#include <iostream>
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <boost/serialization/utility.hpp>
#include "opencv_serialization.hpp"
#include "feature_channel_factory.hpp"
#include <boost/random.hpp>

struct SimplePatchFeature {
  int featureChannel;
  cv::Rect_<int> rectA;
  cv::Rect_<int> rectB;

  void print() {
    std::cout << "FC: " << featureChannel << std::endl;
    std::cout << "Rect A " << rectA.x << ", " << rectA.y << ", " << rectA.width << " " << rectA.height << std::endl;
    std::cout << "Rect B " << rectB.x << ", " << rectB.y << ", " << rectB.width << " " << rectB.height << std::endl;
  }

  void generate(int patch_size, boost::mt19937* rng,
      int num_feature_channels = 0,
      float max_sub_patch_ratio = 1.0) {
    if (num_feature_channels > 1) {
      boost::uniform_int<> dist_feat(0, num_feature_channels - 1);
      boost::variate_generator<boost::mt19937&, boost::uniform_int<> > rand_feat(*rng, dist_feat);
      featureChannel = rand_feat();
    } else {
      featureChannel = 0;
    }

    int size = static_cast<int>(patch_size * max_sub_patch_ratio);

    boost::uniform_int<> dist_size(1, (size - 1) * 0.75);
    boost::variate_generator<boost::mt19937&, boost::uniform_int<> > rand_size(*rng, dist_size);
    rectA.width = rand_size();
    rectA.height = rand_size();
    rectB.width = rand_size();
    rectB.height = rand_size();

    boost::uniform_int<> dist_x(0, size - rectA.width - 1);
    boost::variate_generator<boost::mt19937&, boost::uniform_int<> > rand_x(*rng, dist_x);
    rectA.x = rand_x();

    boost::uniform_int<> dist_y(0, size - rectA.height - 1);
    boost::variate_generator<boost::mt19937&, boost::uniform_int<> > rand_y(*rng, dist_y);
    rectA.y = rand_y();

    boost::uniform_int<> dist_x_b(0, size - rectB.width - 1);
    boost::variate_generator<boost::mt19937&, boost::uniform_int<> > rand_x_b(*rng, dist_x_b);
    rectB.x = rand_x_b();

    boost::uniform_int<> dist_y_b(0, size - rectB.height - 1);
    boost::variate_generator<boost::mt19937&, boost::uniform_int<> > rand_y_b(*rng, dist_y_b);
    rectB.y = rand_y_b();

    assert( rectA.x >= 0 and rectB.x>=0 and rectA.y >= 0 and rectB.y>=0);
    assert( rectA.x+rectA.width < patch_size and rectA.y+rectA.height < patch_size);
    assert( rectB.x+rectB.width < patch_size and rectB.y+rectB.height < patch_size);
  }

  friend class boost::serialization::access;
  template<class Archive>
  void serialize(Archive & ar, const unsigned int version) {
    ar & featureChannel;
    ar & rectA;
    ar & rectB;
  }
};

struct SimplePixelFeature {
  int featureChannel;
  cv::Point_<int> point_a;
  cv::Point_<int> point_b;

  void print() {
    std::cout << "FC: " << featureChannel << std::endl;
    std::cout << "Point A " << point_a.x << ", " << point_a.y << std::endl;
    std::cout << "Point B " << point_b.x << ", " << point_b.y << std::endl;
  }
  void generate(int patch_size, boost::mt19937* rng,
      int num_feature_channels = 0, float max_sub_patch_ratio = 1.0) {

    if (num_feature_channels > 1) {
      boost::uniform_int<> dist_feat(0, num_feature_channels - 1);
      boost::variate_generator<boost::mt19937&, boost::uniform_int<> > rand_feat(*rng, dist_feat);
      featureChannel = rand_feat();
    } else {
      featureChannel = 0;
    }

    boost::uniform_int<> dist_size(1, patch_size);
    boost::variate_generator<boost::mt19937&, boost::uniform_int<> > rand_size(*rng, dist_size);

    point_a.x = rand_size();
    point_a.y = rand_size();
    point_b.x = rand_size();
    point_b.y = rand_size();
  }

  friend class boost::serialization::access;
  template<class Archive>
  void serialize(Archive & ar, const unsigned int version) {
    ar & featureChannel;
    ar & point_a;
    ar & point_b;
  }
};

template<typename Feature>
class ThresholdSplit {
public:
  ThresholdSplit() {
    margin = 0;
  }
  Feature feature;

  double info;
  double gain;
  double oob;
  int threshold;
  int margin;
  int depth;
  int num_thresholds;
  float split_mode;
  void generate(int patch_size, boost::mt19937* rng, int num_feature_channels = 0) {
    feature.generate(patch_size, rng, num_feature_channels);
    margin = 0;
    num_thresholds = 25;
  }

  void print() {
    feature.print();
    std::cout << " " << threshold << std::endl;
  }

  friend class boost::serialization::access;
  template<class Archive>
  void serialize(Archive & ar, const unsigned int version) {
    ar & feature;
    ar & info;
    ar & gain;
    ar & threshold;

  }
};

class ImageSample {
public:
  ImageSample() {
  }
  ;
  ImageSample(const cv::Mat img, std::vector<int> features, FeatureChannelFactory& fcf, bool useIntegral);
  ImageSample(const cv::Mat img, std::vector<int> features, bool useIntegral);

  int evalTest(const SimplePatchFeature& test, const cv::Rect rect) const;
  int evalTest(const SimplePixelFeature& test, const cv::Rect rect) const;

  // Extract features from image
  void extractFeatureChannels(const cv::Mat& img, std::vector<cv::Mat>& vImg, std::vector<int> features, bool useIntegral,
      FeatureChannelFactory& fcf) const;

  void getSubPatches(cv::Rect rect, std::vector<cv::Mat>& tmpPatches);

  virtual ~ImageSample();

  int width() const {
    return featureChannels[0].cols;
  }

  int height() const {
    return featureChannels[0].rows;
  }

  void show() const {
    cv::imshow("Image Sample", featureChannels[0]);
    cv::waitKey(0);
  }
  std::vector<cv::Mat> featureChannels;

private:
  bool useIntegral;

};

#endif /* PATCHFEATURES_H_ */
