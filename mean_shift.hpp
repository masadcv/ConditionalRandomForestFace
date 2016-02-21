/*
 * mean_shift.hpp
 *
 *  Created on: Jul 22, 2012
 *      Author: Matthias Dantone
 */

#ifndef MEAN_SHIFT_HPP_
#define MEAN_SHIFT_HPP_

#include "face_utils.hpp"
#include <vector>

struct MeanShiftOption {
  MeanShiftOption() :
      kernel_size(10), max_iterations(7), stopping_criteria(0.05) {
  };
  int kernel_size;
  int max_iterations;
  float stopping_criteria;
};

class MeanShift {
public:
  MeanShift() {
  };

  static float dist_l2(const cv::Point a, const cv::Point b) {
    return sqrt((a.x - b.x) * (a.x - b.x) + (a.y - b.y) * (a.y - b.y));
  }

  static void shift(const std::vector<Vote>& votes, cv::Point_<int>& result,
      MeanShiftOption& option) {
    shift(votes, result, option.max_iterations,
        option.kernel_size, option.stopping_criteria);
  }
  static void shift(const std::vector<Vote>& votes, cv::Point_<int>& result,
      int num_iterations, int kernel, float stopping_criteria) {
    bool covergenz = false;
    cv::Point_<float> mean;
    getMean(votes, mean);

    for (int i = 0; i < num_iterations and covergenz == false; i++) {
      cv::Point_<float> shifted_mean;
      getWeightedMean(votes, mean, kernel, shifted_mean);

      if (dist_l2(shifted_mean, mean) < stopping_criteria)
        covergenz = true;
      mean = shifted_mean;
    }
    result = mean;
  }

  virtual ~MeanShift() {
  };
private:

  static void getWeightedMean(const std::vector<Vote>& votes,
      const cv::Point_<float> mean, float lamda, cv::Point_<float>& shifted_mean) {
    shifted_mean = cv::Point(0.0, 0.0);
    float sum_w = 0;
    for (unsigned int i = 0; i < votes.size(); i++) {

      if (!votes[i].check)
        continue;

      float d = dist_l2(mean, votes[i].pos);
      d = 1.0 / exp(d / lamda);
      float w = votes[i].weight * d;
      shifted_mean.x += votes[i].pos.x * w;
      shifted_mean.y += votes[i].pos.y * w;
      sum_w += w;
    }

    if (sum_w > 0) {
      shifted_mean.x /= sum_w;
      shifted_mean.y /= sum_w;
    }
  }

  static void getMean(const std::vector<Vote>& votes, cv::Point_<float>& mean) {

    mean = cv::Point(0.0, 0.0);
    float sum_w = 0;
    for (unsigned int i = 0; i < votes.size(); i++) {

      if (!votes[i].check)
        continue;

      float w = votes[i].weight;
      mean.x += votes[i].pos.x * w;
      mean.y += votes[i].pos.y * w;
      sum_w += w;
    }

    if (sum_w > 0) {
      mean.x /= sum_w;
      mean.y /= sum_w;
    }
  }
};

#endif /* MEAN_SHIFT_HPP_ */
