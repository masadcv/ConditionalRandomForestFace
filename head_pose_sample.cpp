/*
 * head_pose_sample.cpp
 *
 *  Created on: May 11, 2012
 *      Author: Matthias Dantone
 */

#include "head_pose_sample.hpp"
#include <boost/numeric/conversion/bounds.hpp>

#include "timing.hpp"
using namespace std;
using namespace cv;

int HeadPoseSample::evalTest(const Split& test) const {
  return image->evalTest(test.feature, rect);
}

bool HeadPoseSample::eval(const Split& test) const {
  return evalTest(test) <= test.threshold;
}

bool HeadPoseSample::generateSplit(const vector<HeadPoseSample*>& data, boost::mt19937* rng, ForestParam fp, Split& split, float split_mode,
    int depth) {
  int patchSize = fp.faceSize * fp.patchSizeRatio;
  int num_feat_channels = data[0]->image->featureChannels.size();
  split.feature.generate(patchSize, rng, num_feat_channels);

  split.num_thresholds = 25;
  split.margin = 0;

  return true;
}

double HeadPoseSample::evalSplit(const vector<HeadPoseSample*>& setA,
    const vector<HeadPoseSample*>& setB, const vector<float>& poppClasses,
    float splitMode, int depth) {
  if (splitMode < 50) {
    double ent_a = entropie(setA);
    double ent_b = entropie(setB);
    return (ent_a * setA.size() + ent_b * setB.size()) / static_cast<double>(setA.size() + setB.size());
  } else {
    int size_a = 0;
    int size_b = 0;
    double ent_a = gain2(setA, &size_a);
    double ent_b = gain2(setB, &size_b);
    return (ent_a * size_a + ent_b * size_b) / static_cast<double>(size_b + size_a);
  }
}

double HeadPoseSample::gain(const vector<HeadPoseSample*>& set,
    int* num_pos_elements) {
  int size = 0;
  std::vector<HeadPoseSample*>::const_iterator itSample;

  //fill points for variance calculation
  float mean_pos = 0;
  for (itSample = set.begin(); itSample < set.end(); ++itSample) {
    if ((*itSample)->isPos) {
      size++;
      mean_pos += (*itSample)->label;
    }
  }
  mean_pos /= size;
  *num_pos_elements = size;

  if (size > 0) {
    float var = 0;
    for (itSample = set.begin(); itSample < set.end(); ++itSample) {
      if ((*itSample)->isPos) {
        var += (mean_pos - (*itSample)->label) * (mean_pos - (*itSample)->label);
      }
    }
    var /= size;
    return -var;
  } else {
    return boost::numeric::bounds<double>::lowest();
  }
}

double HeadPoseSample::gain2(const vector<HeadPoseSample*>& set,
    int* num_pos_elements) {
  int n = 0;
  int sum = 0;
  int sq_sum = 0;

  std::vector<HeadPoseSample*>::const_iterator itSample;
  for (itSample = set.begin(); itSample < set.end(); ++itSample) {
    if ((*itSample)->isPos) {
      n++;
      int l = (*itSample)->label;
      sum += l;
      sq_sum += l * l;
    }
  }
  *num_pos_elements = n;

  double mean = static_cast<float>(sum) / n;
  double variance = static_cast<float>(sq_sum) / n - mean * mean;
  return -variance;

}

double HeadPoseSample::entropie_pose(const vector<HeadPoseSample*>& set) {
  double n_entropy = 0;
  for (int i = 0; i < 5; i++) {
    vector<HeadPoseSample*>::const_iterator itSample;
    int size = 0;
    int p = 0;
    for (itSample = set.begin(); itSample < set.end(); ++itSample) {
      if ((*itSample)->isPos) {
        size += 1;

        if ((*itSample)->label == i) {
          p += 1;
        }
      }
    }
    double p_pos = float(p) / size;
    if (p_pos > 0)
      n_entropy += p_pos * log(p_pos);
  }
  return n_entropy;
}

double HeadPoseSample::entropie(const vector<HeadPoseSample*>& set) {
  double n_entropy = 0;
  vector<HeadPoseSample*>::const_iterator itSample;
  int p = 0;
  for (itSample = set.begin(); itSample < set.end(); ++itSample)
    if ((*itSample)->isPos)
      p += 1;

  double p_pos = float(p) / set.size();
  if (p_pos > 0)
    n_entropy += p_pos * log(p_pos);

  double p_neg = float(set.size() - p) / set.size();
  if (p_neg > 0)
    n_entropy += p_neg * log(p_neg);

  return n_entropy;
}

void HeadPoseSample::makeLeaf(HeadPoseLeaf& leaf,
    const std::vector<HeadPoseSample*>& set,
    const std::vector<float>& poppClasses,
    int leaf_id) {
  std::vector<HeadPoseSample*>::const_iterator itSample;
  int size = 0;
  for (itSample = set.begin(); itSample < set.end(); ++itSample) {
    if ((*itSample)->isPos) {
      size++;
    }
  }
  leaf.forgound = size / static_cast<float>(set.size());
  leaf.nSamples = set.size();

  leaf.hist_labels.clear();
  leaf.hist_labels.resize(5, 0);
  if (size > 0) {
    for (itSample = set.begin(); itSample < set.end(); ++itSample) {
      if ((*itSample)->isPos)
        leaf.hist_labels[(*itSample)->label]++;
    }
    for (unsigned int i = 0; i < leaf.hist_labels.size(); i++) {
      cout << leaf.hist_labels[i] << ", ";
    }
    cout << endl;
  } else {
    cout << "leaf with only neg images. " << set.size() << endl;
    for (unsigned int i = 0; i < leaf.hist_labels.size(); i++) {
      leaf.hist_labels[i] = 0;
    }
  }
}

void HeadPoseSample::calcWeightClasses(std::vector<float>& poppClasses,
    const std::vector<HeadPoseSample*>& set) {
  poppClasses.resize(5, 0);
  int size = 0;
  std::vector<HeadPoseSample*>::const_iterator itSample;
  for (itSample = set.begin(); itSample < set.end(); ++itSample) {
    if ((*itSample)->isPos) {
      size++;
      poppClasses[(*itSample)->label]++;
    }
  }
  std::cout << "Class Histogram: " << std::endl;
  for (unsigned int i = 0; i < poppClasses.size(); i++) {
    std::cout << i << " -> " << poppClasses[i] << " " << poppClasses[i] / size << std::endl;
    poppClasses[i] /= size;
  }
}

void HeadPoseSample::show() {
  cv::imshow("X", image->featureChannels[0](rect));
  cv::Mat face = image->featureChannels[0].clone();
  cv::rectangle(face, rect, cv::Scalar(255, 255, 255, 0));
  if (label >= 0) {
    cv::rectangle(face, roi, cv::Scalar(255, 255, 255, 0));
  }
  cv::imshow("Y", face);
  cv::waitKey(0);
}
;
