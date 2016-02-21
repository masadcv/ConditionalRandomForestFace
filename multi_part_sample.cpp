/*
 * FFDSample.cpp
 *
 *  Created on: May 5, 2011
 *      Author: Matthias Dantone
 */

#include "multi_part_sample.hpp"
#include <boost/numeric/conversion/bounds.hpp>
#include <stdio.h>
#include <math.h>
#include <iostream>
using namespace cv;
using namespace std;

MPSample::MPSample(const ImageSample* image_, cv::Rect rect_,
    const cv::Rect roi_, const std::vector<cv::Point> ann_parts, float size_,
    bool label, float lamda) :
    image(image_), rect(rect_), roi(roi_), size(size_) {
  int offx = rect.width / 2;
  int offy = rect.height / 2;

  Point center_patch(rect.x + offx, rect.y + offy);
  Point center_bbox(roi.x + roi.width / 2, roi.y + roi.height / 2);

  num_parts = ann_parts.size();
  dist = Mat(1, num_parts, CV_32FC1, Scalar::all(0.0));
  part_offsets.resize(num_parts);
  isPos = false;
  for (int i = 0; i < num_parts; i++) {
    part_offsets[i].x = (ann_parts[i].x - center_patch.x);
    part_offsets[i].y = (ann_parts[i].y - center_patch.y);

    cv::Point_<float> offset = cv::Point_<float>(part_offsets[i].x / size, part_offsets[i].y / size);
    float norm = cv::norm(offset);
    if (norm == 0) {
      dist.at<float>(0, i) = 1;
    } else {
      dist.at<float>(0, i) = 1 / exp(norm / lamda);

      if (dist.at<float>(0, i) > 0.09)
        isPos = true;
    }
  }

  patch_offset.x = (center_bbox.x - center_patch.x);
  patch_offset.y = (center_bbox.y - center_patch.y);
}

MPSample::MPSample(const ImageSample* patch_, cv::Rect rect_, int n_points, float size_) :
    image(patch_), rect(rect_), size(size_) {

  num_parts = n_points;
  dist = Mat(1, n_points, CV_32FC1, Scalar::all(0.0));

  isPos = false;
  for (int i = 0; i < n_points; i++) {
    patch_offset.x = 0;
    patch_offset.y = 0;
    dist.at<float>(0, i) = 0;

  }

  //compute distance to face center
  distToCenter = 0;

}

MPSample::MPSample(const ImageSample* patch_, cv::Rect rect_) :
    image(patch_), rect(rect_) {
}

int MPSample::evalTest(const Split& test) const {
  return image->evalTest(test.feature, rect);
}

bool MPSample::eval(const Split& test) const {
  return evalTest(test) <= test.threshold;
}

double MPSample::evalSplit(const std::vector<MPSample*>& setA,
    const std::vector<MPSample*>& setB, const std::vector<float>& poppClasses,
    float splitMode, int depth) {

  int mode = int(splitMode) / 50;
  if (splitMode < 50 or depth < 2) {
    mode = 0;
  } else {
    mode = 1;
  }

  mode = 1;
  int size = setA.size() + setB.size();
  if (mode == 0) {
    double ent_a = entropie(setA);
    double ent_b = entropie(setB);
    return (ent_a * setA.size() + ent_b * setB.size()) / static_cast<double>(size);
  } else {
    double ent_a = entropie_parts(setA);
    double ent_b = entropie_parts(setB);
    return (ent_a * setA.size() + ent_b * setB.size()) / static_cast<double>(size);
  }

}

double MPSample::entropie_parts(const std::vector<MPSample*>& set) {
  double n_entropy = 0;
  double num_parts = set[0]->num_parts;
  cv::Mat sum = set[0]->dist.clone();
  sum.setTo(Scalar::all(0.0));

  vector<MPSample*>::const_iterator itSample;
  for (itSample = set.begin(); itSample < set.end(); itSample++)
    add(sum, (*itSample)->dist, sum);

  sum /= static_cast<float>(set.size());float
  p;
  for (int i = 0; i < num_parts; i++) {
    p = sum.at<float>(0, i);
    if (p > 0)
      n_entropy += p * log(p);
  }
  return n_entropy;

}

double MPSample::entropie(const std::vector<MPSample*>& set) {
  double n_entropy = 0;
  vector<MPSample*>::const_iterator itSample;
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

bool MPSample::generateSplit(const std::vector<MPSample*>& data, boost::mt19937* rng, ForestParam fp, Split& split, float split_mode,
    int depth) {
  int patchSize = fp.faceSize * fp.patchSizeRatio;
  int num_feat_channels = data[0]->image->featureChannels.size();
  split.feature.generate(patchSize, rng, num_feat_channels);

  split.num_thresholds = 25;
  split.margin = 0;

  return true;
}

void MPSample::makeLeaf(MPLeaf& leaf, const std::vector<MPSample*>& set, const std::vector<float>& poppClasses, int leaf_id) {
  int num_parts;
  if (set.size() > 0) {
    num_parts = set[0]->part_offsets.size();
  } else {
    leaf.forgound = 0;
    num_parts = 0;
    cout << "something is wrong " << endl;
  }

  int nElements = set.size();

  leaf.parts_offset.clear();
  leaf.parts_offset.resize(num_parts);
  leaf.variance.resize(num_parts);
  leaf.pF.resize(num_parts);
  leaf.nSamples = nElements;

  for (int j = 0; j < num_parts; j++) {
    leaf.parts_offset[j] = Point(0, 0);
    leaf.variance[j] = boost::numeric::bounds<float>::highest();
    leaf.pF[j] = 0;
  }
  leaf.patch_offset = cv::Point(0, 0);
  leaf.forgound = 0;

  std::vector<MPSample*>::const_iterator itSample;
  int size = 0;
  for (itSample = set.begin(); itSample < set.end(); ++itSample) {
    if ((*itSample)->isPos)
      size++;
  }

  if (size > 0) {
    for (int j = 0; j < num_parts; j++) {

      cv::Point_<int> mean(0.0, 0.0);
      float sumDist = 0;

      for (itSample = set.begin(); itSample < set.end(); ++itSample) {
        if ((*itSample)->isPos) {
          sumDist += (*itSample)->dist.at<float>(0, j);
          mean += (*itSample)->part_offsets[j];
        }
      }
      mean.x /= static_cast<int>(size);
      mean.y /= static_cast<int>(size);

      leaf.pF[j] = static_cast<float>(sumDist) / size;
      leaf.parts_offset[j] = mean;

      double var = 0.0;
      for (itSample = set.begin(); itSample < set.end(); ++itSample) {
        if ((*itSample)->isPos) {
          int x = (*itSample)->part_offsets[j].x;
          int y = (*itSample)->part_offsets[j].y;
          float dist = sqrt((x - mean.x) * (x - mean.x) + (y - mean.y) * (y - mean.y));
          var += dist;
        }
      }
      var /= size;
      leaf.variance[j] = var;

//            cout <<"leaf: offset["<<mean.x<<","<<mean.y<<"] pf:"<<leaf.pF[j]<< " var:" << var<<endl;
//            {
//				cv::Point_<int> mean(0.0,0.0);
//				cv::Point_<int> mean_sq(0.0,0.0);
//
//				for ( itSample = set.begin(); itSample < set.end(); ++itSample ){
//					if( (*itSample)->isPos){
//						mean += (*itSample)->part_offsets[j];
//						mean_sq.x += (*itSample)->part_offsets[j].x * (*itSample)->part_offsets[j].x;
//						mean_sq.y += (*itSample)->part_offsets[j].y * (*itSample)->part_offsets[j].y;
//					}
//				}
//
//				mean.x /= size;
//				mean.y /= size;
//
//				var = mean_sq.x / size - mean.x * mean.x+
//				      mean_sq.y / size - mean.y * mean.y;
//	            cout <<"leaf: offset["<<mean.x<<","<<mean.y<<"] pf:"<<leaf.pF[j]<< " var:" << sqrt(var) <<endl;
//
//            }

    }
    leaf.patch_offset = cv::Point(0, 0);
    for (itSample = set.begin(); itSample < set.end(); ++itSample) {
      if ((*itSample)->isPos) {
        leaf.patch_offset += (*itSample)->patch_offset;
      }
    }
    leaf.patch_offset.x /= static_cast<int>(size);
    leaf.patch_offset.y /= static_cast<int>(size);

    leaf.forgound = size / static_cast<float>(set.size());

  }
}

//not needed for this task
void MPSample::calcWeightClasses(std::vector<float>& poppClasses, const std::vector<MPSample*>& set) {
  poppClasses.resize(1);
  int size = 0;
  std::vector<MPSample*>::const_iterator itSample;

  //count samples near the feature point
  for (itSample = set.begin(); itSample < set.end(); ++itSample) {
    if ((*itSample)->isPos) {
      size++;
    }
  }
  poppClasses[0] = size / static_cast<float>(set.size() - size);

}

