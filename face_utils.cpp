/*
 * face_utils.hpp
 *
 *  Created on: Jan 19, 2012
 *      Author: Matthias Dantone
 */

#include "opencv_serialization.hpp"
#include <boost/archive/binary_oarchive.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/utility.hpp>
#include <boost/filesystem/convenience.hpp>
#include <boost/filesystem.hpp>
#include <boost/iostreams/device/file.hpp>
#include <boost/iostreams/stream.hpp>
#include <boost/lexical_cast.hpp>

#include <opencv2/opencv.hpp>
#include <fstream>
#include <math.h>

#include "face_utils.hpp"

void plot_ffd_votes(const cv::Mat& face,
    std::vector<std::vector<Vote> >& votes, std::vector<cv::Point> results,
    std::vector<cv::Point> gt) {
  int num_parts = static_cast<int> (votes.size());
  std::vector < cv::Mat > plots;
  cv::Mat plotFace = face.clone();

  for (int i = 0; i < num_parts; i++) {
    cv::Mat plot = cv::Mat(face.cols, face.rows, CV_32FC1);
    plot.setTo(cv::Scalar::all(0.0));

    for (unsigned int j = 0; j < votes[i].size(); j++) {
      Vote& v = votes[i][j];
      if (v.pos.x > 0 and v.pos.x < face.cols and v.pos.y > 0 and v.pos.y
          < face.rows) {

        plot.at<float> (v.pos.y, v.pos.x) += v.weight;
      }
    }
    plots.push_back(plot);
    if (i < static_cast<int> (results.size())) {
      int x = results[i].x;
      int y = results[i].y;
      cv::circle(plotFace, cv::Point_<int>(x, y), 3, cv::Scalar(255, 255, 255,
          0));
    }
    if (i < static_cast<int> (gt.size())) {
      int x = gt[i].x;
      int y = gt[i].y;
      cv::circle(plotFace, cv::Point_<int>(x, y), 3, cv::Scalar(0, 0, 0, 0));
    }
  }
  for (int i = 0; i < num_parts; i++) {
    cv::imshow("Plot Votes", plots[i]);
    cv::imshow("Face results", plotFace);
  }
  cv::waitKey(0);
}

void plot_face(const cv::Mat& img, FaceAnnotation ann) {
  cv::Mat plot = img.clone();
  for (int i = 0; i < static_cast<int> (ann.parts.size()); i++) {
    int x = ann.parts[i].x + ann.bbox.x;
    int y = ann.parts[i].y + ann.bbox.y;
    cv::circle(plot, cv::Point_<int>(x, y), 3, cv::Scalar(0, 0, 0, 0));
  }
  cv::rectangle(plot, ann.bbox, cv::Scalar(0, 0, 255), 3);
  cv::imshow("Face ", plot);
  cv::waitKey(0);
}

void get_headpose_votes_mt(const ImageSample& sample, const Forest<
    HeadPoseSample>& forest, cv::Rect face_box,
    std::vector<HeadPoseLeaf*>& leafs, int step_size) {

  ForestParam param = forest.getParam();
  int patch_size = param.faceSize * param.patchSizeRatio;
  int num_treads = boost::thread::hardware_concurrency();
  boost::thread_pool::executor e(num_treads);

  std::vector < HeadPoseSample > samples;
  samples.reserve((face_box.width - patch_size + 1) * (face_box.height
      - patch_size + 1));
  int num_trees = forest.trees.size();
  for (int x = face_box.x; x < face_box.x + face_box.width - patch_size; x
      += step_size) {
    for (int y = face_box.y; y < face_box.y + face_box.height - patch_size; y
        += step_size) {
      cv::Rect patch_box(x, y, patch_size, patch_size);
      samples.push_back(HeadPoseSample(&sample, patch_box));
    }
  }

  leafs.resize(samples.size() * num_trees);
  for (unsigned int i = 0; i < samples.size(); i++) {
    e.submit(boost::bind(&Forest<HeadPoseSample>::evaluate_mt, forest,
        &samples[i], &leafs[i * num_trees]));
  }
  e.join_all();
}

void get_ffd_votes_mt(const ImageSample& sample,
    const Forest<MPSample>& forest, cv::Rect face_box, std::vector<std::vector<
        Vote> >& votes, MultiPartEstimatorOption option) {

  ForestParam param = forest.getParam();
  int patch_size = param.faceSize * param.patchSizeRatio;
  int num_parts = static_cast<int> (votes.size());
  int num_treads = boost::thread::hardware_concurrency();
  boost::thread_pool::executor e(num_treads);

  //collect all samples
  std::vector < MPSample > samples;
  samples.reserve((face_box.width - patch_size + 1) * (face_box.height
      - patch_size + 1));
  int num_trees = forest.trees.size();
  for (int x = face_box.x; x < face_box.x + face_box.width - patch_size; x
      += option.step_size) {
    for (int y = face_box.y; y < face_box.y + face_box.height - patch_size; y
        += option.step_size) {
      cv::Rect patch_box(x, y, patch_size, patch_size);
      samples.push_back(MPSample(&sample, patch_box));
    }
  }

  //evaluate samples
  std::vector<MPLeaf*> leafs;
  leafs.resize(samples.size() * num_trees);
  for (unsigned int i = 0; i < samples.size(); i++) {
    e.submit(boost::bind(&Forest<MPSample>::evaluate_mt, forest, &samples[i],
        &leafs[i * num_trees]));
  }
  e.join_all();

  // parse leafs
  std::vector<MPLeaf*>::iterator itLeaf;
  int i_sample = 0;
  for (itLeaf = leafs.begin(); itLeaf < leafs.end(); itLeaf++) {
    assert(int(samples.size()) > (i_sample / num_trees));
    int off_set_x = samples[i_sample / num_trees].rect.x + patch_size / 2;
    int off_set_y = samples[i_sample / num_trees].rect.y + patch_size / 2;
    for (int i = 0; i < num_parts; i++) {

      float min_pf = option.min_pf;
      if (i == 0 || i == 7) {
        min_pf *= 1.5;
      }
      if ((*itLeaf)->forgound > option.min_forground && (*itLeaf)->pF[i]
          > min_pf && (*itLeaf)->variance[i] < option.max_variance
          && (*itLeaf)->nSamples > option.min_samples) {
        Vote v;
        v.pos.x = (*itLeaf)->parts_offset[i].x + off_set_x;
        v.pos.y = (*itLeaf)->parts_offset[i].y + off_set_y;
        v.weight = (*itLeaf)->forgound;// * (*itLeaf)->pF[i];
        v.check = true;
        votes[i].push_back(v);
      }
    }
    i_sample++;
  }
}

bool load_annotations(std::vector<FaceAnnotation>& annotations, std::string url) {
  if (boost::filesystem::exists(url.c_str())) {
    std::string filename(url.c_str());
    boost::iostreams::stream < boost::iostreams::file_source > file(
        filename.c_str());
    std::string line;
    while (std::getline(file, line)) {
      std::vector < std::string > strs;
      boost::split(strs, line, boost::is_any_of(" "));

      FaceAnnotation ann;

      ann.url = strs[0];
      ann.bbox.x = boost::lexical_cast<int>(strs[1]);
      ann.bbox.y = boost::lexical_cast<int>(strs[2]);
      ann.bbox.width = boost::lexical_cast<int>(strs[3]);
      ann.bbox.height = boost::lexical_cast<int>(strs[4]);

      ann.pose = boost::lexical_cast<int>(strs[5]);

      int num_points = boost::lexical_cast<int>(strs[6]);
      ann.parts.resize(num_points);
      for (int i = 0; i < num_points; i++) {
        ann.parts[i].x = boost::lexical_cast<int>(strs[7 + 2 * i]);
        ann.parts[i].y = boost::lexical_cast<int>(strs[8 + 2 * i]);
      }
      annotations.push_back(ann);
    }
    return true;
  }
  return false;
}

// rescale image
void rescale_img(const cv::Mat& src, cv::Mat& dest, float scale,
    FaceAnnotation& ann) {

  cv::resize(src, dest, cv::Size(src.cols * scale, src.rows * scale), 0, 0);

  for (unsigned int j = 0; j < ann.parts.size(); j++) {
    ann.parts[j].x *= scale;
    ann.parts[j].y *= scale;
  }
  ann.bbox.x *= scale;
  ann.bbox.y *= scale;
  ann.bbox.width *= scale;
  ann.bbox.height *= scale;
}

float areaUnderCurve(float x1, float x2, double mean, double std) {
  double sum = 0;
  double stepSize = 0.01;
  double t;
  for (double x = x1; x < x2; x += stepSize) {
    t = (x - mean) / std;
    sum += exp(-0.5 * (t * t)) * stepSize;
  }

  return sum * 1.0 / (std * sqrt(2 * M_PI));
}

cv::Rect intersect(const cv::Rect r1, const cv::Rect r2) {
  cv::Rect intersection;

  // find overlapping region
  intersection.x = (r1.x < r2.x) ? r2.x : r1.x;
  intersection.y = (r1.y < r2.y) ? r2.y : r1.y;
  intersection.width = (r1.x + r1.width < r2.x + r2.width) ? r1.x + r1.width
      : r2.x + r2.width;
  intersection.width -= intersection.x;
  intersection.height = (r1.y + r1.height < r2.y + r2.height) ? r1.y
      + r1.height : r2.y + r2.height;
  intersection.height -= intersection.y;

  // check for non-overlapping regions
  if ((intersection.width <= 0) || (intersection.height <= 0)) {
    intersection = cvRect(0, 0, 0, 0);
  }
  return intersection;
}

void extract_face(const cv::Mat& img, FaceAnnotation& ann, cv::Mat& face,
    int offset_x, int offset_y) {
  // extract face
  cv::Rect bigbox = cv::Rect(ann.bbox.x - offset_x, ann.bbox.y - offset_y,
      ann.bbox.width + offset_x * 2, ann.bbox.height + offset_y * 2);
  cv::Rect facebbox = intersect(bigbox, cv::Rect(0, 0, img.cols, img.rows));

  face = img(facebbox);
  //update GT
  for (unsigned int j = 0; j < ann.parts.size(); j++) {
    ann.parts[j].x -= (facebbox.x - ann.bbox.x);
    ann.parts[j].y -= (facebbox.y - ann.bbox.y);
  }

  ann.bbox = facebbox;
  ann.bbox.x = 0;
  ann.bbox.y = 0;
}

bool loadConfigFile( std::string filename, ForestParam& param ){
  if (boost::filesystem::exists(filename.c_str())) {
    boost::iostreams::stream < boost::iostreams::file_source > file(
        filename.c_str());
    std::string line;
    if (file.is_open()) {

      // Path to images
      std::getline(file, line);
      std::getline(file, line);
      param.imgPath = line;
      std::cout << "Image path" << param.imgPath << std::endl;

      // Path to trees
      std::getline(file, line);
      std::getline(file, line);
      param.treePath = line;
      std::cout << "paths to trees" << param.treePath << std::endl;

      // Number of trees
      std::getline(file, line);
      std::getline(file, line);
      param.nTrees = boost::lexical_cast<int>(line);

      // Number of tests
      std::getline(file, line);
      std::getline(file, line);
      param.nTests = boost::lexical_cast<int>(line);

      // Max deth
      std::getline(file, line);
      std::getline(file, line);
      param.max_d = boost::lexical_cast<int>(line);

      // Min samples per Node
      std::getline(file, line);
      std::getline(file, line);
      param.min_s = boost::lexical_cast<int>(line);

      // Samples per Tree
      std::getline(file, line);
      std::getline(file, line);
      param.nSamplesPerTree = boost::lexical_cast<int>(line);

      // Patches Per Sample
      std::getline(file, line);
      std::getline(file, line);
      param.nPatchesPerSample = boost::lexical_cast<int>(line);

      std::getline(file, line);
      std::getline(file, line);
      param.faceSize = boost::lexical_cast<int>(line);

      std::getline(file, line);
      std::getline(file, line);
      param.patchSizeRatio = boost::lexical_cast<float>(line);

      // number of Feature Channels
      std::getline(file, line);
      std::getline(file, line);
      std::vector < std::string > strs;
      boost::split(strs, line, boost::is_any_of(" "));
      for (unsigned int i = 0; i < strs.size(); i++)
        param.features.push_back(boost::lexical_cast<float>(strs[i]));

      return true;
    }
  }
  std::cout << "FILE NOT FOUND, default inizialization " << std::endl;
  //default values for ForestParam
  param.max_d = 15;
  param.min_s = 20;
  param.nTests = 250;
  param.nTrees = 10;
  param.nPatchesPerSample = 200;
  param.nSamplesPerTree = 500;
  param.faceSize = 100;
  param.patchSizeRatio = 0.25;

  return false;
}
