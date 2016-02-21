/*
 * train.cpp
 *
 *  Created on: May 2, 2012
 *      Author: Matthias Dantone
 */

#include "forest.hpp"
#include "multi_part_sample.hpp"
#include "head_pose_sample.hpp"
#include "face_utils.hpp"
#include <istream>
#include <opencv2/opencv.hpp>
#include <boost/progress.hpp>

using namespace std;
int main(int argc, char** argv) {
  string config_file = argv[1];
  int off_set = boost::lexical_cast<int>(argv[2]);

  cout << "configuration file: " << config_file << endl;
  cout << "off set: " << off_set << endl;

  std::vector<FaceAnnotation> annotations;
  ForestParam param;
  assert(loadConfigFile(config_file, param));
  cout << "config file parsed." << endl;

  Tree<HeadPoseSample>* tree;
  char savePath[200];
  sprintf(savePath, "%s%03d.txt", param.treePath.c_str(), off_set);
  bool unfinished_tree = false;
  std::ifstream ifs(savePath);
  if (!ifs) {
    std::cout << "tree not found" << std::endl;
  } else {
    try {
      boost::archive::binary_iarchive ia(ifs);
      ia >> tree;
      if (tree->isFinished()) {
        std::cout << "complete tree reloaded" << std::endl;
        exit(0);
      } else {
        unfinished_tree = true;
        std::cout << "unfinished tree reloaded,  keep growing" << std::endl;
      }
    } catch (boost::archive::archive_exception& ex) {
      std::cout << "Reload Tree: Archive Exception during deserializing: " << ex.what() << std::endl;
      std::cout << "not able to load  " << savePath << std::endl;
    }
  }

  boost::mt19937 rng;
  rng.seed(off_set + 1);
  srand(off_set + 1);

  load_annotations(annotations, param.imgPath);
  std::random_shuffle(annotations.begin(), annotations.end());
  std::vector<std::vector<FaceAnnotation> > ann(5);
  for (unsigned int i = 0; i < annotations.size(); i++) {
    int label = annotations[i].pose + 2;
    ann[label].push_back(annotations[i]);
  }

  int sample_per_class = ann[0].size();
  for (unsigned int i = 0; i < ann.size(); i++)
    sample_per_class = min(sample_per_class, static_cast<int>(ann[i].size()));

sample_per_class  = min(sample_per_class, static_cast<int>(param.nSamplesPerTree));
  annotations.clear();
  for (unsigned int i = 0; i < ann.size(); i++) {
    for (int j = 0; j < sample_per_class; j++) {
      annotations.push_back(ann[i][j]);
    }
  }

  cout << "elements per class" << sample_per_class << endl;
  cout << "num samples " << annotations.size() << endl;
  cout << "reserved patches " << sample_per_class * (ann.size() + 1) * param.nPatchesPerSample << endl;

  std::vector<HeadPoseSample*> hp_samples;

  //5 classes + 1 neg class
  hp_samples.reserve(sample_per_class * (ann.size()) * param.nPatchesPerSample);

  boost::progress_display show_progress(annotations.size());
  for (int i = 0; i < static_cast<int>(annotations.size()); i++, ++show_progress) {
    // load image
    const cv::Mat image = cv::imread(annotations[i].url, 1);
    if (image.data == NULL)
    {
      std::cerr << "could not load " << annotations[i].url << std::endl;
      continue;
    }

    //extract patches
    int patch_size = param.faceSize * param.patchSizeRatio;

    cv::Mat img_rescaled;
    float scale = static_cast<float>(param.faceSize) / annotations[i].bbox.width;
    rescale_img(image, img_rescaled, scale, annotations[i]);

    ImageSample* sample = new ImageSample(img_rescaled, param.features, false);

    //sample pos images
    boost::uniform_int<> dist(0, param.faceSize - patch_size - 1);
    boost::variate_generator<boost::mt19937&, boost::uniform_int<> > rand_patch(rng, dist);
    for (int j = 0; j < param.nPatchesPerSample; j++) {

      cv::Rect bbox = cv::Rect(annotations[i].bbox.x + rand_patch(), annotations[i].bbox.y + rand_patch(), patch_size, patch_size);

      int head_pose = annotations[i].pose + 2; // range is between -2 and +2. but we shift it to 0 - 5
      HeadPoseSample* s = new HeadPoseSample(sample, annotations[i].bbox, bbox, head_pose);
      hp_samples.push_back(s);
      //s->show();

    }

    continue;

    //sample neg images
    boost::uniform_int<> dist_neg_x(0, img_rescaled.cols - patch_size - 1);
    boost::uniform_int<> dist_neg_y(0, img_rescaled.rows - patch_size - 1);
    boost::variate_generator<boost::mt19937&, boost::uniform_int<> > rand_neg_x(rng, dist_neg_x);
    boost::variate_generator<boost::mt19937&, boost::uniform_int<> > rand_neg_y(rng, dist_neg_y);
    for (int j = 0; j < param.nPatchesPerSample; j++) {
      cv::Rect bbox;

      int save = 0;
      while (save < 1000) {
        bbox = cv::Rect(rand_neg_x(), rand_neg_y(), patch_size, patch_size);
        cv::Rect inter = intersect(bbox, annotations[i].bbox);

        if (inter.height == 0 and inter.width == 0)
          break;
        save++;
      }
      HeadPoseSample* s = new HeadPoseSample(sample, annotations[i].bbox, bbox, -1);
      hp_samples.push_back(s);
      //s->show();
    }

  }
  cout << "used patches " << hp_samples.size() << endl;

  Forest<MPSample> forest;
  Forest<HeadPoseSample> forest2;

  Timing jobTimer = Timing();

  if (unfinished_tree) {
    tree->grow(hp_samples, jobTimer, &rng);
  } else {
    tree = new Tree<HeadPoseSample>(hp_samples, param, &rng, savePath, jobTimer);
  }

  //Tree<MPSample> tree(samples, param, &rng, savePath, jobTimer);
//	Tree<HeadPoseSample> tree2( hp_samples, param, &rng, savePath, jobTimer);
  return 0;
}
