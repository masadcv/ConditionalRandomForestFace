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
using namespace cv;
int main(int argc, char** argv) {
  string config_file = argv[1];
  int off_set = boost::lexical_cast<int>(argv[2]);

  cout << "configuration file: " << config_file << endl;
  cout << "off set: " << off_set << endl;

  std::vector<FaceAnnotation> annotations;
  ForestParam param;
  assert(loadConfigFile(config_file, param));
  cout << "config file parsed." << endl;

  Tree<MPSample>* tree;
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

  std::vector<MPSample*> samples;

  int num_samples = param.nSamplesPerTree * param.nPatchesPerSample;
  samples.reserve(num_samples);

  boost::progress_display show_progress(param.nSamplesPerTree);
  for (int i = 0; i < static_cast<int>(annotations.size()) and static_cast<int>(samples.size()) < num_samples; i++, ++show_progress) {
    // load image
    const cv::Mat image = cv::imread(annotations[i].url, 1);
    if (image.data == NULL)
    {
      std::cerr << "could not load " << annotations[i].url << std::endl;
      continue;
    }

    Mat img_gray;
    cvtColor(image, img_gray, CV_BGR2GRAY);
    // extract patches
    int patch_size = param.faceSize * param.patchSizeRatio;

    cv::Mat img_rescaled;
    float scale = static_cast<float>(param.faceSize) / annotations[i].bbox.width;
    rescale_img(img_gray, img_rescaled, scale, annotations[i]);

    int offset = annotations[i].bbox.width * .1;
    cv::Mat face;
    extract_face(img_rescaled, annotations[i], face, 0, offset);

    equalizeHist(face, face);
    ImageSample* sample = new ImageSample(face, param.features, false);

    //sample pos images
    boost::uniform_int<> dist_x(1, face.cols - patch_size - 2);
    boost::uniform_int<> dist_y(1, face.rows - patch_size - 2);

    boost::variate_generator<boost::mt19937&, boost::uniform_int<> > rand_x(rng, dist_x);
    boost::variate_generator<boost::mt19937&, boost::uniform_int<> > rand_y(rng, dist_y);

    for (int j = 0; j < param.nPatchesPerSample; j++) {

      cv::Rect bbox = cv::Rect(rand_x(), rand_y(), patch_size, patch_size);
      MPSample* s = new MPSample(sample, bbox, Rect(0, 0, face.cols, face.rows), annotations[i].parts, param.faceSize, 1);
      samples.push_back(s);
    }

//        //sample neg images
//        boost::uniform_int<> dist_neg_x( 0 , img_rescaled.cols-patch_size -1);
//        boost::uniform_int<> dist_neg_y( 0 , img_rescaled.rows-patch_size -1);
//		boost::variate_generator<boost::mt19937&, boost::uniform_int<> > rand_neg_x(rng, dist_neg_x);
//		boost::variate_generator<boost::mt19937&, boost::uniform_int<> > rand_neg_y(rng, dist_neg_y);
//		for( int j = 0; j < param.nPatchesPerSample; j++ )
//		{
//            cv::Rect bbox;
//
//            int save = 0;
//            while( save < 1000 )
//            {
//            	bbox = cv::Rect( rand_neg_x(), rand_neg_y(), patch_size,patch_size);
//            	cv::Rect inter = intersect( bbox, annotations[i].bbox);
//
//            	if( inter.height == 0 and inter.width ==0 )
//            		break;
//            	save ++;
//            }
//            HeadPoseSample* s = new HeadPoseSample( sample, annotations[i].bbox, bbox, -1);
//			hp_samples.push_back(s);
//			//s->show();
//		}

  }

  Timing jobTimer;
  jobTimer.start();
  if (unfinished_tree) {
    tree->grow(samples, jobTimer, &rng);
  } else {
    tree = new Tree<MPSample>(samples, param, &rng, savePath, jobTimer);
  }

  return 0;
}

