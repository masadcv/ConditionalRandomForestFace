/*
 * demo.cpp
 *
 *  Created on: Aug 11, 2012
 *      Author: Matthias Dantone
 */

#include "forest.hpp"
#include "multi_part_sample.hpp"
#include "head_pose_sample.hpp"
#include "face_utils.hpp"
#include <istream>
#include <opencv2/opencv.hpp>
#include <boost/progress.hpp>
#include "face_forest.hpp"
#include "feature_channel_factory.hpp"
#include "timing.hpp"
using namespace std;
using namespace cv;

float dist(const Point& a, const Point&b) {
  return sqrt((a.x - b.x) * (a.x - b.x) + (a.y - b.y) * (a.y - b.y));
}
float get_inter_occular_dist( const FaceAnnotation& ann ) {
  Point center_left;
  center_left.x = (ann.parts[0].x+ann.parts[2].x)/2;
  center_left.y = (ann.parts[0].y+ann.parts[2].y)/2;

  Point center_right;
  center_right.x = (ann.parts[6].x+ann.parts[7].x)/2;
  center_right.y = (ann.parts[6].y+ann.parts[7].y)/2;

  return( dist(center_left, center_right));
}
void eval_forest(FaceForestOptions option, vector<FaceAnnotation>& annotations) {
  //init face forest
  FaceForest ff(option);

  vector<vector<float> > errors;
  boost::progress_display show_progress(annotations.size());
  for (int i = 0; i < static_cast<int>(annotations.size()); ++i, ++show_progress) {

    // load image
    Mat image;
    image = cv::imread(annotations[i].url, 1);
    if (image.data == NULL) {
      std::cerr << "could not load " << annotations[i].url << std::endl;
      continue;
    }

    // convert to grayscale
    Mat img_gray;
    cvtColor(image, img_gray, CV_BGR2GRAY);

    Face face;
    ff.analize_face(img_gray, annotations[i].bbox, face);

    vector<float> err;
    float inter_occular_dist = get_inter_occular_dist(annotations[i]);
    for (int j = 0; j < face.ffd_cordinates.size(); j++) {
      float d = dist(annotations[i].parts[j], face.ffd_cordinates[j]);
      err.push_back(d / inter_occular_dist);
    }
    errors.push_back(err);
  }

  //write to file
  ofstream outFile;
  string f = "/home/mdantone/scratch/grid/cvpr_public/experiments/error_updated.txt";
  outFile.open(f.c_str(), ios::out);
  for( int i=0; i < errors.size(); i++){
    for( int j=0; j < errors[i].size(); j++){
      outFile << errors[i][j] << " ";
    }
    outFile << "\n";
  }
}

int main(int argc, char** argv) {

  if (argc < 3) {
    cout << "ERROR during flag parsing" << endl;
    cout << "you need to set 4 flags: \n mode (0==training, 1==evaluate)" << endl;
    cout << " path to ffd config file" << endl;
    cout << " path to headpose config file" << endl;
    cout << " path to face cascade" << endl;

  }

  // mode 0: training fiducial point detector
  //      1: evaluate
  int mode = 1;
//  std::string ffd_config_file = "data/config_ffd.txt";
//  std::string headpose_config_file = "data/config_headpose.txt";
//  std::string face_cascade = "data/haarcascade_frontalface_alt.xml";
  std::string ffd_config_file = "/scratch/mdantone/grid/cvpr_public/cvpr_public/data/config_ffd.txt";
  std::string headpose_config_file = "/scratch/mdantone/grid/cvpr_public/cvpr_public/data/config_headpose.txt";
  std::string face_cascade = "/scratch/mdantone/grid/cvpr_public/cvpr_public/data/haarcascade_frontalface_alt.xml";

  if (argc > 3) {
    try {
      mode = boost::lexical_cast<int>(argv[1]);
      ffd_config_file = argv[2];
      headpose_config_file = argv[3];
      face_cascade = argv[4];
    } catch (char * str) {
      cout << "ERROR during flag parsing" << endl;
    }
  }

  // parse config file
  ForestParam mp_param;
  assert(loadConfigFile(ffd_config_file, mp_param));

  // loading GT
  vector<FaceAnnotation> annotations;
  load_annotations(annotations, mp_param.imgPath);

  FaceForestOptions option;
  option.face_detection_option.path_face_cascade = face_cascade;

  ForestParam head_param;
  assert(loadConfigFile(headpose_config_file, head_param));

  option.head_pose_forest_param = head_param;
  option.mp_forest_param = mp_param;

  eval_forest(option, annotations);

  return 0;
}
