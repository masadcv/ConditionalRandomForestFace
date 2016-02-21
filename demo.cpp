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

void train_forest( ForestParam param,
		vector<FaceAnnotation>& annotations ){
	int off_set = 0;
	//init random generator
	boost::mt19937 rng;
	rng.seed( off_set + 1 );
	srand( off_set + 1);

	//shuffle annotations
	std::random_shuffle( annotations.begin(), annotations.end());


	//allocate memory
	std::vector<ImageSample> samples;
	samples.resize( param.nSamplesPerTree );
	std::vector<MPSample*> mp_samples;
	int num_samples = param.nSamplesPerTree*param.nPatchesPerSample;
	mp_samples.reserve( num_samples );

	int patch_size = param.faceSize*param.patchSizeRatio;

	boost::progress_display show_progress( param.nSamplesPerTree );
	for( int i=0; i < static_cast<int>(annotations.size()) and
		static_cast<int>(mp_samples.size()) < num_samples; i++, ++show_progress) {
		// load image
		const cv::Mat image = cv::imread(annotations[i].url,1);
		if (image.data == NULL){
			std::cerr << "could not load " << annotations[i].url << std::endl;
			continue;
		}

		//convert image to grayscale
		Mat img_gray;
		cvtColor( image, img_gray, CV_BGR2GRAY );

		//rescale image to a common size
		cv::Mat img_rescaled;
		float scale =  static_cast<float>(param.faceSize)/annotations[i].bbox.width ;
		rescale_img( img_gray, img_rescaled, scale, annotations[i]);

		//enlarge the bounding box.
		int offset = annotations[i].bbox.width * .1;
		cv::Mat face;
		extract_face( img_rescaled, annotations[i],face, 0,  offset );

		//normalize imgae
		equalizeHist( face, face );

		//create image sample
		samples[i] = ImageSample(face,param.features,true);

		//randomly sample patches within the face
		boost::uniform_int<> dist_x( 1 , face.cols-patch_size-2);
		boost::uniform_int<> dist_y( 1 , face.rows-patch_size-2);
		boost::variate_generator<boost::mt19937&, boost::uniform_int<> > rand_x(rng, dist_x);
		boost::variate_generator<boost::mt19937&, boost::uniform_int<> > rand_y(rng, dist_y);
		for( int j = 0; j < param.nPatchesPerSample; j++ ) {
			cv::Rect bbox = cv::Rect( rand_x(), rand_y(), patch_size,patch_size);
			MPSample* s = new MPSample( &samples[i], bbox, Rect(0,0,face.cols, face.rows), annotations[i].parts, param.faceSize, 1 );
			mp_samples.push_back( s );

			//show patch
//			s->show();
		}
	}

	//start the training
	Timing jobTimer;
	jobTimer.start();
    char savePath[200];
	sprintf(savePath,"%s%03d.txt",param.treePath.c_str(),off_set);
	Tree<MPSample> tree( mp_samples, param, &rng, savePath, jobTimer);
}

void eval_forest( FaceForestOptions option,
		vector<FaceAnnotation>& annotations ){
	//init face forest
	FaceForest ff(option);

	for( int i=0; i < static_cast<int>(annotations.size()); ++i){
    cout << annotations[i].url << endl;

		// load image
		Mat image;
		image = cv::imread(annotations[i].url,1);
		if (image.data == NULL){
			std::cerr << "could not load " << annotations[i].url << std::endl;
			continue;
		}

		// convert to grayscale
		Mat img_gray;
    cvtColor( image, img_gray, CV_BGR2GRAY );

    bool use_predefined_bbox = true;
    vector<Face> faces;
    if( use_predefined_bbox ){
        Face face;
        ff.analize_face( img_gray, annotations[i].bbox, face );
        faces.push_back(face);
    }else{
        ff.analize_image( img_gray, faces );
    }

    cout << "ffd estimated" << endl;
    //draw results
    FaceForest::show_results( image, faces );
	}
}

int main(int argc, char** argv)
{

	if( argc < 3){
		cout << "ERROR during flag parsing" << endl;
		cout << "you need to set 4 flags: \n mode (0==training, 1==evaluate)" << endl;
		cout << " path to ffd config file" << endl;
		cout << " path to headpose config file" << endl;
		cout << " path to face cascade" << endl;

	}

	int mode = 1;
	std::string ffd_config_file = "data/config_ffd.txt";
	std::string headpose_config_file = "data/config_headpose.txt";
	std::string face_cascade = "data/haarcascade_frontalface_alt.xml";
	if( argc > 3 ){
		try{
			mode = boost::lexical_cast<int>(argv[1]);
			ffd_config_file = argv[2];
			headpose_config_file = argv[3];
			face_cascade = argv[4];
		}catch( char * str ) {
			cout << "ERROR during flag parsing" << endl;
		}
	}


	// parse config file
	ForestParam mp_param;
	assert(loadConfigFile(ffd_config_file, mp_param));

	// loading GT
	vector<FaceAnnotation> annotations;
	load_annotations( annotations, mp_param.imgPath);


	if( mode == 0){
		train_forest( mp_param, annotations );

	}else if( mode == 1 ){

		FaceForestOptions option;
		option.face_detection_option.path_face_cascade = face_cascade;

		ForestParam head_param;
		assert(loadConfigFile(headpose_config_file, head_param));
		option.head_pose_forest_param = head_param;
		option.mp_forest_param = mp_param;

		eval_forest(option, annotations);
	}else{
		cout << "unknown mode: " << mode << endl;
	}

	return 0;
}
