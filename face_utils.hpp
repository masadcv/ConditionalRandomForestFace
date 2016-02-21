/*
 * face_utils.hpp
 *
 *  Created on: Jan 19, 2012
 *      Author: Matthias Dantone
 */

#ifndef FACE_UTILS_HPP_
#define FACE_UTILS_HPP_

#include <opencv2/opencv.hpp>
#include "forest.hpp"
#include "head_pose_sample.hpp"
#include "multi_part_sample.hpp"



struct FaceAnnotation {
    // number of facial feature points
    std::vector<cv::Point> parts;

    // url to original image
    std::string url;

    // bounding box
    cv::Rect bbox;

    // head pose
    int pose;
};

struct Vote {
	Vote(): check(false){};
	cv::Point2i pos;
	float weight;
	bool check;
};

void get_headpose_votes_mt( const ImageSample& sample,
		const Forest<HeadPoseSample>& forest,
		cv::Rect face_box,
		std::vector<HeadPoseLeaf*>& leafs,
		int step_size = 5);

void get_ffd_votes_mt( const ImageSample& sample,
						const Forest<MPSample>& forest,
						cv::Rect face_box,
						std::vector<std::vector<Vote> >& votes,
						MultiPartEstimatorOption option = MultiPartEstimatorOption());

// loads and parse annotations
// returns false if file not found
bool load_annotations( std::vector<FaceAnnotation>& annotations, std::string url);

// computes the area under curve
float areaUnderCurve( float x1, float x2, double mean, double std );

// returns the intersection
cv::Rect intersect( const cv::Rect r1, const cv::Rect r2);

// extract a region of interest
void extract_face( const cv::Mat& img, FaceAnnotation& ann,
		cv::Mat& face , int offset_x, int offset_y);

// rescale image
void rescale_img( const cv::Mat& src,
        cv::Mat& dest,
        float scale,
        FaceAnnotation& ann);

// loads and parse the config file
// in case of failure this function returns default values
bool loadConfigFile( std::string filename, ForestParam& param );

// plots all the votes for each part
void plot_ffd_votes(const cv::Mat& face,
        std::vector<std::vector<Vote> >& votes,
        std::vector<cv::Point> results,
        std::vector<cv::Point> gt);

// displays the annotations
void plot_face(const cv::Mat& img, FaceAnnotation ann );

#endif /* FACE_UTILS_HPP_ */
