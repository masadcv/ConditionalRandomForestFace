/*
 * PatchFeatures.cpp
 *
 *  Created on: May 3, 2011
 *      Author: Matthias Dantone
 */

#include "image_sample.hpp"
#include "feature_channel_factory.hpp"
#include "math.h"
#include <iostream>

using namespace boost;
using namespace std;
using namespace cv;

ImageSample::ImageSample(const cv::Mat img, std::vector<int> features,
    FeatureChannelFactory& fcf,
    bool useIntegral_ = false) :
    useIntegral(useIntegral_) {
  extractFeatureChannels(img, featureChannels, features, useIntegral, fcf);
}

ImageSample::ImageSample(const cv::Mat img,
    std::vector<int> features,
    bool useIntegral_ = false) :
    useIntegral(useIntegral_) {
  FeatureChannelFactory fcf = FeatureChannelFactory();

  extractFeatureChannels(img, featureChannels, features, useIntegral, fcf);
}

int ImageSample::evalTest(const SimplePatchFeature& test,
    const cv::Rect rect) const {
  int p1 = 0;
  int p2 = 0;

  const cv::Mat ptC = featureChannels[test.featureChannel];
  if (!useIntegral) {

    Mat tmp = ptC(Rect(test.rectA.x + rect.x, test.rectA.y + rect.y, test.rectA.width, test.rectA.height));
    p1 = (cv::sum(tmp))[0] / static_cast<float>(test.rectA.width * test.rectA.height);

    Mat tmp2 = ptC(Rect(test.rectB.x + rect.x, test.rectB.y + rect.y, test.rectB.width, test.rectB.height));
    p2 = (cv::sum(tmp2))[0] / static_cast<float>(test.rectB.width * test.rectB.height);

  } else {

    int a = ptC.at<float>(rect.y + test.rectA.y, rect.x + test.rectA.x);
    int b = ptC.at<float>(rect.y + test.rectA.y, rect.x + test.rectA.x + test.rectA.width);
    int c = ptC.at<float>(rect.y + test.rectA.y + test.rectA.height, rect.x + test.rectA.x);
    int d = ptC.at<float>(rect.y + test.rectA.y + test.rectA.height, rect.x + test.rectA.x + test.rectA.width);
    p1 = (d - b - c + a) / static_cast<float>(test.rectA.width * test.rectA.height);

    a = ptC.at<float>(rect.y + test.rectB.y, rect.x + test.rectB.x);
    b = ptC.at<float>(rect.y + test.rectB.y, rect.x + test.rectB.x + test.rectB.width);
    c = ptC.at<float>(rect.y + test.rectB.y + test.rectB.height, rect.x + test.rectB.x);
    d = ptC.at<float>(rect.y + test.rectB.y + test.rectB.height, rect.x + test.rectB.x + test.rectB.width);
    p2 = (d - b - c + a) / static_cast<float>(test.rectB.width * test.rectB.height);
  }

  return p1 - p2;

}

int ImageSample::evalTest(const SimplePixelFeature& test, const cv::Rect rect) const {
  return featureChannels[test.featureChannel].at<unsigned char>(rect.y + test.point_a.y, rect.x + test.point_a.x)
      - featureChannels[test.featureChannel].at<unsigned char>(rect.y + test.point_b.y, rect.x + test.point_b.x);
}

void ImageSample::extractFeatureChannels(const Mat& img,
    std::vector<Mat>& vImg,
    vector<int> features,
    bool useIntegral,
    FeatureChannelFactory& fcf) const {

  cv::Mat img_gray;
  if (img.channels() == 1) {
    img_gray = img;
  } else {
    cv::cvtColor(img, img_gray, CV_RGB2GRAY);
  }

  sort(features.begin(), features.end());
  for (unsigned int i = 0; i < features.size(); i++) {
    fcf.extractChannel(features[i], useIntegral, img_gray, vImg);
  }
}

void ImageSample::getSubPatches(Rect rect, vector<Mat>& tmpPatches) {
  for (unsigned int i = 0; i < featureChannels.size(); i++)
    tmpPatches.push_back(featureChannels[i](rect));
}

ImageSample::~ImageSample() {
  for (unsigned int i = 0; i < featureChannels.size(); i++)
    featureChannels[i].release();
  featureChannels.clear();
}

