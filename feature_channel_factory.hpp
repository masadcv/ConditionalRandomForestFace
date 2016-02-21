/*
 * FeatureChannelExtractor.h
 *
 *  Created on: Sep 7, 2011
 *      Author: Matthias Dantone
 */

#ifndef FEATURECHANNELEXTRACTOR_H_
#define FEATURECHANNELEXTRACTOR_H_

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <opencv2/imgproc/imgproc.hpp>
#include <boost/thread.hpp>
#include "thread_pool.hpp"
#include "timing.hpp"

#define FC_GRAY     0
#define FC_GABOR    1
#define FC_SOBEL    2
#define FC_MIN_MAX  3
#define FC_CANNY    4
#define FC_NORM     5

class FeatureChannelFactory {
public:

  FeatureChannelFactory() {
  }

  void extractChannel(int type, bool useIntegral, const cv::Mat& src,
      std::vector<cv::Mat>& channels) {
    if (type == FC_GRAY) {
      if (useIntegral) {
        cv::Mat int_img;
        cv::integral(src, int_img, CV_32F);
        channels.push_back(int_img);
      } else {
        channels.push_back(src);
      }
    } else if (type == FC_NORM) {
      cv::Mat normal;
      cv::equalizeHist(src, normal);
      if (useIntegral) {
        cv::Mat int_img;
        cv::integral(normal, int_img, CV_32F);
        channels.push_back(int_img);
      } else {

        channels.push_back(normal);
      }

    } else if (type == FC_GABOR) {

      //check if kernels are initzialized
      if (reals.size() == 0)
        init_gabor_kernels();

      int old_size = channels.size();

      bool multithreaded = true;
      if (multithreaded) {
        channels.resize(channels.size() + reals.size());
        int num_treads = boost::thread::hardware_concurrency();
        boost::thread_pool::executor e(num_treads);
        for (unsigned int i = 0; i < reals.size(); i++) {
          e.submit(boost::bind(&FeatureChannelFactory::gabor_transform,
              this, src, &channels[old_size + i], useIntegral, i, old_size));
        }
        e.join_all();
      } else {
        for (unsigned int i = 0; i < reals.size(); i++) {
          cv::Mat final;
          cv::Mat r_mat;
          cv::Mat i_mat;
          cv::filter2D(src, r_mat, CV_32F, reals[i]);
          cv::filter2D(src, i_mat, CV_32F, imags[i]);
          cv::pow(r_mat, 2, r_mat);
          cv::pow(i_mat, 2, i_mat);
          cv::add(i_mat, r_mat, final);
          cv::pow(final, 0.5, final);
          cv::normalize(final, final, 0, 1, CV_MINMAX, CV_32F);

          if (useIntegral) {
            cv::Mat img;
            final.convertTo(img, CV_8UC1, 255);

            cv::Mat integral_img;
            cv::integral(img, integral_img, CV_32F);
            channels.push_back(integral_img);
          } else {
            final.convertTo(final, CV_8UC1, 255);
            channels.push_back(final);
          }
        }
      }
    } else if (type == FC_SOBEL){
      cv::Mat sob_x(src.size(), CV_8U);
      cv::Mat sob_y(src.size(), CV_8U);

      cv::Sobel(src, sob_x, CV_8U, 0, 1);
      cv::Sobel(src, sob_y, CV_8U, 1, 0);

      if (useIntegral) {
        cv::Mat sob_x_int, sob_y_int;
        cv::integral(sob_x, sob_x_int, CV_32F);
        cv::integral(sob_y, sob_y_int, CV_32F);
        channels.push_back(sob_x_int);
        channels.push_back(sob_y_int);
      } else {
        channels.push_back(sob_x);
        channels.push_back(sob_y);
      }
    } else if (type == FC_MIN_MAX) {
      cv::Mat kernel(cv::Size(3, 3), CV_8UC1);
      kernel.setTo(cv::Scalar(1));
      cv::Mat img_min(src.size(), CV_8U);
      cv::Mat img_max(src.size(), CV_8U);

      cv::erode(src, img_min, kernel);
      cv::dilate(src, img_max, kernel);

      if (useIntegral) {
        cv::Mat img_min_int, img_max_int;
        cv::integral(img_min, img_min_int, CV_32F);
        cv::integral(img_max, img_max_int, CV_32F);
        channels.push_back(img_min_int);
        channels.push_back(img_max_int);
      } else {
        channels.push_back(img_min);
        channels.push_back(img_max);
      }
    } else if (type == FC_CANNY) {

      cv::Mat cannyImg;
      cv::Canny(src, cannyImg, -1, 5);
      if (useIntegral) {
        cv::Mat int_img;
        cv::integral(cannyImg, int_img, CV_32F);
        channels.push_back(int_img);
      } else {
        channels.push_back(cannyImg);
      }
    } else {
      std::cerr << "unkown feature channel" << std::endl;
    }
  }

private:

  void gabor_transform(const cv::Mat& src, cv::Mat* dst,
      bool useIntegral, int index, int old_size) const {
    cv::Mat final;
    cv::Mat r_mat;
    cv::Mat i_mat;
    cv::filter2D(src, r_mat, CV_32F, reals[index]);
    cv::filter2D(src, i_mat, CV_32F, imags[index]);
    cv::pow(r_mat, 2, r_mat);
    cv::pow(i_mat, 2, i_mat);
    cv::add(i_mat, r_mat, final);
    cv::pow(final, 0.5, final);
    cv::normalize(final, final, 0, 1, CV_MINMAX);

    if (useIntegral) {
      cv::Mat img;
      final.convertTo(img, CV_8UC1, 255);
      cv::Mat integral_img;
      cv::integral(img, integral_img, CV_32F);
      *dst = integral_img;

    } else {
      final.convertTo(final, CV_8UC1, 255);
      *dst = final;
    }
  }

  void init_gabor_kernels() {
    std::cout << "initizialize gabor kernels" << std::endl;
    //create kernels
    int NuMin = 0;
    int NuMax = 4;
    int MuMin = 0;
    int MuMax = 7;
    double sigma = 1. / 2.0 * CV_PI;
    double dF = sqrt(2.0);

    int iMu = 0;
    int iNu = 0;

    for (iNu = NuMin; iNu <= NuMax; iNu++)
      for (iMu = MuMin; iMu < MuMax; iMu++)
        createKernel(iMu, iNu, sigma, dF);
    std::cout << "initizialized gabor kernels" << std::endl;

  };

  void createKernel(int iMu, int iNu, double sigma, double dF) {
    //Initilise the parameters
    double F = dF;
    double k = (CV_PI / 2) / pow(F, (double) iNu);
    double phi = CV_PI * iMu / 8;

    double width = round((sigma / k) * 6 + 1);
    if (fmod(width, 2.0) == 0.0)
      width++;

    //create kernel
    cv::Mat m_real = cv::Mat(width, width, CV_32FC1);
    cv::Mat m_imag = cv::Mat(width, width, CV_32FC1);

    int x, y;
    double dReal;
    double dImag;
    double dTemp1, dTemp2, dTemp3;

    int off_set = (width - 1) / 2;
    for (int i = 0; i < width; i++) {
      for (int j = 0; j < width; j++) {
        x = i - off_set;
        y = j - off_set;
        dTemp1 = (pow(k, 2) / pow(sigma, 2)) * exp(-(pow((double) x, 2) + pow((double) y, 2)) * pow(k, 2) / (2 * pow(sigma, 2)));
        dTemp2 = cos(k * cos(phi) * x + k * sin(phi) * y) - exp(-(pow(sigma, 2) / 2));
        dTemp3 = sin(k * cos(phi) * x + k * sin(phi) * y);
        dReal = dTemp1 * dTemp2;
        dImag = dTemp1 * dTemp3;
        m_real.at<float>(j, i) = dReal;
        m_imag.at<float>(j, i) = dImag;
      }
    }

    reals.push_back(m_real);
    imags.push_back(m_imag);
  };

  //gabor kernels
  std::vector<cv::Mat> reals;
  std::vector<cv::Mat> imags;
};

#endif /* FEATURECHANNELEXTRACTOR_H_ */
