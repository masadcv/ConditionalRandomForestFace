/*
 * SplitGen.h
 *
 *  Created on: Aug 10, 2011
 *      Author: Matthias Dantone
 */

#ifndef SPLITGEN_H_
#define SPLITGEN_H_

#include <vector>
#include <boost/numeric/conversion/bounds.hpp>
#include <boost/limits.hpp>
#include <boost/random.hpp>
#include "timing.hpp"
#include <algorithm>
#include <boost/thread.hpp>
#include "thread_pool.hpp"

typedef std::pair<int, unsigned int> IntIndex;
struct less_than {
  bool operator()(const IntIndex& a, const IntIndex& b) const {
    return a.first < b.first;
  }
  bool operator()(const IntIndex& a, const int& b) const {
    return a.first < b;
  }
};

template<typename Sample>
class SplitGen {

public:
  typedef typename Sample::Split Split;

  SplitGen(const std::vector<Sample*>& data_, std::vector<Split>& splits_,
      boost::mt19937* rng_, ForestParam fp_,
      std::vector<float>& weightClasses_, int depth_, float split_mode_) :
      data(data_), splits(splits_), rng(rng_), fp(fp_), weightClasses(weightClasses_), depth(depth_), split_mode(split_mode_) {

  };

  virtual ~SplitGen() {
  };

  void generate_mt(int stripe) {
    boost::mt19937 rng_thread(abs(stripe + 1) * std::time(NULL));

    if (Sample::generateSplit(data, &rng_thread, fp, splits[stripe], split_mode, depth)) {
      std::vector<IntIndex> valSet(data.size());
      for (unsigned int l = 0; l < data.size(); ++l) {
        valSet[l].first = data[l]->evalTest(splits[stripe]);
        valSet[l].second = l;
      }
      std::sort(valSet.begin(), valSet.end());
      findThreshold(data, valSet, splits[stripe], &rng_thread);
      splits[stripe].oob = 0;
    } else {
      splits[stripe].threshold = 0;
      splits[stripe].info = boost::numeric::bounds<double>::lowest();
      splits[stripe].gain = boost::numeric::bounds<double>::lowest();
      splits[stripe].oob = boost::numeric::bounds<double>::highest();
    }
  }

  void generate() {
    int num_treads = boost::thread::hardware_concurrency();
    boost::thread_pool::executor e(num_treads);
    for (int stripe = 0; stripe < static_cast<int>(splits.size()); stripe++) {
      e.submit(boost::bind(&SplitGen::generate_mt, this, stripe));
    }
    e.join_all();
  };

  static void splitVec(const std::vector<Sample*>& data,
      const std::vector<IntIndex>& valSet, std::vector<Sample*>& setA,
      std::vector<Sample*>& setB, int threshold, int margin = 0) {

    // search largest value such that val<t
    std::vector<IntIndex>::const_iterator it_first, it_second;

    it_first = lower_bound(valSet.begin(), valSet.end(), threshold - margin, less_than());
    if (margin == 0)
      it_second = it_first;
    else
      it_second = lower_bound(valSet.begin(), valSet.end(), threshold + margin, less_than());

    // Split training data into two sets A,B accroding to threshold t
    setA.resize(it_second - valSet.begin());
    setB.resize(valSet.end() - it_first);

    std::vector<IntIndex>::const_iterator it = valSet.begin();
    typename std::vector<Sample*>::iterator itSample;
    for (itSample = setA.begin(); itSample < setA.end(); ++itSample, ++it)
      (*itSample) = data[it->second];

    it = it_first;
    for (itSample = setB.begin(); itSample < setB.end(); ++itSample, ++it)
      (*itSample) = data[it->second];

  };

  static void splitVec(const std::vector<Sample*>& data,
      const std::vector<IntIndex>& valSet,
      std::vector<std::vector<Sample*> >& sets,
      int threshold, int margin) {

    // search largest value such that val<t
    std::vector<IntIndex>::const_iterator it_first, it_second;

    it_first = lower_bound(valSet.begin(), valSet.end(), threshold - margin, less_than());
    if (margin == 0)
      it_second = it_first;
    else
      it_second = lower_bound(valSet.begin(), valSet.end(), threshold + margin, less_than());

    if (it_first == it_second) // no intersection between the two thresholds
        {
      std::vector<IntIndex>::const_iterator it = it_first;

      sets.resize(2);
      // Split training data into two sets A,B accroding to threshold t
      sets[0].resize(it - valSet.begin());
      sets[1].resize(data.size() - sets[0].size());

      it = valSet.begin();
      typename std::vector<Sample*>::iterator itSample;
      for (itSample = sets[0].begin(); itSample < sets[0].end(); ++itSample, ++it)
        (*itSample) = data[it->second];

      it = valSet.begin() + sets[0].size();
      for (itSample = sets[1].begin(); itSample < sets[1].end(); ++itSample, ++it)
        (*itSample) = data[it->second];

      assert( (sets[0].size() + sets[1].size()) == data.size());

    } else {

      sets.resize(3);
      // Split training data into two sets A,B accroding to threshold t
      sets[0].resize(it_first - valSet.begin());
      sets[1].resize(it_second - it_first);
      sets[2].resize(valSet.end() - it_second);

      std::vector<IntIndex>::const_iterator it = valSet.begin();
      typename std::vector<Sample*>::iterator itSample;
      for (itSample = sets[0].begin(); itSample < sets[0].end(); ++itSample, ++it)
        (*itSample) = data[it->second];

      it = valSet.begin() + sets[0].size();
      for (itSample = sets[1].begin(); itSample < sets[1].end(); ++itSample, ++it)
        (*itSample) = data[it->second];

      it = valSet.begin() + sets[0].size() + sets[1].size();

      for (itSample = sets[2].begin(); itSample < sets[2].end(); ++itSample, ++it)
        (*itSample) = data[it->second];

      assert( (sets[0].size() + sets[1].size() + sets[2].size()) == data.size());

    }

  };

private:

  void findThreshold(const std::vector<Sample*>& data,
      const std::vector<IntIndex>& valSet,
      Split& split, boost::mt19937* rng_) const {
    split.gain = boost::numeric::bounds<double>::lowest();
    split.info = boost::numeric::bounds<double>::lowest();

    int min_Val = valSet.front().first;
    int max_val = valSet.back().first;
    int valueRange = max_val - min_Val;

    if (valueRange > 0) {
//            double info = Sample::entropie( data, weightClasses, split_mode);

      int nThreshlds = split.num_thresholds;
      bool use_margin = false;
      if (use_margin)
        nThreshlds = 20;

      // Find best threshold
      boost::uniform_int<> dist_tr(0, valueRange);
      boost::variate_generator<boost::mt19937&, boost::uniform_int<> > rand_tr(*rng_, dist_tr);

      int m = std::min(abs(min_Val), abs(max_val));
      if (m <= 0)
        m = 1;
      boost::uniform_int<> dist_margin(0, m);
      boost::variate_generator<boost::mt19937&, boost::uniform_int<> > rand_margin(*rng, dist_margin);
      for (int j = 0; j < nThreshlds; ++j) {
        // Generate some random thresholds
        int tr = rand_tr() + min_Val;
        int margin = 0;

        std::vector<std::vector<Sample*> > sets;
        if (use_margin)
          margin = rand_margin();

        splitVec(data, valSet, sets, tr, margin);

        unsigned int min = 2;
        if (sets[0].size() < min or sets[1].size() < min)
          continue;

        double infoNew = Sample::evalSplit(sets[0], sets[1], weightClasses, split_mode, depth);

        if (infoNew > split.info) {
          split.threshold = tr;
          split.info = infoNew;
          split.gain = infoNew;
          split.margin = margin;
        }
      }
    }
  }

  const std::vector<Sample*>& data;
  std::vector<Split>& splits;
  boost::mt19937* rng;
  ForestParam fp;
  const std::vector<float>& weightClasses;
  float depth;
  float split_mode;
};

#endif /* SPLITGEN_H_ */
