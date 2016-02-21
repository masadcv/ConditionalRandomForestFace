/*
 * CRTree.h
 *
 *  Created on: May 3, 2011
 *      Author: mdantone
 */

#ifndef CRTREE_H_
#define CRTREE_H_

#include "tree_node.hpp"
#include "split_gen.hpp"
#include "timing.hpp"
#include <string>
#include <iostream>
#include <fstream>

template<typename Sample>
class Tree {
public:
  typedef typename Sample::Split Split;
  typedef typename Sample::Leaf Leaf;

  Tree() {
    timer.restart();
    lastSavePoint = 0;
  }

  Tree(const std::vector<Sample*>& data, ForestParam fp_, boost::mt19937* rng_, std::string savePath_, Timing jobTimer = Timing()) :
      fp(fp_), rng(rng_), savePath(savePath_), timer(jobTimer) {
    timer.restart();
    lastSavePoint = 0;
    root = new TreeNode<Sample>(0);
    num_nodes = pow(2.0, int(fp.max_d + 1)) - 1;
    i_node = 0;
    i_leaf = 0;

    Sample::calcWeightClasses(class_weights, data);

    std::cout << "Start Training" << std::endl;
    grow(root, data);
    save(savePath);
  };

  //start growing the tree
  void grow(const std::vector<Sample*>& data, Timing jobTimer, boost::mt19937* rng_) {
    rng = rng_;
    timer = jobTimer;
    timer.restart();
    lastSavePoint = timer.elapsed();

    std::cout << int((i_node / num_nodes) * 100) << "% : LOADED TREE " << std::endl;
    if (!isFinished()) {
      i_node = 0;
      i_leaf = 0;
      grow(root, data);
      save(savePath);
    }
  }

  //keep growing
  void grow(TreeNode<Sample>* node, const std::vector<Sample*>& data) {
    int depth = node->getDepth();

    // count element
    int nElements = data.size();
    std::vector<Sample*> setA;
    std::vector<Sample*> setB;
    if (nElements < fp.min_s or depth >= fp.max_d or node->isLeaf()) {
      node->createLeaf(data, class_weights, i_leaf);
      i_node += pow(2.0, int((fp.max_d - depth) + 1)) - 1;
      i_leaf++;
      std::cout << int((i_node / num_nodes) * 100) <<
          "% (" << i_leaf << "): make leaf ( depth: " << depth <<
          ", elements: " << data.size() << ")" << std::endl;
    } else {
      Split bestSplit;
      if (node->hasSplit()) //only in reload mode.
      {
        bestSplit = node->getSplit();
        split(data, bestSplit, setA, setB);
        i_node++;
        std::cout << int((i_node / float(num_nodes)) * 100) << "% : split( depth: " << depth << ", elements: " << nElements << ") ["
            << setA.size() << ", " << setB.size() << "], oob: 0 " << std::endl;
        grow(node->left, setA);
        grow(node->right, setB);

      } else {
        bool testFound = findOptimalSplit(data, bestSplit, setA, setB, depth);
        if (testFound) {

          split(data, bestSplit, setA, setB);
          node->setSplit(bestSplit);

          i_node++;

          TreeNode<Sample>* left = new TreeNode<Sample>(depth + 1);
          node->addLeftChild(left);

          TreeNode<Sample>* right = new TreeNode<Sample>(depth + 1);
          node->addRightChild(right);

          autoSave();
          std::cout << int((i_node / float(num_nodes)) * 100) << "% : split( depth: " << depth << ", elements: " << nElements << ") ["
              << setA.size() << ", " << setB.size() << "]" << std::endl;

          grow(left, setA);
          grow(right, setB);
        } else {
          std::cout << "no valid split found " << std::endl;
          node->createLeaf(data, class_weights, i_leaf);
          i_leaf++;
          i_node += (int) pow(2.0, int((fp.max_d - depth) + 1)) - 1;
          std::cout << int((i_node / float(num_nodes)) * 100) << "% (" << i_leaf << "): make leaf ( depth: " << depth << ", elements: "
              << data.size() << ")" << std::endl;
        }
      }
    }
  };

  bool findOptimalSplit(const std::vector<Sample*>& data,
      Split& best_split, std::vector<Sample*>& set_a,
      std::vector<Sample*>& set_b, int depth) {
    best_split.info = boost::numeric::bounds<double>::lowest();
    best_split.gain = boost::numeric::bounds<double>::lowest();
    best_split.oob = boost::numeric::bounds<double>::highest();
    int num_splits = fp.nTests;

    std::vector<Split> splits(num_splits);

    double timeStamp = timer.elapsed();

    boost::uniform_int<> dist_split(0, 100);
    boost::variate_generator<boost::mt19937&, boost::uniform_int<> > rand_split(*rng, dist_split);
    int split_mode = rand_split();
    SplitGen<Sample> sg(data, splits, rng, fp, class_weights, depth, split_mode);
    sg.generate();

    std::cout << timer.elapsed() << ": for split: " << (timer.elapsed() - timeStamp) << " ("
        << (timer.elapsed() - timeStamp) / float(data.size()) << ") mode: " << split_mode << std::endl;
    for (unsigned i = 0; i < splits.size(); i++) {

      if (splits[i].info > best_split.info) {
        best_split = splits[i];
      }
    }
    if (best_split.info != boost::numeric::bounds<double>::lowest())
      return true;
    return false;
  }

  void split(const std::vector<Sample*>& data, Split& best_split,
      std::vector<Sample*>& set_a, std::vector<Sample*>& set_b) {
    //generate Value for each Patch
    std::vector<IntIndex> valSet(data.size());
    for (unsigned int l = 0; l < data.size(); ++l) {
      valSet[l].first = data[l]->evalTest(best_split);
      valSet[l].second = l;
    }
    std::sort(valSet.begin(), valSet.end());

    SplitGen<Sample>::splitVec(data, valSet, set_a, set_b, best_split.threshold, best_split.margin);
  }

  //sends the sample down the tree and return a pointer to the leaf.
  static void evaluate(const Sample* sample, TreeNode<Sample>* node,
      std::vector<Leaf*>& leafs) {
    if (node->isLeaf())
      leafs.push_back(node->getLeaf());
    else {
      if (node->eval(sample)) {
        evaluate(sample, node->left, leafs);
      } else {
        evaluate(sample, node->right, leafs);
      }
    }

  }

  static void evaluate_mt(const Sample* sample,
      TreeNode<Sample>* node, Leaf** leaf) {
    if (node->isLeaf()) {
      *leaf = node->getLeaf();
    } else {
      if (node->eval(sample)) {
        evaluate_mt(sample, node->left, leaf);
      } else {
        evaluate_mt(sample, node->right, leaf);
      }
    }

  }
  void autoSave() {
    int tStamp = timer.elapsed();
    int saveInterval = 150000;
    //save every 10 minutes
    if ((tStamp - lastSavePoint) > saveInterval) {
      lastSavePoint = timer.elapsed();
      std::cout << timer.elapsed() << ": save at " << lastSavePoint << std::endl;
      save(savePath);
    }

  }

  //saves the tree recursive
  //it can also save unfinished trees
  void save(std::string path) {
    try {
      std::ofstream ofs(path.c_str());
      boost::archive::binary_oarchive oa(ofs);
      oa << this;
      ofs.flush();
      ofs.close();
      std::cout << "saved " << path << std::endl;
    } catch (boost::archive::archive_exception& ex) {
      std::cout << "Archive Exception during serializing:" << std::endl;
      std::cout << ex.what() << std::endl;
      std::cout << "it was tree: " << path << std::endl;
    }
  }

  static bool load(Tree** t, std::string path) {

    //check if file exist
    std::ifstream ifs(path.c_str());
    if (!ifs) {
      std::cout << "Tree not found " << path << std::endl;
      return false;
    }

    //load tree
    try {
      boost::archive::binary_iarchive ia(ifs);
      ia >> *t;
    } catch (boost::archive::archive_exception& ex) {
      std::cout << "Archive Exception during deserializing:" << std::endl;
      std::cout << ex.what() << std::endl;
      std::cout << "it was tree: " << path << std::endl;
    } catch (int e) {
      std::cout << path << "EXCEPTION " << e << std::endl;

    }
    ifs.close();
    return true;
  }

  bool isFinished() {
    if (num_nodes == 0)
      return false;
    return i_node == num_nodes;
  }

  virtual ~Tree() {
    if (root)
      delete root;
  };

  std::vector<float> getClassWeights() {
    return class_weights;
  }

  // root node of the tree
  TreeNode<Sample>* root;

  ForestParam fp;
private:
  boost::mt19937* rng;

  //population throw classes
  std::vector<float> class_weights;

  //for statistic reason
  float num_nodes;
  float i_node;
  int i_leaf;

  // the lastest saving timestamp
  int lastSavePoint;

  // saving path of the trees
  std::string savePath;

  Timing timer;

  friend class boost::serialization::access;
  template<class Archive>
  void serialize(Archive & ar, const unsigned int version) {
    ar & num_nodes;
    ar & i_node;
    ar & fp;
    ar & savePath;
    ar & class_weights;
    ar & root;
  }
};
#endif /* CRTREE_H_ */
