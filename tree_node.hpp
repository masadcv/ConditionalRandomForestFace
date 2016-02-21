/*
 * Tree.h
 *
 *  Created on: May 2, 2011
 *      Author: Matthias Dantone
 */

#ifndef TREE_NODE_H_
#define TREE_NODE_H_

#include <vector>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/serialization/vector.hpp>

struct ForestParam {
  int max_d;
  int min_s;
  int nTests;
  int nTrees;
  int nSamplesPerTree;
  int nPatchesPerSample;
  int faceSize;
  int measuremode;
  int nFeatureChannels;
  float patchSizeRatio;
  std::string treePath;
  std::string imgPath;
  std::string featurePath;
  std::vector<int> features;

  friend class boost::serialization::access;
  template<class Archive>
  void serialize(Archive & ar, const unsigned int version) {
    ar & max_d;
    ar & min_s;
    ar & nTests;
    ar & nTrees;
    ar & nSamplesPerTree;
    ar & nPatchesPerSample;
    ar & faceSize;
    ar & measuremode;
    ar & nFeatureChannels;
    ar & patchSizeRatio;
    ar & treePath;
    ar & imgPath;
    ar & featurePath;
    ar & features;
  }

};

template<typename Sample>
class TreeNode {
public:
  typedef typename Sample::Split Split;
  typedef typename Sample::Leaf Leaf;

  TreeNode() :
      depth(-1), right(NULL), left(NULL), is_leaf(false), has_split(false) {
  }
  ;

  TreeNode(int depth_) :
      depth(depth_), right(NULL), left(NULL), is_leaf(false), has_split(false) {
  }
  ;

  void createLeaf(const std::vector<Sample*>& data, const std::vector<float>& weightClasses, int iLeaf = -1) {

    Sample::makeLeaf(leaf, data, weightClasses, iLeaf);
    is_leaf = true;
    has_split = false;
  }
  ;

  Leaf* getLeaf() {
    return &leaf;
  }

  Split getSplit() {
    return split;
  }

  void setSplit(Split split_) {
    has_split = true;
    is_leaf = false;
    split = split_;
  }
  void setLeaf(Leaf leaf_) {
    is_leaf = true;
    leaf = leaf_;
  }

  void addLeftChild(TreeNode<Sample>* leftChild) {
    left = leftChild;
  }
  ;

  void addRightChild(TreeNode<Sample>* rightChild) {
    right = rightChild;
  }
  ;

  bool eval(const Sample* s) const {
    return s->eval(split);
  }

  bool isLeaf() const {
    return is_leaf;
  }
  ;

  bool hasSplit() const {
    return has_split;
  }
  ;

  int getDepth() {
    return depth;
  }
  ;
  void collectLeafs(std::vector<Leaf*>& leafs) {
    if (!is_leaf) {
      right->collectLeafs(leafs);
      left->collectLeafs(leafs);
    } else {
      leaf.depth = depth;
      leafs.push_back(&leaf);
    }
  }
  ;

  ~TreeNode() {
    if (left)
      delete left;
    if (right)
      delete right;

  }
  ;

  int depth;
  Leaf leaf;
  Split split;

  TreeNode<Sample>* right;
  TreeNode<Sample>* left;
private:

  bool is_leaf;
  bool has_split;

  friend class boost::serialization::access;
  template<class Archive>
  void serialize(Archive & ar, const unsigned int version) {
    ar & depth;
    ar & is_leaf;
    ar & has_split;
    if (has_split) {
      ar & split;
    }
    if (!is_leaf) {
      ar & left;
      ar & right;
    } else {
      ar & leaf;
    }
  }
};

#endif /* TREE_NODE_H_ */
