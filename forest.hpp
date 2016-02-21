/*
 * CRForest.h
 *
 *  Created on: May 4, 2011
 *      Author: Matthias Dantone
 */

#ifndef CRFOREST_H_
#define CRFOREST_H_

#include "tree.hpp"

template<typename Sample>
class Forest {
public:
  typedef typename Sample::Split Split;
  typedef typename Sample::Leaf Leaf;
  Forest() {
  };
  Forest(ForestParam tp) :
      param(tp) {
  };

  Forest(const std::vector<Sample*> data, ForestParam tp, boost::mt19937* rng) {
    for (int i = 0; i < tp.nTrees; i++) {
      Tree<Sample>* tree = new Tree<Sample>(data, tp, rng);
      trees.push_back(tree);
    }
  };

  void addTree(Tree<Sample>* t) {
    trees.push_back(t);
  }

  //sends the Sample down the tree
  void evaluate(const Sample* f, std::vector<Leaf*>& leafs) const {
    for (unsigned int i = 0; i < trees.size(); i++)
      trees[i]->evaluate(f, trees[i]->root, leafs);
  }

  void evaluate_mt(const Sample* f, Leaf** leafs) const {
    for (unsigned int i = 0; i < trees.size(); i++) {
      trees[i]->evaluate_mt(f, trees[i]->root, leafs);
      leafs++;
    }
  }

  //stores the tree
  void save(std::string url, int offset = 0) {
    for (unsigned int i = 0; i < trees.size(); i++) {

      char buffer[200];
      sprintf(buffer, "%s%03d.txt", url.c_str(), i + offset);

      std::string path = buffer;
      trees[i]->save(buffer);
    }
  }

  void load(std::string url, ForestParam tp, int max_trees = -1) {
    param = tp;
    if (max_trees == -1)
      max_trees = tp.nTrees;
    std::cout << tp.nTrees << " to load." << std::endl;
    for (int i = 0; i < tp.nTrees; i++) {
      if (static_cast<int>(trees.size()) > max_trees)
        continue;
      char buffer[200];
      sprintf(buffer, "%s%03d.txt", url.c_str(), i);
      std::string tree_path = buffer;
      load_tree(tree_path, trees);
    }
    std::cout << trees.size() << " trees loaded" << std::endl;
  }

  static bool load_tree(std::string url, std::vector<Tree<Sample>*>& trees) {

    Tree<Sample>* tree;
    Tree<Sample>::load(&tree, url);

    if (tree->isFinished()) {
      trees.push_back(tree);
    } else {
      delete tree;
      return false;
    }
    return true;
  }

  ForestParam getParam() const {
    return param;
  }

  void setParam(ForestParam fp) {
    param = fp;
  }

  std::vector<float> getClassWeights() {
    return trees[0]->getClassWeights();
  }

  void getAllLeafs(std::vector<std::vector<Leaf*> >& leafs) {
    leafs.resize(trees.size());
    for (unsigned int i = 0; i < trees.size(); i++)
      trees[i]->root->collectLeafs(leafs[i]);

  }
  std::vector<Tree<Sample>*> trees;

private:
  ForestParam param;
  friend class boost::serialization::access;
  template<class Archive>
  void serialize(Archive & ar, const unsigned int version) {
    ar & trees;
  }

};

#endif /* CRFOREST_H_ */
