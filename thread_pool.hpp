/*
 * thread_pool.hpp
 *
 *  Created on: Jul 9, 2012
 *      Author: Matthias Dantone
 */

#ifndef THREAD_POOL_HPP_
#define THREAD_POOL_HPP_

#include <boost/asio.hpp>
#include <boost/bind.hpp>
#include <boost/thread.hpp>

namespace boost {
namespace thread_pool {

using boost::bind;
using boost::thread;
using boost::thread_group;
using boost::asio::io_service;

class executor {
public:
  executor(size_t n = 10) :
      service_(n), worker_(new boost::asio::io_service::work(service_)) {
    for (size_t i = 0; i < n; i++) {
      pool_.create_thread(bind(&io_service::run, &service_));
    }
  }

  ~executor() {
    worker_.reset();
    service_.stop();
    pool_.join_all();
  }

  void join_all() {
    worker_.reset();
    pool_.join_all();
  }
  template<typename F> void submit(F task) {
    service_.post(task);
  }

protected:
  thread_group pool_;
  io_service service_;
  boost::shared_ptr<io_service::work> worker_;

};
}
}

#endif /* THREAD_POOL_HPP_ */
