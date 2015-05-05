#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

// bottom[0] is the demos, bottom[1] is the samples. Each are N x T
template <typename Dtype>
void IOCLossLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  // LossLayer<Dtype>::Reshape(bottom, top);
  vector<int> loss_shape(0);  // Loss layers output a scalar; 0 axes.
  top[0]->Reshape(loss_shape);
  T_ = bottom[0]->channels();
  nd_ = bottom[0]->num();
  ns_ = bottom[1]->num();
  CHECK_EQ(bottom[0]->channels(), bottom[1]->channels());
  demo_counts_.Reshape(nd_, 1, 1, 1);
  sample_counts_.Reshape(ns_, 1, 1, 1);
}

template <typename Dtype>
void IOCLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  Dtype loss(0.0);
  Dtype* dc = demo_counts_.mutable_cpu_data();
  Dtype* sc = sample_counts_.mutable_cpu_data();


  // Sum over time and compute max value for safe logsum.
  for (int i = 0; i < nd_; ++i) {
    dc[i] = 0.0;
    for (int t = 0; t < T_; ++t) {
      dc[i] += bottom[0]->data_at(i,t,0,0);
    }
    loss += dc[i];
  }
  // Divide by number of demos.
  loss /= (Dtype)nd_;

  max_val_ = -dc[0];
  for (int i = 0; i < ns_; ++i) {
    sc[i] = 0.0;
    for (int t = 0; t < T_; ++t) {
      sc[i] += bottom[1]->data_at(i,t,0,0);
    }
    if (-sc[i] > max_val_) max_val_ = -sc[i];
  }

  for (int i = 0; i < nd_; ++i) {
    if (-dc[i] > max_val_) max_val_ = -dc[i];
  }

  // Do a safe log-sum-exp operation.
  for (int i = 0; i < nd_; ++i) {
    dc[i] = -dc[i] - max_val_;
  }
  for (int i = 0; i < ns_; ++i) {
    sc[i] = -sc[i] - max_val_;
  }

  caffe_exp(nd_, dc, dc);
  caffe_exp(ns_, sc, sc);

  partition_ = 0.0;
  for (int i = 0; i < nd_; ++i) partition_ += dc[i];
  for (int i = 0; i < ns_; ++i) partition_ += sc[i];

  loss += log(partition_) + max_val_;

  top[0]->mutable_cpu_data()[0] = loss;
}

template <typename Dtype>
void IOCLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {

  const Dtype loss_weight = top[0]->cpu_diff()[0];
  const Dtype* dc = demo_counts_.cpu_data();
  const Dtype* sc = sample_counts_.cpu_data();
  // Compute gradient w.r.t. demos
  Dtype* demo_bottom_diff = bottom[0]->mutable_cpu_diff();
  Dtype* sample_bottom_diff = bottom[1]->mutable_cpu_diff();

  for (int i = 0; i < nd_; ++i) {
    for (int t = 0; t < T_; ++t) {
      demo_bottom_diff[i*T_ + t] = (1.0 / (Dtype)nd_) - (dc[i] / partition_);
    }
  }

  for (int i = 0; i < ns_; ++i) {
    for (int t = 0; t < T_; ++t) {
      sample_bottom_diff[i*T_ + t] = - sc[i] / partition_;
    }
  }

  caffe_scal(nd_*T_, loss_weight, demo_bottom_diff);
  caffe_scal(ns_*T_, loss_weight, sample_bottom_diff);
}

INSTANTIATE_CLASS(IOCLossLayer);
REGISTER_LAYER_CLASS(IOCLoss);
}  // namespace caffe
