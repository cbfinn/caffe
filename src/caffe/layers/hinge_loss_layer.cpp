#include <algorithm>
#include <cfloat>
#include <cmath>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void HingeLossLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  vector<int> loss_shape(0);  // Loss layers output a scalar; 0 axes.
  top[0]->Reshape(loss_shape);
  temp_.Reshape(bottom[0]->shape());
}

template <typename Dtype>
void HingeLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  offset_ = this->layer_param_.hinge_loss_param().offset();
  //LOG(INFO) << "hinge loss forward";
  const Dtype* bottom_data = bottom[0]->cpu_data();
  //Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
  Dtype* temp = temp_.mutable_cpu_data();
  int num = bottom[0]->num();
  int count = bottom[0]->count();

  caffe_copy(count, bottom_data, temp);
  //caffe_copy(count, bottom_data, bottom_diff);

  for (int i = 0; i < count; ++i) {
      //if (temp[i] != temp[i]) {
        //LOG(INFO) << "Encountered nan in temp[i], i=" << i << ", temp[i]=" << temp[i];
      //}
      temp[i] = std::max(Dtype(0), temp[i]+offset_);
      //LOG(INFO) << "diff " << i << ", " << temp[i];
  }
  Dtype* loss = top[0]->mutable_cpu_data();
  switch (this->layer_param_.hinge_loss_param().norm()) {
  case HingeLossParameter_Norm_L1:
    loss[0] = caffe_cpu_asum(count, temp) / num;
    break;
  case HingeLossParameter_Norm_L2:
    loss[0] = caffe_cpu_dot(count, temp, temp) / num;
    break;
  default:
    LOG(FATAL) << "Unknown Norm";
  }
}

template <typename Dtype>
void HingeLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    int num = bottom[0]->num();
    int count = bottom[0]->count();
    caffe_copy(count, temp_.cpu_data(), bottom_diff);

    const Dtype loss_weight = top[0]->cpu_diff()[0];
    switch (this->layer_param_.hinge_loss_param().norm()) {
    case HingeLossParameter_Norm_L1:
      caffe_cpu_sign(count, bottom_diff, bottom_diff);
      caffe_scal(count, loss_weight / num, bottom_diff);
      break;
    case HingeLossParameter_Norm_L2:
      caffe_scal(count, loss_weight * 2 / num, bottom_diff);
      break;
    default:
      LOG(FATAL) << "Unknown Norm";
    }
  }
}

INSTANTIATE_CLASS(HingeLossLayer);
REGISTER_LAYER_CLASS(HingeLoss);

}  // namespace caffe
