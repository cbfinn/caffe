#include <algorithm>
#include <vector>

#include "caffe/common_layers.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void FixedBatchNormLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  int num = bottom[0]->num();
  int channels = bottom[0]->channels();
  int spatial_dim = bottom[0]->height() * bottom[0]->width();

  //Read global stats
  const Dtype* global_mean = bottom[1]->gpu_data();
  const Dtype* global_var = bottom[2]->gpu_data();
  const Dtype* moving_average_mult = bottom[3]->cpu_data();


  // XXX this should not be here
  Dtype eps = 1e-5;
  
  // elementwise square
  // XXX how does this compare to caffe_gpu_mul?
  caffe_gpu_powx(bottom[0]->count(), bottom_data, Dtype(2),
      temp_.mutable_gpu_data());

  // computes variance using var(X) = E(X^2) - (EX)^2

  // mean of bottom and bottom ** 2


  caffe_copy(mean_.count(),global_mean,mean_.mutable_gpu_data());
  caffe_gpu_scal(mean_.count(),Dtype(1)/moving_average_mult[0],mean_.mutable_gpu_data());
  caffe_copy(variance_.count(),global_var,variance_.mutable_gpu_data());
  caffe_gpu_scal(variance_.count(),Dtype(1)/moving_average_mult[0],variance_.mutable_gpu_data());

  // elementwise square of mean
  caffe_gpu_powx(mean_.count(), mean_.gpu_data(), Dtype(2), temp_.mutable_gpu_data());

  caffe_gpu_sub(mean_.count(), variance_.gpu_data(), temp_.gpu_data(),
      variance_.mutable_gpu_data());  // variance

  caffe_gpu_add_scalar(variance_.count(), eps, variance_.mutable_gpu_data());
  caffe_gpu_powx(variance_.count(), variance_.gpu_data(), Dtype(0.5),
        variance_.mutable_gpu_data());

  // do mean and variance normalization
  // subtract mean
  caffe_copy(bottom[0]->count(), bottom_data, top_data);

  caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num, channels, 1, 1,
        num_sum_.gpu_data(), mean_.gpu_data(), 0., num_by_chans_.mutable_gpu_data());
  caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, channels*num, spatial_dim, 1, -1,
      num_by_chans_.gpu_data(), sum_multiplier_.gpu_data(), 1., top_data);

  caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num, channels, 1, 1,
          num_sum_.gpu_data(), variance_.gpu_data(), 0., num_by_chans_.mutable_gpu_data());
  caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, channels*num, spatial_dim, 1, 1.,
        num_by_chans_.gpu_data(), sum_multiplier_.gpu_data(), 0.,
        temp_.mutable_gpu_data());
  caffe_gpu_div(temp_.count(), top_data, temp_.gpu_data(), top_data);
}

template <typename Dtype>
void FixedBatchNormLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  CHECK(!use_global_stats_);
  const Dtype* top_diff = top[0]->gpu_diff();
  const Dtype* top_data = top[0]->gpu_data();
  Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
  int num = bottom[0]->num();
  int channels = bottom[0]->channels();
  int spatial_dim = bottom[0]->height() * bottom[0]->width();

  for(int i = 0; i<this->blobs_.size(); ++i){
    caffe_gpu_set(this->blobs_[i]->count(),Dtype(0),this->blobs_[i]->mutable_gpu_diff());
  }

  caffe_gpu_mul(temp_.count(), top_data, top_diff, bottom_diff);
  caffe_gpu_gemv<Dtype>(CblasNoTrans, channels*num, spatial_dim, 1.,
    bottom_diff,
    sum_multiplier_.gpu_data(), 0., num_by_chans_.mutable_gpu_data());
  caffe_gpu_gemv<Dtype>(CblasTrans, num, channels, 1.,
    num_by_chans_.gpu_data(),num_sum_.gpu_data(),0., mean_.mutable_gpu_data());

  caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num, channels, 1, 1,
        num_sum_.gpu_data(), mean_.gpu_data(), 0., num_by_chans_.mutable_gpu_data());
  caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, channels*num, spatial_dim, 1, 1.,
        num_by_chans_.gpu_data(), sum_multiplier_.gpu_data(), 0.,
        bottom_diff);
  caffe_gpu_mul(temp_.count(), top_data, bottom_diff, bottom_diff);

  caffe_gpu_gemv<Dtype>(CblasNoTrans, channels*num, spatial_dim, 1.,
      top_diff,
      sum_multiplier_.gpu_data(), 0., num_by_chans_.mutable_gpu_data());
  caffe_gpu_gemv<Dtype>(CblasTrans, num, channels, 1.,
      num_by_chans_.gpu_data(),num_sum_.gpu_data(),0., mean_.mutable_gpu_data());

  caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num, channels, 1, 1,
        num_sum_.gpu_data(), mean_.gpu_data(), 0., num_by_chans_.mutable_gpu_data());
  caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num*channels, spatial_dim, 1, 1.,
        num_by_chans_.gpu_data(), sum_multiplier_.gpu_data(), 1.,
        bottom_diff);

  caffe_gpu_axpby(temp_.count(), Dtype(1), top_diff, Dtype(-1. / (num * spatial_dim)),
      bottom_diff);

  caffe_gpu_div(temp_.count(), bottom_diff, temp_.gpu_data(), bottom_diff);
}

INSTANTIATE_LAYER_GPU_FUNCS(FixedBatchNormLayer);
}  // namespace caffe
