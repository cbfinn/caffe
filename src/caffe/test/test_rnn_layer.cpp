#include <cstring>
#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/sequence_layers.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {

template <typename TypeParam>
class RNNLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  RNNLayerTest() : num_output_(7) {
    blob_bottom_vec_.push_back(&blob_bottom_);
    blob_bottom_vec_.push_back(&blob_bottom_flush_);
    blob_top_vec_.push_back(&blob_top_);

    ReshapeBlobs(1, 3);

    layer_param_.mutable_recurrent_param()->set_num_output(num_output_);
    FillerParameter* weight_filler =
        layer_param_.mutable_recurrent_param()->mutable_weight_filler();
    weight_filler->set_type("gaussian");
    weight_filler->set_std(0.2);
    FillerParameter* bias_filler =
        layer_param_.mutable_recurrent_param()->mutable_bias_filler();
    bias_filler->set_type("gaussian");
    bias_filler->set_std(0.1);

    layer_param_.set_phase(TEST);
  }

  void ReshapeBlobs(int num_timesteps, int num_instances) {
    blob_bottom_.Reshape(num_timesteps, num_instances, 3, 2);
    vector<int> shape(2);
    shape[0] = num_timesteps;
    shape[1] = num_instances;
    blob_bottom_flush_.Reshape(shape);

    FillerParameter filler_param;
    filler_param.set_min(-1);
    filler_param.set_max(1);
    UniformFiller<Dtype> filler(filler_param);
    filler.Fill(&blob_bottom_);
  }

  int num_output_;
  LayerParameter layer_param_;
  Blob<Dtype> blob_bottom_;
  Blob<Dtype> blob_bottom_flush_;
  Blob<Dtype> blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(RNNLayerTest, TestDtypesAndDevices);

TYPED_TEST(RNNLayerTest, TestSetUp) {
  typedef typename TypeParam::Dtype Dtype;
  RNNLayer<Dtype> layer(this->layer_param_);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  vector<int> expected_top_shape = this->blob_bottom_.shape();
  expected_top_shape.resize(3);
  expected_top_shape[2] = this->num_output_;
  EXPECT_TRUE(this->blob_top_.shape() == expected_top_shape);
}

TYPED_TEST(RNNLayerTest, TestForward) {
  typedef typename TypeParam::Dtype Dtype;
  const int kNumTimesteps = 3;
  const int num = this->blob_bottom_.shape(1);
  this->ReshapeBlobs(kNumTimesteps, num);

  // Fill the flush blob with <0, 1, 1, ..., 1>,
  // indicating a sequence that begins at the first timestep
  // then continues for the rest of the sequence.
  for (int t = 0; t < kNumTimesteps; ++t) {
    for (int n = 0; n < num; ++n) {
      this->blob_bottom_flush_.mutable_cpu_data()[t * num + n] = t > 0;
    }
  }

  // Process the full sequence in a single batch.
  FillerParameter filler_param;
  filler_param.set_mean(0);
  filler_param.set_std(1);
  GaussianFiller<Dtype> sequence_filler(filler_param);
  sequence_filler.Fill(&this->blob_bottom_);
  shared_ptr<RNNLayer<Dtype> > layer(new RNNLayer<Dtype>(this->layer_param_));
  Caffe::set_random_seed(1701);
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  LOG(INFO) << "Calling forward for full sequence RNN";
  layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);

  // Copy the inputs and outputs to reuse/check them later.
  Blob<Dtype> bottom_copy(this->blob_bottom_.shape());
  bottom_copy.CopyFrom(this->blob_bottom_);
  Blob<Dtype> top_copy(this->blob_top_.shape());
  top_copy.CopyFrom(this->blob_top_);

  // Process the batch one timestep at a time;
  // check that we get the same result.
  this->ReshapeBlobs(1, num);
  layer.reset(new RNNLayer<Dtype>(this->layer_param_));
  Caffe::set_random_seed(1701);
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  const int bottom_count = this->blob_bottom_.count();
  const int top_count = this->blob_top_.count();
  const Dtype kEpsilon = 1e-5;
  for (int t = 0; t < kNumTimesteps; ++t) {
    caffe_copy(bottom_count, bottom_copy.cpu_data() + t * bottom_count,
               this->blob_bottom_.mutable_cpu_data());
    for (int n = 0; n < num; ++n) {
      this->blob_bottom_flush_.mutable_cpu_data()[n] = t > 0;
    }
    LOG(INFO) << "Calling forward for RNN timestep " << t;
    layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
    for (int i = 0; i < top_count; ++i) {
      ASSERT_LT(t * top_count + i, top_copy.count());
      EXPECT_NEAR(this->blob_top_.cpu_data()[i],
                  top_copy.cpu_data()[t * top_count + i], kEpsilon)
         << "t = " << t << "; i = " << i;
    }
  }

  // Process the batch one timestep at a time with all flush blobs set to 0.
  // Check that we get a different result, except in the first timestep.
  Caffe::set_random_seed(1701);
  layer.reset(new RNNLayer<Dtype>(this->layer_param_));
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  for (int t = 0; t < kNumTimesteps; ++t) {
    caffe_copy(bottom_count, bottom_copy.cpu_data() + t * bottom_count,
               this->blob_bottom_.mutable_cpu_data());
    for (int n = 0; n < num; ++n) {
      this->blob_bottom_flush_.mutable_cpu_data()[n] = 0;
    }
    LOG(INFO) << "Calling forward for RNN timestep " << t;
    layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
    for (int i = 0; i < top_count; ++i) {
      if (t == 0) {
        EXPECT_NEAR(this->blob_top_.cpu_data()[i],
                    top_copy.cpu_data()[t * top_count + i], kEpsilon)
           << "t = " << t << "; i = " << i;
      } else {
        EXPECT_NE(this->blob_top_.cpu_data()[i],
                  top_copy.cpu_data()[t * top_count + i])
           << "t = " << t << "; i = " << i;
      }
    }
  }
}

TYPED_TEST(RNNLayerTest, TestGradient) {
  typedef typename TypeParam::Dtype Dtype;
  RNNLayer<Dtype> layer(this->layer_param_);
  GradientChecker<Dtype> checker(1e-2, 1e-3);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_, 0);
}

TYPED_TEST(RNNLayerTest, TestGradientNonZeroFlush) {
  typedef typename TypeParam::Dtype Dtype;
  RNNLayer<Dtype> layer(this->layer_param_);
  GradientChecker<Dtype> checker(1e-2, 1e-3);
  for (int i = 0; i < this->blob_bottom_flush_.count(); ++i) {
    this->blob_bottom_flush_.mutable_cpu_data()[i] = i > 2;
  }
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_, 0);
}

TYPED_TEST(RNNLayerTest, TestGradientNonZeroFlushBufferSize2) {
  typedef typename TypeParam::Dtype Dtype;
  this->ReshapeBlobs(2, 2);
  // fill the values
  FillerParameter filler_param;
  UniformFiller<Dtype> filler(filler_param);
  filler.Fill(&this->blob_bottom_);
  RNNLayer<Dtype> layer(this->layer_param_);
  GradientChecker<Dtype> checker(1e-2, 1e-3);
  for (int i = 0; i < this->blob_bottom_flush_.count(); ++i) {
    this->blob_bottom_flush_.mutable_cpu_data()[i] = i > 2;
  }
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_, 0);
}

TYPED_TEST(RNNLayerTest, TestNumHiddenShape) {
  // Check number of parameters with num_hidden setting.

  typedef typename TypeParam::Dtype Dtype;
  const int nout = 11;
  const int nhid = 7;
  const int nin = 6;

  LayerParameter new_params(this->layer_param_);
  new_params.mutable_rnn_param()->set_num_hidden(nhid);
  new_params.mutable_recurrent_param()->set_num_output(nout);
  RNNLayer<Dtype> layer(new_params);

  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  vector<int> expected_top_shape = this->blob_bottom_.shape();
  expected_top_shape.resize(3);
  expected_top_shape[2] = nout;
  EXPECT_TRUE(this->blob_top_.shape() == expected_top_shape);

  vector<shared_ptr<Blob<Dtype> > > params = layer.UnrolledNet()->params();
  int total_count = 0;
  for(typename vector<shared_ptr<Blob<Dtype> > >::iterator it=params.begin(); it!=params.end(); ++it){
    shared_ptr<Blob<Dtype> > blob = (*it);
    total_count += blob->count();
  }
  const int expected_count = (nin*nhid+nhid*nhid+nhid)+(nhid*nout+nout);
  EXPECT_EQ(total_count, expected_count);
}

TYPED_TEST(RNNLayerTest, TestGradientWithNumHidden) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter new_params(this->layer_param_);
  new_params.mutable_rnn_param()->set_num_hidden(30);
  RNNLayer<Dtype> layer(new_params);

  GradientChecker<Dtype> checker(1e-2, 1e-3);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_, 0);
}

}  // namespace caffe