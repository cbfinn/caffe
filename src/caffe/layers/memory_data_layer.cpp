#include <opencv2/core/core.hpp>

#include <vector>

#include "caffe/data_layers.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"

namespace caffe {

template <typename Dtype>
void MemoryDataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
     const vector<Blob<Dtype>*>& top) {
  const MemoryDataParameter param = this->layer_param_.memory_data_param();
  batch_size_ = param.input_shapes(0).dim(0);
  for (int i = 0; i < param.input_shapes_size(); ++i) {
    top[i]->Reshape(param.input_shapes(i));
    CHECK_EQ(batch_size_, param.input_shapes(i).dim(0)) <<
      "Inconsistent batch sizes across blobs in memory data layer";
    channels_.push_back(param.input_shapes(i).dim(1));
    height_.push_back(param.input_shapes(i).dim(2));
    width_.push_back(param.input_shapes(i).dim(3));
    size_.push_back(channels_[i] * height_[i] * width_[i]);
    // added_data_[i].Reshape(batch_size_, channels_[i], height_[i], width_[i]);
    data_.push_back(NULL);
  }
}

/*
template <typename Dtype>
void MemoryDataLayer<Dtype>::AddDatumVector(const vector<Datum>& datum_vector) {
  CHECK(!has_new_data_) <<
      "Can't add data until current data has been consumed.";
  size_t num = datum_vector.size();
  CHECK_GT(num, 0) << "There is no datum to add.";
  CHECK_EQ(num % batch_size_, 0) <<
      "The added data must be a multiple of the batch size.";
  added_data_.Reshape(num, channels_, height_, width_);
  added_label_.Reshape(num, 1, 1, 1);
  // Apply data transformations (mirror, scale, crop...)
  this->data_transformer_->Transform(datum_vector, &added_data_);
  // Copy Labels
  Dtype* top_label = added_label_.mutable_cpu_data();
  for (int item_id = 0; item_id < num; ++item_id) {
    top_label[item_id] = datum_vector[item_id].label();
  }
  // num_images == batch_size_
  Dtype* top_data = added_data_.mutable_cpu_data();
  Reset(top_data, top_label, num);
  has_new_data_ = true;
}
*/

/*
template <typename Dtype>
void MemoryDataLayer<Dtype>::AddMatVector(const vector<cv::Mat>& mat_vector,
    const vector<int>& labels) {
  size_t num = mat_vector.size();
  CHECK(!has_new_data_) <<
      "Can't add mat until current data has been consumed.";
  CHECK_GT(num, 0) << "There is no mat to add";
  CHECK_EQ(num % batch_size_, 0) <<
      "The added data must be a multiple of the batch size.";
  added_data_.Reshape(num, channels_, height_, width_);
  added_label_.Reshape(num, 1, 1, 1);
  // Apply data transformations (mirror, scale, crop...)
  this->data_transformer_->Transform(mat_vector, &added_data_);
  // Copy Labels
  Dtype* top_label = added_label_.mutable_cpu_data();
  for (int item_id = 0; item_id < num; ++item_id) {
    top_label[item_id] = labels[item_id];
  }
  // num_images == batch_size_
  Dtype* top_data = added_data_.mutable_cpu_data();
  Reset(top_data, top_label, num);
  has_new_data_ = true;
}
*/


template <typename Dtype>
void MemoryDataLayer<Dtype>::Reset(vector<Dtype*> data, int n) {
  CHECK_EQ(n % batch_size_, 0) << "n must be a multiple of batch size";
  for (int i = 0; i < data.size(); ++i) {
    CHECK(data[i]);
    data_[i] = data[i];
  }
  // Warn with transformation parameters since a memory array is meant to
  // be generic and no transformations are done with Reset().
  if (this->layer_param_.has_transform_param()) {
    LOG(WARNING) << this->type() << " does not transform array data on Reset()";
  }
  n_ = n;
  pos_ = 0;
}


template <typename Dtype>
void MemoryDataLayer<Dtype>::SetBatchSize(int new_batch_size) {
  CHECK_GT(new_batch_size, 0) << "new batch size must be positive.";
  if (!shuffle_)
    CHECK_EQ(n_ % new_batch_size, 0) << "n must be multiple of batch size";
  batch_size_ = new_batch_size;
  pos_ = 0;
}


/*
template <typename Dtype>
void MemoryDataLayer<Dtype>::Reset(Dtype* data, Dtype* labels, int n) {
  CHECK(data);
  CHECK(labels);
  CHECK_EQ(n % batch_size_, 0) << "n must be a multiple of batch size";
  // Warn with transformation parameters since a memory array is meant to
  // be generic and no transformations are done with Reset().
  if (this->layer_param_.has_transform_param()) {
    LOG(WARNING) << this->type() << " does not transform array data on Reset()";
  }
  data_ = data;
  labels_ = labels;
  n_ = n;
  pos_ = 0;
}
*/

/*
template <typename Dtype>
void MemoryDataLayer<Dtype>::set_batch_size(int new_size) {
  CHECK(!has_new_data_) <<
      "Can't change batch_size until current data has been consumed.";
  batch_size_ = new_size;
  added_data_.Reshape(batch_size_, channels_, height_, width_);
  added_label_.Reshape(batch_size_, 1, 1, 1);
}
*/

template <typename Dtype>
void MemoryDataLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  for (int i = 0; i < top.size(); ++i) {
    CHECK(data_[i]) << "MemoryDataLayer needs to be initalized by calling Reset";
    top[i]->Reshape(batch_size_, channels_[i], height_[i], width_[i]);
    top[i]->set_cpu_data(data_[i] + pos_ * size_[i]);
  }
  pos_ = (pos_ + batch_size_) % n_;
  //if (pos_ == 0)
  //has_new_data_ = false;
}

INSTANTIATE_CLASS(MemoryDataLayer);
REGISTER_LAYER_CLASS(MemoryData);

}  // namespace caffe
