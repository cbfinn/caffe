// Fillers are random number generators that fills a blob using the specified
// algorithm. The expectation is that they are only going to be used during
// initialization time and will not involve any GPUs.

#ifndef CAFFE_FILLER_HPP
#define CAFFE_FILLER_HPP

#include <string>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/syncedmem.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

/// @brief Fills a Blob with constant or randomly-generated data.
template <typename Dtype>
class Filler {
 public:
  explicit Filler(const FillerParameter& param) : filler_param_(param) {}
  virtual ~Filler() {}
  virtual void Fill(Blob<Dtype>* blob) = 0;
 protected:
  FillerParameter filler_param_;
};  // class Filler


/// @brief Fills a Blob with constant values @f$ x = 0 @f$.
template <typename Dtype>
class ConstantFiller : public Filler<Dtype> {
 public:
  explicit ConstantFiller(const FillerParameter& param)
      : Filler<Dtype>(param) {}
  virtual void Fill(Blob<Dtype>* blob) {
    Dtype* data = blob->mutable_cpu_data();
    const int count = blob->count();
    const Dtype value = this->filler_param_.value();
    CHECK(count);
    for (int i = 0; i < count; ++i) {
      data[i] = value;
    }
    CHECK_EQ(this->filler_param_.sparse(), -1)
         << "Sparsity not supported by this Filler.";
  }
};

/// @brief Fills a Blob with uniformly distributed values @f$ x\sim U(a, b) @f$.
template <typename Dtype>
class UniformFiller : public Filler<Dtype> {
 public:
  explicit UniformFiller(const FillerParameter& param)
      : Filler<Dtype>(param) {}
  virtual void Fill(Blob<Dtype>* blob) {
    CHECK(blob->count());
    caffe_rng_uniform<Dtype>(blob->count(), Dtype(this->filler_param_.min()),
        Dtype(this->filler_param_.max()), blob->mutable_cpu_data());
    CHECK_EQ(this->filler_param_.sparse(), -1)
         << "Sparsity not supported by this Filler.";
  }
};

/// @brief Fills a Blob with Gaussian-distributed values @f$ x = a @f$.
template <typename Dtype>
class GaussianFiller : public Filler<Dtype> {
 public:
  explicit GaussianFiller(const FillerParameter& param)
      : Filler<Dtype>(param) {}
  virtual void Fill(Blob<Dtype>* blob) {
    Dtype* data = blob->mutable_cpu_data();
    CHECK(blob->count());
    caffe_rng_gaussian<Dtype>(blob->count(), Dtype(this->filler_param_.mean()),
        Dtype(this->filler_param_.std()), blob->mutable_cpu_data());
    int sparse = this->filler_param_.sparse();
    CHECK_GE(sparse, -1);
    if (sparse >= 0) {
      // Sparse initialization is implemented for "weight" blobs; i.e. matrices.
      // These have num == channels == 1; width is number of inputs; height is
      // number of outputs.  The 'sparse' variable specifies the mean number
      // of non-zero input weights for a given output.
      CHECK_GE(blob->num_axes(), 1);
      const int num_outputs = blob->shape(0);
      Dtype non_zero_probability = Dtype(sparse) / Dtype(num_outputs);
      rand_vec_.reset(new SyncedMemory(blob->count() * sizeof(int)));
      int* mask = reinterpret_cast<int*>(rand_vec_->mutable_cpu_data());
      caffe_rng_bernoulli(blob->count(), non_zero_probability, mask);
      for (int i = 0; i < blob->count(); ++i) {
        data[i] *= mask[i];
      }
    }
  }

 protected:
  shared_ptr<SyncedMemory> rand_vec_;
};

/** @brief Fills a Blob with values @f$ x \in [0, 1] @f$
 *         such that @f$ \forall i \sum_j x_{ij} = 1 @f$.
 */
template <typename Dtype>
class PositiveUnitballFiller : public Filler<Dtype> {
 public:
  explicit PositiveUnitballFiller(const FillerParameter& param)
      : Filler<Dtype>(param) {}
  virtual void Fill(Blob<Dtype>* blob) {
    Dtype* data = blob->mutable_cpu_data();
    DCHECK(blob->count());
    caffe_rng_uniform<Dtype>(blob->count(), 0, 1, blob->mutable_cpu_data());
    // We expect the filler to not be called very frequently, so we will
    // just use a simple implementation
    int dim = blob->count() / blob->num();
    CHECK(dim);
    for (int i = 0; i < blob->num(); ++i) {
      Dtype sum = 0;
      for (int j = 0; j < dim; ++j) {
        sum += data[i * dim + j];
      }
      for (int j = 0; j < dim; ++j) {
        data[i * dim + j] /= sum;
      }
    }
    CHECK_EQ(this->filler_param_.sparse(), -1)
         << "Sparsity not supported by this Filler.";
  }
};

/**
 * @brief Fills a Blob with values @f$ x \sim U(-a, +a) @f$ where @f$ a @f$
 *        is set inversely proportional to the number of incoming nodes.
 *
 * A Filler based on the paper [Bengio and Glorot 2010]: Understanding
 * the difficulty of training deep feedforward neuralnetworks, but does not
 * use the fan_out value.
 *
 * It fills the incoming matrix by randomly sampling uniform data from
 * [-scale, scale] where scale = sqrt(3 / fan_in) where fan_in is the number
 * of input nodes. You should make sure the input blob has shape (num, a, b, c)
 * where a * b * c = fan_in.
 *
 * TODO(dox): make notation in above comment consistent with rest & use LaTeX.
 */
template <typename Dtype>
class XavierFiller : public Filler<Dtype> {
 public:
  explicit XavierFiller(const FillerParameter& param)
      : Filler<Dtype>(param) {}
  virtual void Fill(Blob<Dtype>* blob) {
    CHECK(blob->count());
    int fan_in = blob->count() / blob->num();
    Dtype scale = sqrt(Dtype(3) / fan_in);
    caffe_rng_uniform<Dtype>(blob->count(), -scale, scale,
        blob->mutable_cpu_data());
    CHECK_EQ(this->filler_param_.sparse(), -1)
         << "Sparsity not supported by this Filler.";
  }
};

/** @brief Fills a Blob with values @f$ x \in {0,x,y} @f$
 * such that the dummy data values are equal to the pixel location.
 */
template <typename Dtype>
class ExpectationDataFiller : public Filler<Dtype> {
 public:
  explicit ExpectationDataFiller(const FillerParameter& param)
      : Filler<Dtype>(param) {}
  virtual void Fill(Blob<Dtype>* blob) {
    Dtype* data = blob->mutable_cpu_data();
    DCHECK(blob->count());
    int width = blob->shape(-1);
    int height = blob->shape(-2);
    const string& option = this->filler_param_.expectation_option();

    // x means E[x], y means E[y]
    if (option != "x" && option != "y") {
      LOG(FATAL) << "Only x or y allowed as expectation data filler, not " << option;
    }

    // Iterate over all channels.
    for (int c = 0; c < blob->count(0,blob->CanonicalAxisIndex(-2)); ++c) {
      for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
          int offset = c*width*height + y*width + x;
          Dtype* weight_ptr = data + offset;
          if (option == "y") {
            weight_ptr[0] = 2*(Dtype(y) / Dtype(height-1) - Dtype(0.5));
          } else {
            weight_ptr[0] = 2*(Dtype(x) / Dtype(width-1) - Dtype(0.5));
          }
        }
      }
    }

    // We expect the filler to not be called very frequently, so we will
    // just use a simple implementation
    CHECK_EQ(this->filler_param_.sparse(), -1)
         << "Sparsity not supported by this Filler.";
  }
};


/** @brief Fills a Blob with values @f$ x \in {0,x,y} @f$
 * such that the output of the inner product layer is the weighted average
 * x and y coordinate for each input channel.
 */
template <typename Dtype>
class ExpectationFiller : public Filler<Dtype> {
 public:
  explicit ExpectationFiller(const FillerParameter& param)
      : Filler<Dtype>(param) {}
  virtual void Fill(Blob<Dtype>* blob) {
    Dtype* data = blob->mutable_cpu_data();
    DCHECK(blob->count());
    int width = this->filler_param_.width();
    int height = this->filler_param_.height();
    const string& option = this->filler_param_.expectation_option();

    // paramater blob should be 1x1xhxw where h is determined by number of
    // channels of the input and w is the dim of the input.
    CHECK_EQ(blob->shape(1), width * height) << blob->shape(1) << " != " << width*height;
    // x means E[x], y means E[y]
    // xy means E[x] and E[y] both output
    // -x^2y^2 means -E[x^2] and -E[y^2] both output
    if (option == "xy" || option == "-x^2y^2") {
      // Output dimension of inner product layer should be either 2 or 1
      CHECK_EQ(blob->shape(0), 2) << "Output point dimension: " << blob->shape(0);
    } else if (option == "x" || option == "y") {
      CHECK_EQ(blob->shape(0), 1) << "Output point dimension: " << blob->shape(0);
    } else {
      LOG(FATAL) << "Unknown expectation filler policy: " << option;
    }

    for (int x = 0; x < width; ++x) {
      for (int y = 0; y < height; ++y) {
        // Iterature over all outputs.
        for (int k = 0; k < blob->shape(0); ++k) {
          int offset = y*width + x;
          Dtype* weight_ptr = data + k*width*height + offset;
          if (k == 0) {
            if (option == "y") {
              weight_ptr[0] = 2*(Dtype(y) / Dtype(height-1) - Dtype(0.5));
            } else if (option == "-x^2y^2") {
              weight_ptr[0] = - pow(2*(Dtype(x) / Dtype(width-1) - Dtype(0.5)), 2);
            } else {  // "x" or "xy"
              weight_ptr[0] = 2*(Dtype(x) / Dtype(width-1) - Dtype(0.5));
            }
          } else if (k==1) {
            if (option == "xy") {
              weight_ptr[0] = 2*(Dtype(y) / Dtype(height-1) - Dtype(0.5));
            } else {
              weight_ptr[0] = - pow(2*(Dtype(y) / Dtype(height-1) - Dtype(0.5)), 2);
            }
          } else {
            LOG(FATAL) << "More than 2 output???";
          }
        }
      }
    }

    // We expect the filler to not be called very frequently, so we will
    // just use a simple implementation
    CHECK_EQ(this->filler_param_.sparse(), -1)
         << "Sparsity not supported by this Filler.";
  }
};



/**
 * @brief Get a specific filler from the specification given in FillerParameter.
 *
 * Ideally this would be replaced by a factory pattern, but we will leave it
 * this way for now.
 */
template <typename Dtype>
Filler<Dtype>* GetFiller(const FillerParameter& param) {
  const std::string& type = param.type();
  if (type == "constant") {
    return new ConstantFiller<Dtype>(param);
  } else if (type == "gaussian") {
    return new GaussianFiller<Dtype>(param);
  } else if (type == "positive_unitball") {
    return new PositiveUnitballFiller<Dtype>(param);
  } else if (type == "uniform") {
    return new UniformFiller<Dtype>(param);
  } else if (type == "xavier") {
    return new XavierFiller<Dtype>(param);
  } else if (type == "expectation") {
    return new ExpectationFiller<Dtype>(param);
  } else if (type == "expectation_data") {
    return new ExpectationDataFiller<Dtype>(param);
  } else {
    CHECK(false) << "Unknown filler name: " << param.type();
  }
  return (Filler<Dtype>*)(NULL);
}

}  // namespace caffe

#endif  // CAFFE_FILLER_HPP_
