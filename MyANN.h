# pragma once

/*
Andrew Lee
5 September 2019
AI.h

This module serves as a convenient wrapper for the TensorFlow C API,
allowing us to deploy ML models without knowing exactly how the API/TensorFlow work.
*/

#include <tensorflow/c/c_api.h>
#include <memory>
#include <iostream>
#include <algorithm>
#include <cstddef>
#include <iterator>
#include <vector>
#include <assert.h>
#include <string.h>
#include <fstream>
#include <stdint.h>

static TF_Buffer *read_tf_buffer_from_file(const char* file);

/**
 * A Wrapper for the C API status object.
 */
class CStatus {
public:
   TF_Status *ptr;
   CStatus()
   {
      ptr = TF_NewStatus();
   }

   /**
    * Dump the current error message.
    */
   void dump_error()const
   {
      std::cerr << "TF status error: " << TF_Message(ptr) << std::endl;
   }

   /**
    * Return a boolean indicating whether there was a failure condition.
    * @return
    */
   inline bool failure()const
   {
      return TF_GetCode(ptr) != TF_OK;
   }

   ~CStatus()
   {
      if (ptr)TF_DeleteStatus(ptr);
   }
};

namespace detail {
   template<class T>
   class TFObjDeallocator;

   template<>
   struct TFObjDeallocator<TF_Status> { static void run(TF_Status *obj) { TF_DeleteStatus(obj); } };

   template<>
   struct TFObjDeallocator<TF_Graph> { static void run(TF_Graph *obj) { TF_DeleteGraph(obj); } };

   template<>
   struct TFObjDeallocator<TF_Tensor> { static void run(TF_Tensor *obj) { TF_DeleteTensor(obj); } };

   template<>
   struct TFObjDeallocator<TF_SessionOptions> { static void run(TF_SessionOptions *obj) { TF_DeleteSessionOptions(obj); } };

   template<>
   struct TFObjDeallocator<TF_Buffer> { static void run(TF_Buffer *obj) { TF_DeleteBuffer(obj); } };

   template<>
   struct TFObjDeallocator<TF_ImportGraphDefOptions> {
      static void run(TF_ImportGraphDefOptions *obj) { TF_DeleteImportGraphDefOptions(obj); }
   };

   template<>
   struct TFObjDeallocator<TF_Session> {
      static void run(TF_Session *obj) {
         CStatus status;
         TF_DeleteSession(obj, status.ptr);
         if (status.failure()) {
            status.dump_error();
         }
      }
   };
}

template<class T> struct TFObjDeleter {
   void operator()(T* ptr) const {
      detail::TFObjDeallocator<T>::run(ptr);
   }
};

template<class T> struct TFObjMeta {
   typedef std::unique_ptr<T, TFObjDeleter<T>> UniquePtr;
};

template<class T> typename TFObjMeta<T>::UniquePtr tf_obj_unique_ptr(T *obj) {
   typename TFObjMeta<T>::UniquePtr ptr(obj);
   return ptr;
}

class MySession {
public:
   typename TFObjMeta<TF_Graph>::UniquePtr graph;
   typename TFObjMeta<TF_Session>::UniquePtr session;

   TF_Output inputs, outputs;
};

/**
 * Load a GraphDef from a provided file.
 * @param filename: The file containing the protobuf encoded GraphDef
 * @param input_name: The name of the input placeholder
 * @param output_name: The name of the output tensor
 * @return
 */

MySession* my_model_load(const char *filename, const char *input_name, const char *output_name);

template<class T> static void free_cpp_array(void* data, size_t length) {
   delete[]((T *)data);
}

/**
 * Deallocator for TF_NewTensor data.
 * @tparam T
 * @param data
 * @param length
 * @param arg
 */
 // Use this function if the data for the TensorFlow model is manually allocated on the heap
template<typename T> static void cpp_array_deallocator(void* data, size_t length, void* arg) {
   delete[]((T *)data);
}

// Use this function if the data for the TensorFlow model is on the stack or stored by a smart pointer
static void null_deallocator(void* data, size_t length, void*arg)
{
   ; // do nothing. This is for if your data is stored on the stack or in a smart pointer/container
}

static TF_Buffer* read_tf_buffer_from_file(const char* file);

constexpr int MY_TENSOR_SHAPE_MAX_DIM = 16;
struct TensorShape {
   int64_t values[MY_TENSOR_SHAPE_MAX_DIM];
   int dim;

   int64_t size()const {
      assert(dim >= 0);
      int64_t v = 1;
      for (int i = 0; i < dim; i++)v *= values[i];
      return v;
   }
};

class MyANN
{
public:
   enum class INPUT_CLASS { LIST, IMAGE_2D, POINTCLOUD_3D };

   static std::unique_ptr<MyANN> New(INPUT_CLASS ModelInput, TF_DataType InputDataType, size_t OutputLength);

   void set_input_shape(std::array<int, 2> ListInputDimensions);
   void set_input_shape(std::array<int, 3> Image2dInputDimensions);
   void set_input_shape(std::array<int, 4> Pointcloud3dInputDimensions);

   virtual bool valid_session() const;
   virtual void load_model(std::string filename, std::string input_tensor_name, std::string output_tensor_name);

   template<typename T>
   [[nodiscard]] std::vector<T> run_model(void* input_data) noexcept; // run_model owns no memory. It will not delete anything.

   virtual ~MyANN() = default;
   virtual MyANN(MyANN& toCopy) = delete;
   virtual MyANN(MyANN&& toMove) = delete;
protected:
   MyANN() = default;
   INPUT_CLASS ModelInputType;
   TensorShape input_shape;
   size_t DataSizeInBytes = 1;
   size_t OutputLength = 0;
   TF_DataType InputDataType = TF_DataType::TF_FLOAT;
   std::unique_ptr<MySession> session = nullptr;
};