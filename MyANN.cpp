// MyANN.cpp implements the AI wrapper for TensorFlow

#include "MyANN.h"
#include <array>
#include <vector>
using namespace AI;

std::unique_ptr<MyANN> MyANN::New(INPUT_CLASS ModelInput, TF_DataType InputDataType, size_t OutputLength)
{
   auto model = std::unique_ptr<MyANN>(new MyANN);
   model->ModelInputType = ModelInput;
   model->InputDataType = InputDataType;
   model->OutputLength = OutputLength;
   return std::move(model);
}

void MyANN::set_input_shape(std::array<int, 2> ListInputDimensions)
{
   if (this->ModelInputType != INPUT_CLASS::LIST)
   {
      std::cout << "Input shape error. For a list/vector input, you need to specify 0: length and 1: number of channels (typically 1)\n"
         << "Example expected format: {1024, 1} corresponds to a normal list with 1024 items.\n";
      exit(1337);
   }
   int dtypesize = static_cast<int>(TF_DataTypeSize(this->InputDataType));
   this->input_shape = { {1, ListInputDimensions.at(0), ListInputDimensions.at(1)}, dtypesize };
   for (const auto& dim : ListInputDimensions)
   {
      this->DataSizeInBytes *= dim;
   }
   this->DataSizeInBytes *= dtypesize;
}
void MyANN::set_input_shape(std::array<int, 3> Image2dInputDimensions)
{
   if (this->ModelInputType != INPUT_CLASS::IMAGE_2D)
   {
      std::cout << "Input shape error. For an image input, you need to specify 0: height, 1:width, and 2: number of channels (typically 1)\n"
         << "Example expected format: {960, 1280, 3} corresponds to a 3-channel (RGB/BGR/etc) image with heigh 960px and width 1280px.\n";
      exit(1337);
   }
   int dtypesize = static_cast<int>(TF_DataTypeSize(this->InputDataType));
   this->input_shape = { {1, Image2dInputDimensions.at(0), Image2dInputDimensions.at(1), Image2dInputDimensions.at(2)}, dtypesize };
   for (const auto& dim : Image2dInputDimensions)
   {
      this->DataSizeInBytes *= dim;
   }
   this->DataSizeInBytes *= dtypesize;
}
void MyANN::set_input_shape(std::array<int, 4> Pointcloud3dInputDimensions)
{
   if (this->ModelInputType != INPUT_CLASS::IMAGE_2D)
   {
      std::cout << "Input shape error. For an PC input, you need to specify 0: x shape, 1: y shape, 2: z shape, and 3: number of channels (typically 1)\n"
         << "Example expected format: {10, 10, 10, 1}\n";
      exit(1337);
   }
   int dtypesize = static_cast<int>(TF_DataTypeSize(this->InputDataType));
   this->input_shape = { {1, Pointcloud3dInputDimensions.at(0), Pointcloud3dInputDimensions.at(1), Pointcloud3dInputDimensions.at(2), Pointcloud3dInputDimensions.at(3)}, dtypesize };
   for (const auto& dim : Pointcloud3dInputDimensions)
   {
      this->DataSizeInBytes *= dim;
   }
   this->DataSizeInBytes *= dtypesize;
}

bool MyANN::valid_session() const
{
   if (!this->session)
      return false;
   return true;
}

void MyANN::load_model(std::string filename, std::string input_tensor_name, std::string output_tensor_name)
{
   this->session = std::unique_ptr<MySession>(my_model_load(filename.c_str(), input_tensor_name.c_str(), output_tensor_name.c_str()));
}

template std::vector<float> MyANN::run_model(void* input_data);
template std::vector<double> MyANN::run_model(void* input_data);
template std::vector<int> MyANN::run_model(void* input_data);
template std::vector<char> MyANN::run_model(void* input_data);
template<typename T>
std::vector<T> MyANN::run_model(void* input_data)
{
   std::vector<T> ret;
   auto input_values = tf_obj_unique_ptr(
      TF_NewTensor(this->InputDataType, this->input_shape.values, this->input_shape.dim, 
         input_data, this->DataSizeInBytes, null_deallocator, nullptr)
   );
   if (!input_values)
   {
      std::cout << "Tensor creation failed!" << std::endl;
      exit(17);
   }
   CStatus status;
   TF_Tensor* inputs[] = { input_values.get() };
   TF_Tensor* outputs[1] = {};
   TF_SessionRun(this->session->session.get(), nullptr,
      &session->inputs, inputs, 1,
      &session->outputs, outputs, 1,
      nullptr, 0, nullptr, status.ptr);
   auto _output_holder = tf_obj_unique_ptr(outputs[0]);
   if (status.failure())
   {
      status.dump_error();
      exit(18);
   }
   TF_Tensor &output = *outputs[0];
   if (TF_TensorType(&output) != this->InputDataType)
   {
      std::cout << "Error, unexpected output tensor type.\n";
      exit(19);
   }
   size_t output_size = TF_TensorByteSize(&output) / TF_DataTypeSize(this->InputDataType);
   assert(output_size == this->OutputLength);
   auto output_array = static_cast<const float*>(TF_TensorData(&output));
   for (int i = 0; i < output_size; i++)
      ret.push_back(output_array[i]);
   return ret;
}

static TF_Buffer* read_tf_buffer_from_file(const char* file) 
{
   std::ifstream t(file, std::ifstream::binary);
   t.exceptions(std::ifstream::failbit | std::ifstream::badbit);
   t.seekg(0, std::ios::end);
   size_t size = t.tellg();
   auto data = std::make_unique<char[]>(size);
   t.seekg(0);
   t.read(data.get(), size);

   TF_Buffer *buf = TF_NewBuffer();
   buf->data = data.release();
   buf->length = size;
   buf->data_deallocator = free_cpp_array<char>;
   return buf;
}

MySession* my_model_load(const char *filename, const char *input_name, const char *output_name)
{
   std::cout << "Loading model " << filename << "\n";
   CStatus status;

   auto graph = tf_obj_unique_ptr(TF_NewGraph());
   {
      // Load a protobuf containing a GraphDef
      auto graph_def = tf_obj_unique_ptr(read_tf_buffer_from_file(filename));
      if (!graph_def) {
         return nullptr;
      }

      auto graph_opts = tf_obj_unique_ptr(TF_NewImportGraphDefOptions());
      TF_GraphImportGraphDef(graph.get(), graph_def.get(), graph_opts.get(), status.ptr);
   }

   if (status.failure()) {
      status.dump_error();
      return nullptr;
   }

   auto input_op = TF_GraphOperationByName(graph.get(), input_name);
   auto output_op = TF_GraphOperationByName(graph.get(), output_name);
   if (!input_op || !output_op) {
      return nullptr;
   }

   auto session = std::make_unique<MySession>();
   {
      auto opts = tf_obj_unique_ptr(TF_NewSessionOptions());
      session->session = tf_obj_unique_ptr(TF_NewSession(graph.get(), opts.get(), status.ptr));
   }

   if (status.failure()) {
      return nullptr;
   }
   assert(session);

   graph.swap(session->graph);
   session->inputs = { input_op, 0 };
   session->outputs = { output_op, 0 };

   return session.release();
}