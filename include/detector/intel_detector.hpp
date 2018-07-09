//note: sudo init 3 if you do not want display
//sudo apt-get install compizconfig-settings-manager and do as http://blog.csdn.net/jiankunking/article/details/69467757
//??? low graphic mode in ubuntu unity plugin
//note: speed will very low if screen is lock or dark, so set dark to never and no dim
#ifndef __IntelDetector_HPP_
#define __IntelDetector_HPP_
#include <caffe/caffe.hpp>
#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>
#endif  // USE_OPENCV
#include <vector>
#include <queue>
#include "detector/kernels.hpp"

#if defined(USE_OPENCV) && defined(HAS_HALF_SUPPORT)
using namespace caffe;  // NOLINT(build/namespaces)
using caffe::Timer;
using std::queue;  
using std::string;

#define RUNWITH_HALF  //liyuming mark:  delete it if you run in float -- fp32

#ifdef RUNWITH_HALF
#define Dtype half
#else
#define Dtype float
#endif

#define CAP_MODE_BGR  0

namespace ice 
{

    class IntelDetector {
        public:
            typedef struct __resultbox {
                float classid;
                float confidence;
                float left;
                float right;
                float top;
                float bottom;
            }resultbox;

            typedef struct __Result {
                vector<resultbox> boxs;
                cv::Mat orgimg;
                cv::Size imgsize;
                int inputid;		
            }Result;

            typedef struct __ImageSize {
                cv::Size isize;
                int inputid;
            }ImageSize;	

            IntelDetector(const string& model_file,
                    const string& weights_file,int min_batch,bool keep_orgimg);
            ~IntelDetector();

            bool Detect(vector<Result>& objects);
            inline int GetCurBatch(){return  num_batch_;}
            inline int GetRGBColor(){return rgbcolor_;}
            inline void SetRGBColor(int color){rgbcolor_=color;}
            inline cv::Size GetNetSize(){return input_geometry_;}
            bool InsertImage(const cv::Mat& orgimg,int inputid,int batch_num);
            void SetBatch(int batch);
            int TryDetect();

        private:
            void get_gpus(vector<int>* gpus);
            void WrapInputLayer(Blob<float>* input_layer);
            cv::Mat PreProcess(const cv::Mat& img);
            void CreateMean();
            void EmptyQueue(queue<Blob<Dtype>*>& que);
            void EmptyQueue(queue<ImageSize>& que);
            void EmptyQueue(queue<cv::Mat>& que);
            boost::shared_ptr<Net<Dtype> > net_;
            std::vector<cv::Mat> input_channels;
            cv::Size input_geometry_;
            queue<Blob<Dtype>*> batchque_;
            queue<ImageSize> imgsizeque_;
            queue<cv::Mat> imgque_;
            pthread_mutex_t mutex ; 
            int nbatch_index_;
            int num_channels_;
            int min_batch_;
            int num_batch_;
            cv::Mat mean_;
            int gpuid_;
            bool keep_orgimg_;
            int max_imgqueue_;
            int curdata_batch_;
            Blob<Dtype>* pbatch_element_;
#ifdef RUNWITH_HALF
            viennacl::ocl::program fp16_ocl_program_;
            Blob<float>* batch_element_float_;
            void PreprocessGPU(int batch_num);
#endif
            int rgbcolor_; //cv::CAP_MODE_BGR  or  cv::CAP_MODE_RGB
    };
}
#endif  // USE_OPENCV
#endif //__IntelDetector_HPP_
