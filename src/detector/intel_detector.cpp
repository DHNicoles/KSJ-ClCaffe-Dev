#include "detector/intel_detector.hpp"

#if defined(USE_OPENCV) && defined(HAS_HALF_SUPPORT)
namespace ice 
{

    // Get all available GPU devices
    void IntelDetector::get_gpus(vector<int>* gpus) {
        int count = 0;
#ifndef CPU_ONLY
        count = Caffe::EnumerateDevices(true);
#else
        NO_GPU;
#endif
        for (int i = 0; i < count; ++i) {
            gpus->push_back(i);
        }
    }

    IntelDetector::IntelDetector(const string& model_file,
            const string& weights_file,int min_batch,bool keep_orgimg) {
        //rgbcolor_=cv::CAP_MODE_BGR;
        rgbcolor_= CAP_MODE_BGR;
        // Set device id and mode
        vector<int> gpus;
        get_gpus(&gpus);
        Caffe::SetDevices(gpus);
        gpuid_ = -1;
        if (gpus.size() != 0) {
#ifndef CPU_ONLY
            for (int i = 0; i < gpus.size(); i++) {
                if (Caffe::GetDevice(gpus[i], true)->backend() == BACKEND_OpenCL) {
                    if (Caffe::GetDevice(gpus[i], true)->CheckVendor("Intel")
                            && Caffe::GetDevice(gpus[i], true)->CheckType("GPU")) {
                        //&& Caffe::GetDevice(gpus[i], true)->CheckCapability("cl_intel_subgroups")) {
                        Caffe::set_mode(Caffe::GPU);
                        Caffe::SetDevice(gpus[i]);
                            gpuid_ = gpus[i];
#ifdef RUNWITH_HALF
                        fp16_ocl_program_ = RegisterMyKernels<Dtype>(&(viennacl::ocl::get_context(Caffe::GetDefaultDevice()->id())));
#endif
                        LOG(INFO) << "Use GPU=" << gpus[i];
                            break;
                    }
                    }
                }
#endif  // !CPU_ONLY
            }
            if(gpuid_<0){
                LOG(INFO) << "!!!!!!!!!!!!!!!!!!!!!!!!!!!!Use CPU!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!";
                CHECK((is_same<Dtype, float>::value)) 
                    << "CPU mode can only use float, not half";
                Caffe::set_mode(Caffe::CPU);
            }

            pthread_mutex_init(&mutex,NULL); 
            min_batch_=min_batch;
            keep_orgimg_ = keep_orgimg;
            /* Load the network. */
            net_.reset(new Net<Dtype>(model_file, TEST, Caffe::GetDefaultDevice()));
            net_->CopyTrainedLayersFrom(weights_file);

            Blob<Dtype>* input_layer = net_->input_blobs()[0];
            num_channels_ = input_layer->channels();
            CHECK(num_channels_ == 3 || num_channels_ == 1)
                << "Input layer should have 1 or 3 channels.";
            input_geometry_ = cv::Size(input_layer->width(), input_layer->height());
            SetBatch(min_batch_);
            CreateMean();
            nbatch_index_ = 0;
            net_->Forward(); //just warm up
#ifdef RUNWITH_HALF
            batch_element_float_ = new Blob<float>();  //I don't know why here must new , if not, it error in cl
#endif
        }

        IntelDetector::~IntelDetector() {
            pthread_mutex_destroy(&mutex);  
            EmptyQueue(batchque_);
            EmptyQueue(imgsizeque_);
            EmptyQueue(imgque_);
#ifdef RUNWITH_HALF
            delete batch_element_float_ ;
#endif
            if(gpuid_>=0)
                caffe::Caffe::TeardownDevice(gpuid_);
        }

        void IntelDetector::EmptyQueue(queue<Blob<Dtype>*>& que)
        {
            while(!que.empty()){
                Blob<Dtype>* pdata = que.front();
                que.pop();
                delete pdata;
            }
        }

        void IntelDetector::EmptyQueue(queue<IntelDetector::ImageSize>& que)
        {
            while(!que.empty()){
                que.pop();
            }
        }

        void IntelDetector::EmptyQueue(queue<cv::Mat>& que)
        {
            while(!que.empty()){
                que.pop();
            }
        }

        //note! do not call it if Detect not finished
        void IntelDetector::SetBatch(int batch)
        {
            if(batch<min_batch_ ||batch==num_batch_)
                return;
            Blob<Dtype>* input_layer = net_->input_blobs()[0];
            num_batch_=batch;
            //max_imgqueue_=num_batch_*2;
            max_imgqueue_= 12;
            LOG(INFO) << "Change Batch to " << num_batch_ ;
            input_layer->Reshape(num_batch_, num_channels_,input_geometry_.height, input_geometry_.width);
            /* Forward dimension change to all layers. */
            net_->Reshape();
        }

        int IntelDetector::TryDetect() {
            int curbatch=0;
            pthread_mutex_lock(&mutex); 
            if(!batchque_.empty())
                curbatch=batchque_.front()->num();
            pthread_mutex_unlock(&mutex);
            //if(curbatch>0) PreprocessGPU(curbatch);	
            return curbatch;
        }

        bool IntelDetector::Detect(vector<IntelDetector::Result>& objects) {
            Blob<Dtype>* pdata;
            pthread_mutex_lock(&mutex); 
            if(!batchque_.empty()){
                pdata = batchque_.front();
                batchque_.pop();
            }
            else{
                pthread_mutex_unlock(&mutex); 	
                return false;
            }

            for (int i=0;i<pdata->num();i++) {
                if(!imgsizeque_.empty()){
                    objects[i].imgsize = imgsizeque_.front().isize;
                    objects[i].inputid = imgsizeque_.front().inputid;
                    imgsizeque_.pop();
                }
                else
                    objects[i].imgsize = cv::Size(0,0);

                if(keep_orgimg_ && !imgque_.empty()){
                    objects[i].orgimg = imgque_.front();
                    imgque_.pop();
                }
            }
            SetBatch(pdata->num());
            pthread_mutex_unlock(&mutex); 

            net_->input_blobs()[0]->ShareData(*pdata);
            net_->Forward();
            /* get the result */
            Blob<Dtype>* result_blob = net_->output_blobs()[0];
            const Dtype* result = result_blob->cpu_data();
            const int num_det = result_blob->height();
            for (int k = 0; k < num_det ; k++) {
                resultbox object;
                int imgid = (int)result[0];
                int w=objects[imgid].imgsize.width;
                int h=objects[imgid].imgsize.height;		
                object.classid = (int)result[1];
                object.confidence = result[2];
                object.left = (int)(result[3] * w);
                object.top = (int)(result[4] * h);
                object.right = (int)(result[5] * w);
                object.bottom = (int)(result[6] * h);
                if (object.left < 0) object.left = 0;
                if (object.top < 0) object.top = 0;
                if (object.right >= w) object.right = w - 1;
                if (object.bottom >= h) object.bottom = h - 1;
                objects[imgid].boxs.push_back(object);
                result+=7;
            }
            delete pdata;
            return true;
        }

        /* Wrap the input layer of the network in separate cv::Mat objects
         * (one per channel). This way we save one memcpy operation */
        void IntelDetector::WrapInputLayer(Blob<float>* input_layer) {
            input_channels.clear();
            int width = input_layer->width();
            int height = input_layer->height();
            float* input_data = input_layer->mutable_cpu_data();
            //if(rgbcolor_==cv::CAP_MODE_BGR){
            if(rgbcolor_== CAP_MODE_BGR){
                for (int i = 0; i < input_layer->channels()*input_layer->num(); ++i) {
                    cv::Mat channel(height, width, CV_32FC1, input_data);
                    input_channels.push_back(channel);
                    input_data += width * height;
                }
            }
            else{
                for (int i = 0; i < input_layer->num(); ++i) {  //RGB-BGR
                    cv::Mat channelB(height, width, CV_32FC1, input_data);
                    input_data += width * height;
                    cv::Mat channelG(height, width, CV_32FC1, input_data);
                    input_data += width * height;
                    cv::Mat channelR(height, width, CV_32FC1, input_data);
                    input_data += width * height;

                    input_channels.push_back(channelR);
                    input_channels.push_back(channelG);
                    input_channels.push_back(channelB);
                }		
            }
        }

        cv::Mat IntelDetector::PreProcess(const cv::Mat& img) {
            cv::Mat sample;
            if (img.channels() == 3 && num_channels_ == 1)
                cv::cvtColor(img, sample, cv::COLOR_BGR2GRAY);
            else if (img.channels() == 4 && num_channels_ == 1)
                cv::cvtColor(img, sample, cv::COLOR_BGRA2GRAY);
            else if (img.channels() == 4 && num_channels_ == 3)
                cv::cvtColor(img, sample, cv::COLOR_BGRA2BGR);
            else if (img.channels() == 1 && num_channels_ == 3)
                cv::cvtColor(img, sample, cv::COLOR_GRAY2BGR);
            else
                sample = img;

            cv::Mat sample_resized;
            if (sample.size() != input_geometry_) {
                cv::resize(sample, sample_resized, input_geometry_);
            }
            else
                sample_resized = sample;

            cv::Mat sample_float;
            if (num_channels_ == 3)
                sample_resized.convertTo(sample_float, CV_32FC3);
            else
                sample_resized.convertTo(sample_float, CV_32FC1);

            //cv::scaleAdd (sample_float, 0.007843, mean_, sample_float_sub_scale); //scaleAdd or (add+multiply)? which speed : ans is scaleAdd
            //cv::add(sample_float, -127.5, sample_float_sub);
            //cv::multiply(sample_float_sub, 0.007843, sample_float_sub_scale);

            return sample_float;
        }

        void IntelDetector::CreateMean() {
            if (num_channels_ == 3)
                mean_= cv::Mat(input_geometry_, CV_32FC3, cv::Scalar(-127.5*0.007843,-127.5*0.007843,-127.5*0.007843));
            else
                mean_= cv::Mat(input_geometry_, CV_32FC1, cv::Scalar(-127.5*0.007843));	
        }

#ifdef RUNWITH_HALF
        void IntelDetector::PreprocessGPU(int batch_num)
        {
            int n=input_geometry_.height*input_geometry_.width*num_channels_*curdata_batch_;
            float scale = 0.007843;
            float mean = -127.5;
#if 1 //gpu cache refresh, speedup from 100fps to 150fps
            Blob<float> batch_element_float_blob;
            batch_element_float_blob.Reshape(batch_num, num_channels_,input_geometry_.height, input_geometry_.width);		
            batch_element_float_blob.set_cpu_data(batch_element_float_->mutable_cpu_data());
            batch_element_float_blob.data().get()->async_gpu_push();
            float* in_data = batch_element_float_blob.mutable_gpu_data();
#else
            float* in_data = batch_element_float_->mutable_gpu_data();
#endif
            Dtype* out_data = pbatch_element_->mutable_gpu_data();

            viennacl::ocl::context &ctx = viennacl::ocl::get_context(Caffe::GetDefaultDevice()->id());
            // Execute kernel
            viennacl::ocl::kernel &oclk_preprocess = fp16_ocl_program_.get_kernel(
                    CL_KERNEL_SELECT("preprocess"));
            viennacl::ocl::enqueue(
                    oclk_preprocess(n, scale, mean, WrapHandle((cl_mem)in_data, &ctx),
                        WrapHandle((cl_mem)out_data, &ctx)),
                    ctx.get_queue());
        }
#endif

        /*
           InsertImage will fill a blob until blob full, if full return the blob point
           */
        bool IntelDetector::InsertImage(const cv::Mat& orgimg,int inputid,int batch_num) {
            bool retvalue=false;
            if(orgimg.cols==0 || orgimg.rows==0)
                return retvalue;

            pthread_mutex_lock(&mutex); 
            if(imgsizeque_.size()>=max_imgqueue_){
                pthread_mutex_unlock(&mutex); 	
                return false;
            }

            if(nbatch_index_==0){  //new a blob
                pbatch_element_=new Blob<Dtype>();
                pbatch_element_->Reshape(batch_num, num_channels_,input_geometry_.height, input_geometry_.width);		
                curdata_batch_ = batch_num;
#ifdef RUNWITH_HALF
                batch_element_float_->Reshape(batch_num, num_channels_,input_geometry_.height, input_geometry_.width);		
                WrapInputLayer(batch_element_float_);
#else
                WrapInputLayer(pbatch_element_);
#endif
                batchque_.push(pbatch_element_);
            }
            ImageSize is;
            is.isize = orgimg.size();
            is.inputid = inputid;
            imgsizeque_.push(is);
            if(keep_orgimg_)
                imgque_.push(orgimg);

            cv::Mat img = PreProcess(orgimg);		
            /* Convert the input image to the input image format of the network. */
#ifndef RUNWITH_HALF
            cv::scaleAdd (img, 0.007843, mean_, img); //scaleAdd or (add+multiply)? which speed : ans is scaleAdd
#endif
            cv::split(img, &input_channels[num_channels_*nbatch_index_]);

            //if full return pbatch_element_
            if(++nbatch_index_>=curdata_batch_){
                nbatch_index_=0;
                retvalue = true;
#ifdef RUNWITH_HALF
                PreprocessGPU(batch_num);  //not more stable speed, but a little bit faster than putting in tyedetect		
#endif
            }
            pthread_mutex_unlock(&mutex); 	
            return retvalue;
        }
    }
#endif  // USE_OPENCV
