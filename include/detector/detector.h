/************************************************************************/
/* detect  .															*/
/************************************************************************/

#ifndef detector_h__
#define detector_h__
#include "../utils/util.h"
#include "../utils/singleton.h"
namespace ice
{
    class SSDNet;

    class Detector :public Singleton<Detector>
    {
        public:
            enum DETECTOR_TYPE
            {
                DETECTOR_TYPE_SSD,
                DETECTOR_TYPE_MTCNN
            };
            enum DETECTOR_SCALE
            {
                DETECTOR_SCALE_GLOBAL,
                DETECTOR_SCALE_LOCAL
            };

            Detector();
            ~Detector();
            void OnInit();
            void Destroy();
            void Detect(cv::Mat& src, std::vector<cv::Rect>& targets, std::vector<float>& score, DETECTOR_SCALE detect_scale = DETECTOR_SCALE_GLOBAL);
            void Detect(std::vector<cv::Mat>& src_v, std::vector<std::vector<cv::Rect> >& targets, std::vector<std::vector<float> >& score, DETECTOR_SCALE detect_scale);
        private:
            SSDNet* ssd_global_net_ptr_;
            SSDNet* ssd_local_net_ptr_;
            DETECTOR_TYPE detector_type;
    };
}
#endif // detector_h__
