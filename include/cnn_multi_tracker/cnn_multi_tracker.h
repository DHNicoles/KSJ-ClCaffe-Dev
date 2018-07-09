/************************************************************************/
/* Multi-tracker based on CNN                                           */
/************************************************************************/
#ifndef __CNN_MULTI_TRACKER_H__
#define __CNN_MULTI_TRACKER_H__

#include "../utils/util.h"
#include "../utils/singleton.h"
namespace ice
{
	class ObjectInfo;
	class CNNMultiTracker :public Singleton<CNNMultiTracker>
	{
	public:
		CNNMultiTracker();
		~CNNMultiTracker();
		void OnInit();
		void OnDestroy();
	    bool IsOutboard(cv::Rect & positionBox);
		void AddTracker(cv::Mat& frame, cv::Rect& positionBox);
		void OnUpdate(cv::Mat & frame);
		//void UpdateObjInfo(cv::Mat & frame);
		void Mark(size_t index);
		void RemoveInvalid();
		void Replace(cv::Mat& frame, cv::Rect& positionBox, size_t index);
		void DrawTrace(cv::Mat& frame);
		void DrawUI(cv::Mat& frame);
		std::map<size_t, ObjectInfo*>& TrackingObectorMap() { return object_info_map_; }
		void SetAllMatchFalse();
		void SetMatchTrue(size_t index);
		void SetBound(cv::Size, cv::Rect);
		//void PrintRecycle();
	private:
		int GetIndex_();
		void RemoveTrackerByIndex_(size_t index);
	private:
		std::map<size_t, ObjectInfo*> object_info_map_;
		std::list<size_t> invalid_index_list_;
		size_t key_max_;
		std::set<size_t> index_pool_set_;
		cv::Mat bound_mask_;
	};
}


#endif // __CNN_MULTI_TRACKER_H__
