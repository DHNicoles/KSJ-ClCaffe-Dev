/************************************************************************/
/* object information container                                         */
/************************************************************************/
#ifndef __OBJECT_INFO_H__
#define __OBJECT_INFO_H__

#include "../utils/util.h"
namespace ice
{
	////////////////////////////////////
	//local def
	////////////////////////////////////
	const int l_life_begin = 1;
	const int l_canBeRecycleScore = 1;

	class ObjectInfo
	{
		public:
			ObjectInfo(int id, cv::Point point, cv::Rect& box);
			~ObjectInfo();
			void OnDestory();
			std::list<cv::Point>& GetTrace() { return trace_; }
			cv::Point& Position() { return trace_.back(); }
			cv::Rect& Box() { return newest_box_; }
			void SetMatchFalse() { match_success_ = false; }
			void SetMatchTrue() { match_success_ = true; }
			bool& IsShit() { return true_shit_; }
			const int Life(){ return life_; }
			const int MaxScore(){ return max_score_; }
			const size_t Index() { return index_; }
			bool CanBeRecycle(){ return max_score_ > l_canBeRecycleScore; }
			//void SetEndTime(std::string endtime){ time_vec[1] = endtime; }
			void OnUpdate(cv::Mat& bound_mask, cv::Mat & frame);
            void AddToMiss(){ ++missed_;}
            void ClearMiss(){ missed_ = 0;}
            double& SearchRatio(){ return search_ratio_;}
            const int MissTime(){ return missed_;}
			const int MaxDisplace(){ return max_displace_; };
			friend std::ostream& operator << (std::ostream& os, ObjectInfo & obj);
		public:
		private:
			bool match_success_;
			size_t index_;
			bool true_shit_;
			int match_time_;
			int life_;
            int missed_;
            double search_ratio_;
			int max_score_;
			cv::Rect newest_box_;
			std::list<cv::Point> trace_;
			double decay_;
			//std::string time_vec[2];

			//count
			//////////////////////////
			//displace
			int max_displace_;
			//////////////////////////

		private:
			//sync
			volatile bool syncing_flag_;		
	};
}

#endif // __OBJECT_INFO_H__
