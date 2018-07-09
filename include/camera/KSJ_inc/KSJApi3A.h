#ifndef __KSJ_API_3A_H__
#define __KSJ_API_3A_H__

#if defined (_WIN32)
#  ifdef KSJAPI_EXPORTS
#    define KSJAPI_IMPORT_EXPORT __declspec(dllexport)
#  else
#    define KSJAPI_IMPORT_EXPORT __declspec(dllimport)
#  endif

#  if defined (_M_IX86) || defined (__i386__)
#    define KSJAPI_CALLTYPE __stdcall
#  else
#    define KSJAPI_CALLTYPE  
#  endif

#elif defined (__GNUC__) && (__GNUC__ >= 4) && (defined (__linux__) || defined (__APPLE__))
#  define KSJAPI_IMPORT_EXPORT __attribute__((visibility("default")))
#  if defined (__i386__)
#    define KSJAPI_CALLTYPE __attribute__((stdcall))
#  else
#    define KSJAPI_CALLTYPE /* default */
#  endif
#  ifndef EXTERN_C
#    define EXTERN_C extern "C"
#  endif

#else
#  error Unknown platform, file needs adaption
#endif


#ifndef _WIN32
#define KSJ_API int
#define __stdcall
#else
#define KSJ_API KSJAPI_IMPORT_EXPORT int KSJAPI_CALLTYPE
#endif

#ifdef __cplusplus
extern "C"{
#endif

	// 设置自动曝光的区域，这个区域是相对预览图像的左上角像素位置
	KSJ_API  KSJ_AESetRegion(int nIndex, int nX, int nY, int nW, int nH);
	// 设置自动曝光的区域，这个区域是相对预览图像的左上角像素位置,是否显示区域
	KSJ_API  KSJ_AEShowRegion(int nIndex, bool bShow);
	// Get AE Region (according to preview image left top coner ) and Show Status.
	KSJ_API  KSJ_AEGetRegion(int nIndex, int *pnX, int *pnY, int *pnW, int *pnH, bool *pbShow);

	KSJ_API  KSJ_AESetPeakAveRatio(int nIndex, float fRatio);
	KSJ_API  KSJ_AEGetPeakAveRatio(int nIndex, float *pfRatio);

	// Set Exposure Time Range When Do Auto Exposure
	KSJ_API  KSJ_AESetExposureTimeRange(int nIndex, float fMinExpMs, float fMaxExpMs);
	KSJ_API  KSJ_AEGetExposureTimeRange(int nIndex, float *pfMinExpMs, float *pfMaxExpMs);

	// AE Status return to callback
	enum KSJ_AE_STATUS 
	{
		KSJ_AE_ONCE_SUCCESS = 0, 
		KSJ_AE_ONCE_FAIL_MAX_COUNT,
		KSJ_AE_ONCE_FAIL_WAVE, 
		KSJ_AE_ONCE_FAIL_OVER_RANGE, 
		KSJ_AE_DOING, 
		KSJ_AE_CONTINUE_SUCCESS, 
		KSJ_AE_CONTINUE_FAIL_WAVE, 
		KSJ_AE_CONTINUE_FAIL_OVER_RANGE 
	};

	// Auto Exposure Ex Callback, Call when AE Doing, Success, Fail.
	typedef void(__stdcall *KSJ_AECALLBACKEX)(KSJ_AE_STATUS Status, int nExpsoureTimeMs, void *lpContext);
	KSJ_API  KSJ_AESetCallbackEx(int nIndex, KSJ_AECALLBACKEX pfAECallbackEx, void *lpContext);

	KSJ_API  KSJ_AESetMaxCount(int nIndex, int nMaxCount);
	KSJ_API  KSJ_AESetTarget(int nIndex, int nTarget);
	KSJ_API  KSJ_AEStartEx(int nIndex, bool bStart);
	KSJ_API  KSJ_AEGetStatusEx(int nIndex, bool *pbStart, int *pnMaxCount, int *pnTarget);

	// Auto Exposure Callback, Call when AE Finish.
	typedef void(__stdcall *KSJ_AECALLBACK)(bool bSuccess, int nResult, void *lpContext);
	// 定义自动曝光操作结束后的回调函数，当自动曝光结束时，KSJ开发包会自动调用AECallback用户指定的回调函数
	KSJ_API  KSJ_AESetCallback(int nIndex, KSJ_AECALLBACK pfAECallback, void *lpContext);
	// 开始或终止自动曝光调节
	KSJ_API  KSJ_AEStart(int nIndex, bool bStart, int nMaxCount, int nTarget);


	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// 设置自动白平衡区域，这个区域是相对预览图像的左上角像素位置
	KSJ_API  KSJ_AWBSetRegion(int nIndex, int nX, int nY, int nW, int nH);
	// 设置自动白平衡区域，这个区域是相对预览图像的左上角像素位置,是否显示区域
	KSJ_API  KSJ_AWBShowRegion(int nIndex, bool bShow);
	KSJ_API  KSJ_AWBGetRegion(int nIndex, int *pnX, int *pnY, int *pnW, int *pnH, bool *pbShow);

	// 设置聚焦评估区域
	KSJ_API  KSJ_AFSetRegion(int nIndex, int nX, int nY, int nW, int nH);
	KSJ_API  KSJ_AFGetRegion(int nIndex, int *pnX, int *pnY, int *pnW, int *pnH, bool *pbShow);
	KSJ_API  KSJ_AFShowRegion(int nIndex, bool bShow);
	// 聚焦评估回调函数, nValue为返回的评估值
	typedef void(__stdcall *KSJ_AFCALLBACK)(int nValue, void *lpContext);
	// 设置聚焦评估回调
	KSJ_API  KSJ_AFSetCallback(int nIndex, KSJ_AFCALLBACK fpAFCallback, void *lpContext);
	// 开始或终止聚焦评估
	KSJ_API  KSJ_AFStart(int nIndex, bool bStart);


#ifdef __cplusplus
}
#endif

#endif
