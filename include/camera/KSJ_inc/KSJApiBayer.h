#ifndef _KSJAPI_BAYER_H__
#define _KSJAPI_BAYER_H__

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
#else
#define KSJ_API KSJAPI_IMPORT_EXPORT int KSJAPI_CALLTYPE
#endif

#ifdef __cplusplus
extern "C"{
#endif

	// Bayer Mode
	enum KSJ_BAYERMODE
	{
		KSJ_BGGR_BGR24 = 0,
		KSJ_GRBG_BGR24,
		KSJ_RGGB_BGR24,
		KSJ_GBRG_BGR24,
		KSJ_BGGR_BGR24_FLIP,
		KSJ_GRBG_BGR24_FLIP,
		KSJ_RGGB_BGR24_FLIP,
		KSJ_GBRG_BGR24_FLIP,
		KSJ_BGGR_BGR32,
		KSJ_GRBG_BGR32,
		KSJ_RGGB_BGR32,
		KSJ_GBRG_BGR32,
		KSJ_BGGR_BGR32_FLIP,
		KSJ_GRBG_BGR32_FLIP,
		KSJ_RGGB_BGR32_FLIP,
		KSJ_GBRG_BGR32_FLIP,
		KSJ_BGGR_GRAY8,
		KSJ_GRBG_GRAY8,
		KSJ_RGGB_GRAY8,
		KSJ_GBRG_GRAY8,
		KSJ_BGGR_GRAY8_FLIP,
		KSJ_GRBG_GRAY8_FLIP,
		KSJ_RGGB_GRAY8_FLIP,
		KSJ_GBRG_GRAY8_FLIP
	};

	// 得到当前相机的默认Bayer模式
	KSJ_API   KSJ_BayerGetDefaultMode(int nIndex, KSJ_BAYERMODE *pBayerMode);
	// 设置Bayer模式，同时影响采集和预览，这个函数除特殊需要调用，否则用户不需调用此功能。
	KSJ_API   KSJ_BayerSetMode(int nIndex, KSJ_BAYERMODE BayerMode);
	// 获得当前Bayer模式
	KSJ_API   KSJ_BayerGetMode(int nIndex, KSJ_BAYERMODE *pBayerMode);

	// Bayer Filter Algorithm 
	enum KSJ_FILTERMODE
	{
		KSJ_NEARESTNEIGHBOR,
		KSJ_BILINEAR,
		KSJ_SMOOTHHUE,
		KSJ_EDGESENSING,
		KSJ_LAPLACIAN,
		KSJ_FASTBILINEAR
	};

	// 设置Bayer Filter模式，同时影响采集和预览，这个函数除特殊需要调用，否则用户不需调用此功能。
	KSJ_API   KSJ_FilterSetMode(int nIndex, KSJ_FILTERMODE FilterMode);
	// 获得当前Bayer模式
	KSJ_API   KSJ_FilterGetMode(int nIndex, KSJ_FILTERMODE *pFilterMode);


#ifdef __cplusplus
}
#endif

#endif