#ifndef _KSJAPI_IMAGE_H__
#define _KSJAPI_IMAGE_H__

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

	// 坏点校正低阈值和高阈值设置（硬件不支持此设置，由软件处理时有效）
	KSJ_API  KSJ_BadPixelCorrectionSetThreshold(int nIndex, unsigned char btLow, unsigned char btHigh);
	KSJ_API  KSJ_BadPixelCorrectionGetThreshold(int nIndex, unsigned char *pbtLow, unsigned char *pbtHigh);

	enum KSJ_THRESHOLD
	{
		KSJ_THRESHOLD_LOW,
		KSJ_THRESHOLD_HIGH
	};
	// 坏点校正设置及阈值设置，低阈值会过滤更多的坏点
	KSJ_API  KSJ_BadPixelCorrectionSet(int nIndex, bool bEnable, KSJ_THRESHOLD Threshold);
	KSJ_API  KSJ_BadPixelCorrectionGet(int nIndex, bool *pbEnable, KSJ_THRESHOLD *pThreshold);

	// Same to Parameter KSJ_CONTRAST（-128~128）， Software Preccess, may influence capture efficiency or preivew frame rate.
	KSJ_API  KSJ_ProcessContrastGetRange(int nIndex, int *pnMinContrast, int *pnMaxContrast);
	KSJ_API  KSJ_ProcessContrastSet(int nIndex, int nContrast);
	KSJ_API  KSJ_ProcessContrastGet(int nIndex, int *pnContrast);

	// Same to Parameter KSJ_BRIGHTNESS（-128~128）, Software Preccess, may influence capture efficiency or preivew frame rate.
	KSJ_API  KSJ_ProcessBrightnessGetRange(int nIndex, int *pnMinBrightness, int *pnMaxBrightness);
	KSJ_API  KSJ_ProcessBrightnessSet(int nIndex, int nBrightness);
	KSJ_API  KSJ_ProcessBrightnessGet(int nIndex, int *pnBrightness);

	// Same to Parameter KSJ_SATURATION（-128~128）, Software Preccess, may influence capture efficiency or preivew frame rate.
	KSJ_API  KSJ_ProcessSaturationGetRange(int nIndex, int *pnMinSaturation, int *pnMaxSaturation);
	KSJ_API  KSJ_ProcessSaturationSet(int nIndex, int nSaturation);
	KSJ_API  KSJ_ProcessSaturationGet(int nIndex, int *pnStaturation);

	// 设置Gamma值，范围 -9 - +9
	KSJ_API  KSJ_GammaGetRange(int nIndex, int *pnMinGamma, int *pnMaxGamm);
	KSJ_API  KSJ_GammaSetValue(int nIndex, int nValue);
	KSJ_API  KSJ_GammaGetValue(int nIndex, int *pnValue);

	typedef enum
	{
		KSJ_ALL_HUES,
		KSJ_RED_HUES,
		KSJ_YELLOW_HUES,
		KSJ_GREEN_HUES,
		KSJ_CYAN_HUES,
		KSJ_BLUE_HUES,
		KSJ_MAGENTA_HUES
	} KSJHueRange;

	// nHue: -180 ~ 180, nSaturation: -100 ~ 100, nLightness: -100 ~ 100
	KSJ_API  KSJ_HueSaturationSet(int nIndex, KSJHueRange HueRange, int nHue, int nSaturation, int nLightness);
	KSJ_API  KSJ_HueSaturationGet(int nIndex, KSJHueRange HueRange, int *pnHue, int *pnSaturation, int *pnLightness);
	KSJ_API  KSJ_HueSaturationSetEnable(int nIndex, bool bEnable);
	KSJ_API  KSJ_HueSaturationGetEnable(int nIndex, bool *pbEnable);

#ifdef __cplusplus
}
#endif

#endif