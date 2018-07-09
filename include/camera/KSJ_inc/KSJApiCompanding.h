#ifndef _KSJAPI_COMPANDING_H__
#define _KSJAPI_COMPANDING_H__

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

	// 参考“凯视佳工业相机API函数说明书（数据压扩篇）”
	// Related API KSJ_GetADCResolution to Get Valid bits of 16bits
	// Please KSJ_QueryFunction ( KSJ_SUPPORT_LUT_DOWNLOAD ) to Get if support download LUT
	// KtbFile Must Created by KSJShow. Download 16 to 8 Look Up Data to Camera, it will lost when power off.
	KSJ_API  KSJ_LutFileDownload(int nIndex, const TCHAR *pszKtbFile);
	// Download 16 to 8 Look Up Data to Camera, it will lost when power off.
	KSJ_API  KSJ_LutDataDownload(int nIndex, unsigned char *pData, int nBytes);
	// Enable LUT
	KSJ_API  KSJ_LutSetEnable(int nIndex, bool bEnable);
	// Get If LUT Enable
	KSJ_API  KSJ_LutGetEnable(int nIndex, bool *pbEnable);

	// Please KSJ_QueryFunction ( KSJ_SUPPORT_GAMMA_LUT_DOWNLOAD ) to Get if support download Gamma LUT
	// KtbFile Must Created by KSJShow. Download 16 to 16 Gamma Look Up Data to Camera, it will lost when power off.
	KSJ_API  KSJ_GammaLutFileDownload(int nIndex, const TCHAR *pszKtbFile);
	// Download Gamma Look Up Data to Camera, it will lost when power off.
	KSJ_API  KSJ_GammaLutDataDownload(int nIndex, unsigned short *pData, int nWords);
	// Enable Gamma LUT
	KSJ_API  KSJ_GammaLutSetEnable(int nIndex, bool bEnable);
	// Get If Gamma LUT Enable
	KSJ_API  KSJ_GammaLutGetEnable(int nIndex, bool *pbEnable);

	// 灵敏度模式设置（使用截取8位方式）
	enum KSJ_SENSITIVITYMODE
	{
		KSJ_LOW = 0,
		KSJ_MID,
		KSJ_HIGH,
		KSJ_HIGHER,
		KSJ_HIGHEST
	};
	// Related API KSJ_GetADCResolution
	KSJ_API  KSJ_SensitivitySetMode(int nIndex, KSJ_SENSITIVITYMODE SensitivityMode);
	KSJ_API  KSJ_SensitivityGetMode(int nIndex, KSJ_SENSITIVITYMODE *pSensitivityMode);


#ifdef __cplusplus
}
#endif

#endif