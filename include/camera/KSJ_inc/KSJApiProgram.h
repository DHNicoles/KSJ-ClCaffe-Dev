#ifndef _KSJAPI_PROGRAM_H__
#define _KSJAPI_PROGRAM_H__

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

	// 参考“凯视佳工业相机内部API函数说明书（初始化厂商读写区域篇）”
	// Program and Modify Serials
	KSJ_API  KSJ_SetSerials(int nIndex, unsigned char btSerials);
	KSJ_API  KSJ_GetSerials(int nIndex, unsigned char *pbtSerials);

	KSJ_API  KSJ_WorkingModeProgram(int nIndex, unsigned short wValue);
	KSJ_API  KSJ_WorkingModeReadout(int nIndex, unsigned short *pValue);

	KSJ_API  KSJ_WorkingModeExProgram(int nIndex, unsigned short wValue);
	KSJ_API  KSJ_WorkingModeExReadout(int nIndex, unsigned short *pValue);

	KSJ_API  KSJ_ColorModeProgram(int nIndex, unsigned short wValue);
	KSJ_API  KSJ_ColorModeReadout(int nIndex, unsigned short *pValue);

	KSJ_API  KSJ_SerialsDescProgram(int nIndex, unsigned char btDesc[16]);
	KSJ_API  KSJ_SerialsDescReadout(int nIndex, unsigned char btDesc[16]);

	KSJ_API  KSJ_LutInfoProgram(int nIndex, unsigned short wValue);
	KSJ_API  KSJ_LutInfoReadout(int nIndex, unsigned short *pValue);

	KSJ_API  KSJ_GammaLutInfoProgram(int nIndex, unsigned short wValue);
	KSJ_API  KSJ_GammaLutInfoReadout(int nIndex, unsigned short *pValue);

	KSJ_API  KSJ_HWBMatrixProgram(int nIndex, float fMatrix[3]);
	KSJ_API  KSJ_HWBMatrixReadout(int nIndex, float fMatrix[3]);
	KSJ_API  KSJ_HWBMatrixUpload(int nIndex, float fMatrix[3]);

	KSJ_API  KSJ_HCCMMatrixProgram(int nIndex, float fMatrix[3][3]);
	KSJ_API  KSJ_HCCMMatrixReadout(int nIndex, float fMatrix[3][3]);
	KSJ_API  KSJ_HCCMMatrixUpload(int nIndex, float fMatrix[3][3]);

	KSJ_API  KSJ_LutFileProgram(int nIndex, TCHAR *pszFile);
	KSJ_API  KSJ_LutDataProgram(int nIndex, unsigned char *pData, int nBytes);
	KSJ_API  KSJ_LutDataReadout(int nIndex, unsigned char *pData, int *pnBytes);

	KSJ_API  KSJ_GammaLutFileProgram(int nIndex, TCHAR *pszFile);
	KSJ_API  KSJ_GammaLutDataProgram(int nIndex, unsigned short *pData, int nWords);
	KSJ_API  KSJ_GammaLutDataReadout(int nIndex, unsigned short *pData, int *pnWords);


#ifdef __cplusplus
}
#endif

#endif