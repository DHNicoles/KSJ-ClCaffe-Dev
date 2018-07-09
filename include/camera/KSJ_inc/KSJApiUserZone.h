#ifndef _KSJAPI_USERZONE_H__
#define _KSJAPI_USERZONE_H__

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

	enum KSJ_EEPROM_TYPE { KSJ_24LC64, KSJ_24LC1024, KSJ_24LC128 };

	KSJ_API  KSJ_EEPROMTypeGet(int nIndex, KSJ_EEPROM_TYPE *pType);

	KSJ_API  KSJ_WrEEPROM(int nIndex, unsigned short uRomAdress, unsigned char btValue);
	KSJ_API  KSJ_RdEEPROM(int nIndex, unsigned short uRomAdress, unsigned char *pbtValue);

	KSJ_API  KSJ_WrEEPROMEx(int nIndex, unsigned int ulRomAdress, unsigned char btValue);
	KSJ_API  KSJ_RdEEPROMEx(int nIndex, unsigned int ulRomAdress, unsigned char *pbtValue);

	KSJ_API  KSJ_WrEEPROMExEx(int nIndex, unsigned int ulRomAdress, unsigned char *pData, int nBytes);
	KSJ_API  KSJ_RdEEPROMExEx(int nIndex, unsigned int ulRomAdress, int nBytes, unsigned char *pData);


#ifdef __cplusplus
}
#endif

#endif