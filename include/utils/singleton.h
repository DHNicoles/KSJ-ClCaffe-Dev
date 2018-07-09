#ifndef __SINGLETON_H__
#define __SINGLETON_H__
#include "util.h"  
namespace ice
{
	template<typename T>
	class Singleton
	{
	public:
		Singleton()
		{
			assert(!ms_pSingleton);
			ms_pSingleton = static_cast<T*>(this);
		}
		~Singleton()
		{
			assert(ms_pSingleton);
			ms_pSingleton = NULL;
		}

		static T* Instance()
		{
			return ms_pSingleton;
		}

	protected:
		static T* ms_pSingleton;
	};

	template<typename T>
	T* Singleton<T>::ms_pSingleton = NULL;
}


#endif // !__SINGLETON_H__
