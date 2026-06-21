#include "RHSoperator.h"

template<class T>
RHSOperator<T>::RHSOperator()
{

}


template<class T>
RHSOperator<T>::~RHSOperator()
{

}

template<class T>
Central1D<T>::Central1D(DataStruct<T> &_U, 
                     DataStruct<T> &_mesh, 
                     FluxFunction<T> &_F):
U(_U), mesh(_mesh), F(_F)
{
  RHS.setSize(_U.getSize());
}

template<class T>
Central1D<T>::~Central1D()
{

}

template<class T>
void Central1D<T>::evalRHS(DataStruct<T> &Uin)
{
  // the BC should be included in the mesh
  // momentarily done here by hand
  T *dataRHS = RHS.getData();
  const T *dataU = Uin.getData();
  const T *dataMesh = mesh.getData();
  const int len = U.getSize();
  
  for(int j = 0; j < len; j++)
  {
    T dx;
    if(j == 0)
    {
      dx = dataMesh[len-1] - dataMesh[len-2];
      dx += dataMesh[1] - dataMesh[0];
      dataRHS[0] = -(F.computeFlux(dataU[1]) - F.computeFlux(dataU[len-2]))/dx;
    }
    else
    {
      dx = dataMesh[j+1] - dataMesh[j-1];
      dataRHS[j] = -(F.computeFlux(dataU[j+1]) - F.computeFlux(dataU[j-1]))/dx;
    }
  }

  dataRHS[len-1] = dataRHS[0];
}
template<class T>
void Central1D<T>::evalRHS(
    DataStruct<T> &Uin,
    T ghostLeft,
    T ghostRight
) {
    T *dataRHS = RHS.getData();

    const T *dataU = Uin.getData();
    const T *dataMesh = mesh.getData();

    const int len = Uin.getSize();

    T h = dataMesh[1] - dataMesh[0];
    T dx = 2.0 * h;

    for(int j = 0; j < len; j++) {

        T uLeft;
        T uRight;

        if(j == 0) {
            uLeft = ghostLeft;
        } else {
            uLeft = dataU[j - 1];
        }

        if(j == len - 1) {
            uRight = ghostRight;
        } else {
            uRight = dataU[j + 1];
        }

        dataRHS[j] = -(
            F.computeFlux(uRight) - F.computeFlux(uLeft)
        ) / dx;
    }
}
template<class T>
void Central1D<T>::eval()
{
  evalRHS(U);
}

template<class T>
void Central1D<T>::eval(DataStruct<T> &Uin)
{
  evalRHS(Uin);
}
template<class T>
void Central1D<T>::eval(
    DataStruct<T> &Uin,
    T ghostLeft,
    T ghostRight
) {
    evalRHS(Uin, ghostLeft, ghostRight);
}

template<class T>
DataStruct<T>& Central1D<T>::ref2RHS()
{
  return RHS;
}


template class RHSOperator<float>;
template class RHSOperator<double>;

template class Central1D<float>;
template class Central1D<double>;
