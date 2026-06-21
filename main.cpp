#include <iostream> // std::cout
#include <fstream> // ofile
#include <string>
#include <math.h>
#include <iomanip> // set precision
#include <mpi.h>

#include "DataStructs.h"
#include "rk4.h"
#include "FluxFunctions.h"
#include "RHSoperator.h"

#ifdef _DOUBLE_
#define FLOATTYPE double
#else
#define FLOATTYPE float
#endif

// declare supporting functions
void write2File(DataStruct<FLOATTYPE> &X, DataStruct<FLOATTYPE> &U, std::string name);
FLOATTYPE calcL2norm(DataStruct<FLOATTYPE> &u, DataStruct<FLOATTYPE> &uinit);
void updateGhosts(
    DataStruct<FLOATTYPE> &U,
    FLOATTYPE &ghostLeft,
    FLOATTYPE &ghostRight,
    int worldRank,
    int worldSize
);

int main(int narg, char **argv)
{
  MPI_Init(&narg, &argv);

  int worldRank, worldSize;
  MPI_Comm_rank(MPI_COMM_WORLD, &worldRank);
  MPI_Comm_size(MPI_COMM_WORLD, &worldSize);
  int numPoints =  80;
  FLOATTYPE k = 2.; // wave number

  if(narg != 3)
  {
    std::cout<< "Wrong number of arguments. You should include:" << std::endl;
    std::cout<< "    Num points" << std::endl;
    std::cout<< "    Wave number" << std::endl;
    MPI_Finalize(); 
    return 1;
  }else
  {
    numPoints = std::stoi(argv[1]);
    k         = std::stod(argv[2]);
  }
    int baseLocalSize = numPoints / worldSize;
    int remainder = numPoints % worldSize;

    int localNumPoints = baseLocalSize;

    if(worldRank == worldSize - 1) {
        localNumPoints += remainder;
    }

    int globalStart = worldRank * baseLocalSize; 
/*
  std::cout << "Rank " << worldRank
          << " de " << worldSize
          << " tiene " << localNumPoints
          << " puntos y empieza en " << globalStart
          << std::endl;
*/
  // solution data
  DataStruct<FLOATTYPE> u(localNumPoints);
  DataStruct<FLOATTYPE> f(localNumPoints);
  DataStruct<FLOATTYPE> xj(localNumPoints);
  FLOATTYPE ghostLeft = 0.0;
  FLOATTYPE ghostRight = 0.0;
  // flux function
  LinearFlux<FLOATTYPE> lf;

  // time solver
  RungeKutta4<FLOATTYPE> rk(u);

  // Initial Condition
  FLOATTYPE *datax = xj.getData();
FLOATTYPE *dataU = u.getData();

for(int j = 0; j < localNumPoints; j++) {

    int globalIndex = globalStart + j;

    datax[j] = FLOATTYPE(globalIndex) / FLOATTYPE(numPoints);

    dataU[j] = sin(k * 2.0 * M_PI * datax[j]);
}

  DataStruct<FLOATTYPE> Uinit;
  Uinit = u;

  // Operator
  Central1D<FLOATTYPE> rhs(u,xj,lf);

  FLOATTYPE CFL = 2.4;
FLOATTYPE dxGlobal = 1.0 / FLOATTYPE(numPoints);
FLOATTYPE dt = CFL * dxGlobal;

  // Output Initial Condition
  write2File(xj, u, "initialCondition.csv");

  FLOATTYPE t_final = 1.;
  FLOATTYPE time = 0.;
  DataStruct<FLOATTYPE> Ui(u.getSize()); // temp. data

  // init timer
  double compTime = MPI_Wtime();

  // main loop
  while(time < t_final)
  {
    if(time+dt >= t_final) dt = t_final - time;

    // take RK step
    rk.initRK();
    for(int s = 0; s < rk.getNumSteps(); s++)
    {
      rk.stepUi(dt);
      Ui = *rk.currentU();
      updateGhosts(Ui, ghostLeft, ghostRight, worldRank, worldSize);
      rhs.eval(Ui, ghostLeft, ghostRight);
      rk.setFi(rhs.ref2RHS());
    }
    rk.finalizeRK(dt);
    time += dt;
  }

  // finisher timer
  double localCompTime = MPI_Wtime() - compTime;
double globalCompTime = 0.0;

MPI_Reduce(
    &localCompTime,
    &globalCompTime,
    1,
    MPI_DOUBLE,
    MPI_MAX,
    0,
    MPI_COMM_WORLD
);

  std::string fileName = "final_rank_" + std::to_string(worldRank) + ".csv";
  write2File(xj, u, fileName);

  // L2 norm
  FLOATTYPE err = calcL2norm(Uinit, u);
if(worldRank == 0){
  std::cout << std::setprecision(6);
  std::cout << "Processes: " << worldSize;
  std::cout << " Comp. time: " << globalCompTime;
  std::cout << " sec. Error: " << err/k;
  std::cout << " kdx: " << k*dxGlobal*2.*M_PI;
  std::cout << std::endl;
}
  MPI_Finalize();
  return 0;
}


// ==================================================================
// AUXILIARY FUNCTIONS
// ==================================================================
void write2File(DataStruct<FLOATTYPE> &X, DataStruct<FLOATTYPE> &U, std::string name)
{
  std::ofstream file;
  file.open(name,std::ios_base::trunc);
  if(!file.is_open()) 
  {
    std::cout << "Couldn't open file for Initial Condition" << std::endl;
    exit(1);
  }
  
  for(int j = 0; j < U.getSize(); j++)
  {
    file << X.getData()[j] << " ," << U.getData()[j] << std::endl;
  }

  file.close();
}

FLOATTYPE calcL2norm(DataStruct<FLOATTYPE> &u, DataStruct<FLOATTYPE> &uinit)
{
  FLOATTYPE err = 0.;
  const FLOATTYPE *dataU = u.getData();
  const FLOATTYPE *dataInit = uinit.getData();

  for(int n = 0; n < u.getSize(); n++)
  {
    err += (dataU[n] - dataInit[n])*(dataU[n] - dataInit[n]);
  }

  return sqrt( err );
}
void updateGhosts(
    DataStruct<FLOATTYPE> &U,
    FLOATTYPE &ghostLeft,
    FLOATTYPE &ghostRight,
    int worldRank,
    int worldSize
) {
    FLOATTYPE *dataU = U.getData();

    int localSize = U.getSize();

    int leftRank = (worldRank - 1 + worldSize) % worldSize;
    int rightRank = (worldRank + 1) % worldSize;

#ifdef _DOUBLE_
    MPI_Datatype mpiFloatType = MPI_DOUBLE;
#else
    MPI_Datatype mpiFloatType = MPI_FLOAT;
#endif

    MPI_Request requests[4];

    /*
      ghostLeft recibe el último valor del proceso izquierdo.
      ghostRight recibe el primer valor del proceso derecho.
    */

    MPI_Irecv(
        &ghostLeft,
        1,
        mpiFloatType,
        leftRank,
        100,
        MPI_COMM_WORLD,
        &requests[0]
    );

    MPI_Irecv(
        &ghostRight,
        1,
        mpiFloatType,
        rightRank,
        200,
        MPI_COMM_WORLD,
        &requests[1]
    );

    /*
      Envío mi primer punto al proceso izquierdo,
      porque para él será su ghostRight.
    */

    MPI_Isend(
        &dataU[0],
        1,
        mpiFloatType,
        leftRank,
        200,
        MPI_COMM_WORLD,
        &requests[2]
    );

    /*
      Envío mi último punto al proceso derecho,
      porque para él será su ghostLeft.
    */

    MPI_Isend(
        &dataU[localSize - 1],
        1,
        mpiFloatType,
        rightRank,
        100,
        MPI_COMM_WORLD,
        &requests[3]
    );

    MPI_Waitall(4, requests, MPI_STATUSES_IGNORE);
}
