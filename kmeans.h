#ifndef COMMON_H
#define COMMON_H

#define N 100000
#define K 10

#ifdef DEBUG
#define NDEBUG
#endif

#define X_AXIS 0
#define Y_AXIS 1
#define DIMENSIONS 2

#define SQUARE(x) ((x)*(x))
#define MOD(x) (((x) >= 0) ? (x) : (-(x)))

typedef struct {
    double loc[DIMENSIONS];
    int clusterId;
} Point;

typedef struct {
    Point pt; // Centroid+clusterId
    unsigned int noOfPoints;
} Cluster;


static double GetDistance(Point p1, Point p2)
{
    return sqrt(SQUARE(p2.loc[X_AXIS]-p1.loc[X_AXIS]) + SQUARE(p2.loc[Y_AXIS]-p1.loc[Y_AXIS]));
}

#if defined (GPU)
static __device__ double GetDistanceGPU(Point p1, Point p2)
{
    return sqrt(SQUARE(p2.loc[X_AXIS]-p1.loc[X_AXIS]) + SQUARE(p2.loc[Y_AXIS]-p1.loc[Y_AXIS]));
}
#endif

#endif
