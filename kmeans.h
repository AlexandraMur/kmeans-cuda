#ifndef KMEANS_H
#define KMEANS_H

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


/*typedef struct {
  int points[DIMENSIONS][N];
  int clusterId[N];
} Points;
*/

static double GetDistance(Point p1, Point p2)
{
    double distance = 0;

    for (int i=0; i < DIMENSIONS; i++) {
      distance += SQUARE(p2.loc[i] - p1.loc[i]);
    }

    return sqrt(distance);
}

#if defined (GPU)
static __device__ double GetDistanceGPU(Point p1, Point p2)
{
    double distance = 0;

    for (int i=0; i < DIMENSIONS; i++) {
      distance += SQUARE(p2.loc[i] - p1.loc[i]);
    }

    return sqrt(distance);
}
#endif

#endif
