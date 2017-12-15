#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <stdbool.h>
#include <string.h>
#include <time.h>
#include "timer.c"

#include "kmeans.h"

#if defined GPU
// From the NVIDIA CUDA programming guide. No idea how it works
__device__ double atomicAdd(double *address, double val)
{
    double old = *address, assumed;
    do {
        assumed = old;
        old = __longlong_as_double(atomicCAS((unsigned long long int*)address,
                                             __double_as_longlong(assumed),
                                             __double_as_longlong(val + assumed)));
    } while (assumed != old);
    return old;
}

__global__ void ResetCentroidForEachCluster(Cluster *clusters)
{
    int pt = blockIdx.x*blockDim.x + threadIdx.x;
    // Set clusters[].noOfPoints to 0. Only for K centroids
    if (pt < K) {
        clusters[pt].noOfPoints = 0;
    }
}

__global__ void ComputeClusters(Point *points, Cluster *clusters, Point *tempPoints)
{
    int pt = blockIdx.x * blockDim.x + threadIdx.x;
    int i;
    double max;
    int inCluster;

    if (pt >= N)
        return;

    // Save the old centroid and clear the x and y components of
    // each point. I'm going to use first K of these to store
    // the sum of co-ordinates of points in this cluster.
    // clusterId field is used to save old centroid for each point
    // so that we know when to stop iterating.
    tempPoints[pt].clusterId = points[pt].clusterId;
    tempPoints[pt].loc[X_AXIS] = 0.0;
    tempPoints[pt].loc[Y_AXIS] = 0.0;

    // Compute the nearest centroid.
    max = GetDistanceGPU(points[pt], clusters[0].pt);
    inCluster = 0;
    for (i = 0; i < K; i++) {
        if (GetDistanceGPU(points[pt], clusters[i].pt) < max) {
            inCluster = i;
            max = GetDistanceGPU(points[pt], clusters[i].pt);
        }
    }
    atomicAdd(&clusters[inCluster].noOfPoints, 1);
    // Bottle neck.
    atomicAdd(&tempPoints[inCluster].loc[X_AXIS], points[pt].loc[X_AXIS]);
    atomicAdd(&tempPoints[inCluster].loc[Y_AXIS], points[pt].loc[Y_AXIS]);

    points[pt].clusterId = inCluster;
}

__global__ void ComputeCentroids(Cluster *clusters, Point *tempPoints)
{
    int pt = blockIdx.x*blockDim.x + threadIdx.x;

    // Now calculate the new centroids.
    if (pt < K) {
        clusters[pt].pt.loc[X_AXIS] = tempPoints[pt].loc[X_AXIS]/clusters[pt].noOfPoints;
        clusters[pt].pt.loc[Y_AXIS] = tempPoints[pt].loc[Y_AXIS]/clusters[pt].noOfPoints;
    }
}

__global__ void RepeatNeeded(Point *points, Point *tempPoints, unsigned int *key)
{
    int pt = blockIdx.x * blockDim.x + threadIdx.x;

    if (pt < N) {
        if (points[pt].clusterId != tempPoints[pt].clusterId) {
                *key = 1;
        }
    }

}

void DoKmeansGPU (Point *points, Cluster *clusters)
{
    Point *dPoints, *dTempPoints;
    Cluster *dClusters;
    unsigned int *repeat, repeatHost;

    cudaMalloc ((void **)&dPoints, sizeof(Point)*N);
    cudaMalloc ((void **)&dClusters, sizeof(Cluster)*K);
    cudaMalloc ((void **)&dTempPoints, sizeof(Point)*N);
    cudaMalloc ((void **)&repeat, sizeof(unsigned int));

    cudaMemcpy(dPoints, points, sizeof(Point)*N, cudaMemcpyHostToDevice);
    cudaMemcpy(dClusters, clusters, sizeof(Cluster)*K, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(256);
    dim3 blocksPerGrid (N / threadsPerBlock.x);

    do {
        ResetCentroidForEachCluster <<< blocksPerGrid, threadsPerBlock >>> (dClusters);
        ComputeClusters <<< blocksPerGrid, threadsPerBlock >>> (dPoints, dClusters, dTempPoints);
        ComputeCentroids <<< blocksPerGrid, threadsPerBlock >>> (dClusters, dTempPoints);

        cudaMemset(repeat, 0, sizeof(unsigned int));

        RepeatNeeded <<<blocksPerGrid, threadsPerBlock >>> (dPoints, dTempPoints, repeat);

        cudaMemcpy(&repeatHost, repeat, sizeof(unsigned int), cudaMemcpyDeviceToHost);
    } while (repeatHost);


    cudaMemcpy(points, dPoints, sizeof(Point)*N, cudaMemcpyDeviceToHost);
    cudaMemcpy(clusters, dClusters, sizeof(Cluster)*K, cudaMemcpyDeviceToHost);

    cudaFree(dPoints);
    cudaFree(dClusters);
    cudaFree(dTempPoints);
    cudaFree(repeat);
}
#endif // definedGPU

void DoKmeansCPU (Point *points, Cluster *clusters)
{

    double max;
    int i, j, inCluster;
    bool changed;
    Point *tempPoints;

    // One for each cluster (and point). For cluster, use x and y, for point, clusterId.
    tempPoints  = (Point *) malloc (sizeof(Point)*N);

    do {

        memset(tempPoints, 0, sizeof(Point)*N);
        for (i = 0; i < K; i++) {
            clusters[i].noOfPoints = 0;
        }
        // Save the old clusterId for each point. Reusing tempPoints
        for (i = 0; i < N; i++) {
            tempPoints[i].clusterId = points[i].clusterId;
        }
        // For each point, find the nearest centroid.
        for (i = 0; i < N; i++) {
            max = GetDistance(points[i], clusters[0].pt);
            inCluster = 0;
            for (j = 0; j < K; j++) {
                if (GetDistance(points[i], clusters[j].pt) < max) {
                    inCluster = j;
                    // TODO: Store these distances, instead of re-computing
                    // (I don't mean from above call, I mean totally for the program).
                    max = GetDistance(points[i], clusters[j].pt);
                }
            }
            clusters[inCluster].noOfPoints++;
            tempPoints[inCluster].loc[X_AXIS] += points[i].loc[X_AXIS];
            tempPoints[inCluster].loc[Y_AXIS] += points[i].loc[Y_AXIS];
            points[i].clusterId = inCluster;
        }

        // Compute new centroid for each cluster
        for (i = 0; i < K; i++) {
            // Assuming that each cluster has atleast one point in it.
            assert(clusters[i].noOfPoints != 0);
            clusters[i].pt.loc[X_AXIS] = tempPoints[i].loc[X_AXIS]/clusters[i].noOfPoints;
            clusters[i].pt.loc[Y_AXIS] = tempPoints[i].loc[Y_AXIS]/clusters[i].noOfPoints;
        }

        // Check if anything has changed
        changed = false;
        for (i = 0; i < N; i++) {
            if (points[i].clusterId != tempPoints[i].clusterId) {
                changed = true;
                break;
            }
        }
    } while (changed);

}

int main (int argc, char *argv[])
{

    Point *pointsCPU;
    Cluster *clustersCPU;
    int i, j;

#ifdef GPU
    Point *pointsGPU;
    Cluster *clustersGPU;
#endif

    srandom(time(NULL));

    pointsCPU = (Point *) malloc(sizeof(Point)*N);
    clustersCPU = (Cluster *) malloc(sizeof(Cluster)*K);

    // Get the points randomly
    for (i = 0; i < N; i++) {
        for (int d=0; d < DIMENSIONS; d++) {
          pointsCPU[i].loc[d] = (random()/1021322);
        }
        pointsCPU[i].clusterId = -1;
    }

    // Initialize clusters
    for (i = 0; i < K; i++) {
        clustersCPU[i].pt.clusterId = i;
        clustersCPU[i].noOfPoints = 0;
        j = random()%N;
        if (pointsCPU[j].clusterId != -1) {
            i--; continue;
            // Potential infinite loop
        }
        pointsCPU[j].clusterId = i;
        clustersCPU[i].pt.loc[X_AXIS] = pointsCPU[j].loc[X_AXIS];
        clustersCPU[i].pt.loc[Y_AXIS] = pointsCPU[j].loc[Y_AXIS];
    }

#ifdef DEBUG
    printf ("Initial points:\n");
    for (i = 0; i < N; i++) {
        printf ("x=%.2f, y=%.2f, clusterId=%d\n", pointsCPU[i].loc[X_AXIS], pointsCPU[i].loc[Y_AXIS], pointsCPU[i].clusterId);
    }
    printf ("Initial clusters:\n");
    for (i = 0; i < K; i++) {
        printf("clusterId=%d, noOfPoints=%d, centroidX=%.2f, centroidY=%.2f\n", clustersCPU[i].pt.clusterId,
               clustersCPU[i].noOfPoints, clustersCPU[i].pt.loc[X_AXIS], clustersCPU[i].pt.loc[Y_AXIS]);
    }
#endif // DEBUG

#ifdef GPU
    pointsGPU = (Point *) malloc (sizeof(Point)*N);
    clustersGPU = (Cluster *) malloc (sizeof(Cluster)*K);

    memcpy(pointsGPU, pointsCPU, sizeof(Point)*N);
    memcpy(clustersGPU, clustersCPU, sizeof(Cluster)*K);

    tstart();
    DoKmeansGPU(pointsGPU, clustersGPU);
    tend();
    printf("%f seconds on GPU.\n", tval());
#endif

    // Data for other versions are copied from here. So don't want it
    // to change before copying.
    tstart();
    DoKmeansCPU(pointsCPU, clustersCPU);
    tend();
    printf("%f seconds on CPU.\n", tval());


#ifdef PRETTY_PRINT
#ifdef GPU
    // Showing GPU dumps
    FILE *fp; char buf[20];
    system ("rm /tmp/*plot");
    for (i = 0; i < N; i++) {
        sprintf(buf, "/tmp/%d.plot", pointsGPU[i].clusterId);
        fp = fopen (buf, "a");
        if (fp) {
            fprintf (fp, "%.2f %.2f #%d GPU\n", pointsGPU[i].loc[X_AXIS], pointsGPU[i].loc[Y_AXIS], pointsGPU[i].clusterId);
            fclose(fp);
        }
    }
#else
    // Showing CPU dumps
    FILE *fp; char buf[20];
    system ("rm /tmp/*plot");
    for (i = 0; i < N; i++) {
        sprintf(buf, "/tmp/%d.plot", pointsCPU[i].clusterId);
        fp = fopen (buf, "a");
        if (fp) {
            fprintf (fp, "%.2f %.2f #%d CPU\n", pointsCPU[i].loc[X_AXIS], pointsCPU[i].loc[Y_AXIS], pointsCPU[i].clusterId);
            fclose(fp);
        }
    }
#endif // #if defined (GPU)
#endif // PRETTY_PRINT

    return 0;
}
