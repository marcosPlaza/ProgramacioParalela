#include <iostream>
#include <omp.h>
#include <chrono>

// 4k resolution
#define WIDTH 3840 // 8294400
#define HEIGHT 2160

#define EXPERIMENT_ITERATIONS 1000

typedef unsigned char uchar;

// 3 bytes in memory; x=g, y=r, z=b
struct _uchar3
{
    uchar x;
    uchar y;
    uchar z;
} __attribute__((aligned(4)));

using uchar3 = _uchar3;

// 4 bytes in memory; x=r, y=g, z=b, w=a
struct _uchar4
{
    uchar x;
    uchar y;
    uchar z;
    uchar w;
};

using uchar4 = _uchar4;

// checking if the conversion was correct
bool checkResults(uchar4 *rgba, uchar3 *grb, int size)
{

    bool correct = true;

    for (int i = 0; i < size; ++i)
    {
        correct &= rgba[i].x == grb[i].y;
        correct &= rgba[i].y == grb[i].x;
        correct &= rgba[i].z == grb[i].z;
        correct &= rgba[i].w == 255;
    }

    return correct;
}

// function to converto to grb space to rgba(with full opacity)
// Note that we are accessing to bytes that are not close each other in every iteration
// Reverse the order of the fors => Improve space locality
void convertGRB2RGBA(uchar3 *grb, uchar4 *rgba, int width, int height)
{
    for (int x = 0; x < width; ++x)
    {
        for (int y = 0; y < height; ++y)
        {
            rgba[width * y + x].x = grb[width * y + x].y;
            rgba[width * y + x].y = grb[width * y + x].x;
            rgba[width * y + x].z = grb[width * y + x].z;
            rgba[width * y + x].w = 255;
        }
    }
}

void convertGRB2RGBA_2(uchar3 *grb, uchar4 *rgba, int width, int height)
{
    /*
        Reversing the order of the for loops...
        we go from rgba[3840*0+0] = grb[3840*0+0] (position = 0)
        then rgba[3840*1+0] = grb[3840*1+0] (position = 3840)
        to rgba[3840*0+0] = grb[3840*0+0] (position = 0)
        then rgba[3840*0+1] = grb[3840*0+1] (position = 1 contiguous position)
    */

    for (int y = 0; y < height; ++y)
    {
        for (int x = 0; x < width; ++x)
        {
            rgba[width * y + x].x = grb[width * y + x].y;
            rgba[width * y + x].y = grb[width * y + x].x;
            rgba[width * y + x].z = grb[width * y + x].z;
            rgba[width * y + x].w = 255;
        }
    }
}

void convertGRB2RGBA_3(uchar3 *grb, uchar4 *rgba, int width, int height)
{
    #pragma omp parallel for
    for (int y = 0; y < height; ++y)
    {
        for (int x = 0; x < width; ++x)
        {
            rgba[width * y + x].x = grb[width * y + x].y;
            rgba[width * y + x].y = grb[width * y + x].x;
            rgba[width * y + x].z = grb[width * y + x].z;
            rgba[width * y + x].w = 255;
        }
    }
}

int main(int argc, char *argv[])
{

    uchar3 *h_grb; // definition of the two objects grb and rgba
    uchar4 *h_rgba;

    int bar_widht = HEIGHT / 3; // 720

    // Alloc and generate BRG bars.
    h_grb = (uchar3 *)malloc(sizeof(uchar3) * WIDTH * HEIGHT); // allocate the space for one image
    for (int i = 0; i < WIDTH * HEIGHT; ++i)
    {
        // fills the vector 0-720 with g=255
        // then 721-1440 with r=255
        // finally 1440-8294400 with b=255
        if (i < bar_widht)
        {
            h_grb[i] = {255, 0, 0};
        }
        else if (i < bar_widht * 2)
        {
            h_grb[i] = {0, 255, 0};
        }
        else
        {
            h_grb[i] = {0, 0, 255};
        }
    }

    // Alloc RGBA pointers
    h_rgba = (uchar4 *)malloc(sizeof(uchar4) * WIDTH * HEIGHT);
    
    int num_executions = 1;
    int sum = 0;

    if (argc != 0)
        num_executions = atoi(argv[1]);

    for (int j = 0; j < num_executions; j++)
    {
        auto t1 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < EXPERIMENT_ITERATIONS; ++i)
        {
            convertGRB2RGBA_3(h_grb, h_rgba, WIDTH, HEIGHT);
        }
        auto t2 = std::chrono::high_resolution_clock::now();

        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
        std::cout << "convertGRB2RGBA_3 time for " << EXPERIMENT_ITERATIONS
                  << " iterations = " << duration << "us" << std::endl;

        bool ok = checkResults(h_rgba, h_grb, WIDTH * HEIGHT);

        if (ok)
        {
            sum += duration;
            std::cout << "Executed!! Results OK." << std::endl;
        }
        else
        {
            std::cout << "Executed!! Results NOT OK." << std::endl;
        }
    }

    float mean = sum/num_executions;
    std::cout << "The mean of time (execution) in " << num_executions << " executions is " << mean <<"s"<<std::endl;

    return 0;
}