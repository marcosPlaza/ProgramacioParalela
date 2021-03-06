#include <iostream>
#include <omp.h>
#include <chrono>
#include <stdexcept>
#include <stdlib.h>
#include <string.h>

// 4k resolution
#define WIDTH 3840 // 8294400
#define HEIGHT 2160

#define EXPERIMENT_ITERATIONS 1000

typedef unsigned char uchar;
using namespace std;

// 3 bytes in memory; x=g, y=r, z=bs
// Exercici 4
/*struct _uchar3
{
    uchar x;
    uchar y;
    uchar z;
} __attribute__((aligned(4)));*/

// Exercici 1-3
struct _uchar3
{
    uchar x;
    uchar y;
    uchar z;
};

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
void convertGRB2RGBA(uchar3 *grb, uchar4 *rgba, int width, int height,int chunk_size)
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

void convertGRB2RGBA_2(uchar3 *grb, uchar4 *rgba, int width, int height,int chunk_size)
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

void convertGRB2RGBA_3(uchar3 *grb, uchar4 *rgba, int width, int height,int chunk_size)
{

    #pragma omp parallel for // exercici 5 a1 - Better choice
    for (int y = 0; y < height; ++y)
    {
        //#pragma omp parallel for // exercici 5 a2
        for (int x = 0; x < width; ++x)
        {
            rgba[width * y + x].x = grb[width * y + x].y;
            rgba[width * y + x].y = grb[width * y + x].x;
            rgba[width * y + x].z = grb[width * y + x].z;
            rgba[width * y + x].w = 255;
        }
    }
}

void convertGRB2RGBA_onefor(uchar3 *grb, uchar4 *rgba, int width, int height,int chunk_size){
    #pragma omp parallel for
    for (int i = 0; i < height*width; i++){
        rgba[i].x = grb[i].y;
        rgba[i].y = grb[i].x;
        rgba[i].z = grb[i].z;
        rgba[i].w = 255;
    }
}

/* SCHEDULING FUNTIONS */

// Static
void convertGRB2RGBA_2_staticscheduling(uchar3 *grb, uchar4 *rgba, int width, int height,int chunk_size)
{
    #pragma omp for schedule(static, chunk_size)
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

// Dynamic
void convertGRB2RGBA_2_dynamicscheduling(uchar3 *grb, uchar4 *rgba, int width, int height, int chunk_size)
{
    #pragma omp for schedule(dynamic, chunk_size)
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

// Guided
void convertGRB2RGBA_2_guidedscheduling(uchar3 *grb, uchar4 *rgba, int width, int height, int chunk_size)
{
    #pragma omp for schedule(guided, chunk_size)
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

// TODO make the one foor loop implementation
ostream& bold_on(ostream& os)
{
    return os << "\e[1m";
}

std::ostream& bold_off(std::ostream& os)
{
    return os << "\e[0m";
}


void help(){
    cout<<"./main "<< bold_on <<"<conversion_function> <number_of_executions> <chuck_size>"<<bold_off<<endl;
    cout<<"\n"<<bold_on<<"<conversion_function>"<<bold_off<<endl;
    cout<<"\tInteger that represents the conversion function among the three implemented."<<endl;
    cout<<"\t1 for convertGRB2RGBA"<<endl;
    cout<<"\t2 for convertGRB2RGBA_2"<<endl;
    cout<<"\t3 for convertGRB2RGBA_3"<<endl;
    cout<<"\t4 for convertGRB2RGBA_onefor"<<endl;
    cout<<"\t5 for convertGRB2RGBA_2_staticscheduling"<<endl;
    cout<<"\t6 for convertGRB2RGBA_2_dynamicscheduling"<<endl;
    cout<<"\t7 for convertGRB2RGBA_2_guidedscheduling"<<endl;
    cout<<"\n"<<bold_on<<"<number_of_executions>"<<bold_off<<endl;
    cout<<"\tInteger that represents the number of executions that has to be done by the program."<<endl;
    cout<<"\n"<<bold_on<<"<chunck_size>"<<bold_off<<endl;
    cout<<"\tInteger that represents the chunck_size on scheduling functions. This value is optional."<<endl;
}

// Select the function you want to use on runtime
void (*func_ptr[7])(uchar3*, uchar4*, int, int, int) = {
    convertGRB2RGBA,
    convertGRB2RGBA_2,
    convertGRB2RGBA_3,
    convertGRB2RGBA_onefor,
    convertGRB2RGBA_2_staticscheduling,
    convertGRB2RGBA_2_dynamicscheduling,
    convertGRB2RGBA_2_guidedscheduling
};

int main(int argc, char *argv[])
{
    // Input control
    int conv_fun;
    int num_executions;
    int chunk_size=1;
    int sum = 0;
    string fun;
    
    if(argc == 2 && strcmp(argv[1],"-h")==0){
        help();
        exit(1);
    }else if(argc == 3 || argc == 4){
        try{
            conv_fun = stoi(argv[1]) - 1;
            num_executions = stoi(argv[2]);
            chunk_size = 1;
            
            //Input values control
            if (conv_fun < 0 || conv_fun > 6 || num_executions < 1){
                cerr<<"Error: invalid values."<<endl;
                exit(1);
            }
            
            if(argc == 3 && (conv_fun == 4 || conv_fun == 5 || conv_fun == 6)){
                cerr<<"Warning: Scheduling conversion function introduced. "<<bold_on<<"<chunk_size>"<<bold_off<<" setted to 1."<<endl;
            }else if(argc == 4){
                chunk_size = stoi(argv[3]);
                if(conv_fun == 0 || conv_fun == 1 || conv_fun == 2 || conv_fun == 3){
                    cerr<<"Error: Invalid conversion function. "<<bold_on<<"<chunk_size>"<<bold_off<<" value was specified!"<<endl;
                    exit(1);
                } else if (chunk_size > HEIGHT || chunk_size < 1){
                    cerr<<"Error: invalid values."<<endl;
                    exit(1);
                }
            }
                
        }catch (std::invalid_argument const &e){
            std::cout << "Bad input: std::invalid_argument thrown" << '\n';
        }
        catch (std::out_of_range const &e){
            std::cout << "Integer overflow: std::out_of_range thrown" << '\n';
        }
    }else{
        cerr<<"Error: invalid argument on the input: ./main <conversion_function> <number_of_executions> <chunck_size>"<<endl;
        cout<<"Please use -h option to see more information about the input rules."<<endl;
        exit(1);
    }
    
    void (*func_ptr_conv_fun)(uchar3*, uchar4*, int, int, int);
    func_ptr_conv_fun = func_ptr[conv_fun];
                    
    cout<<"-------------------------------------------------------------------"<<endl;
    cout<<bold_on<<" CONFIGURATION"<<bold_off<<endl;
    cout<<"\tThe number of executions is setted to "<<num_executions<<"."<<endl;
    
    switch(conv_fun){
        case 0:
            fun = "convGRB2RGBA";
            break;
        case 1:
            fun = "convGRB2RGBA_2";
            break;
        case 2:
            fun = "convGRB2RGBA_3";
            break;
        case 3:
            fun = "convGRB2RGBA_onefor";
            break;
        case 4:
            fun = "convGRB2RGBA_2_staticscheduling";
            break;
        case 5:
            fun = "convGRB2RGBA_2_dynamicscheduling";
            break;
        case 6:
            fun = "convGRB2RGBA_2_guidedscheduling";
            break;
    }

    cout<<"\tThe conversion function selected is "<<bold_on<<fun<<bold_off<<"."<<endl;
    cout<<"\tThe "<<bold_on<<"chunk_size "<<bold_off<<"is setted to "<<chunk_size<<"."<<endl;
    cout<<"-------------------------------------------------------------------"<<endl;
    cout<<bold_on<<" EXECUTION"<<bold_off<<endl;
    
    uchar3 *h_grb; // definition of the two objects grb and rgba
    uchar4 *h_rgba;

    int bar_widht = HEIGHT / 3; // 720

    // Allocate and generate BRG bars.
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

    // Allocate RGBA pointers
    h_rgba = (uchar4 *)malloc(sizeof(uchar4) * WIDTH * HEIGHT);

    for (int j = 0; j < num_executions; j++)
    {
        auto t1 = chrono::high_resolution_clock::now();
        
        // for sin paralelizar
         /*for (int i = 0; i < EXPERIMENT_ITERATIONS; i++)
             func_ptr_conv_fun(h_grb, h_rgba, WIDTH, HEIGHT, chunk_size);*/
         
        
        //for paralelizado. Comentar este y descomentar el anterior si se requiere.
        #pragma omp parallel //exercice 6 & 7
        {
            //unicamente imprimimos una vez 
            #pragma omp critical
            {
                cout<<"\tHey I'm the thread with id "<< omp_get_thread_num() <<"."<<endl;   
            }
            
            //cada hilo se reparte el trabajo
            #pragma omp for
            for (int i = 0; i < EXPERIMENT_ITERATIONS; ++i)
            {
                func_ptr_conv_fun(h_grb, h_rgba, WIDTH, HEIGHT, chunk_size);
            }
        }
        
        auto t2 = chrono::high_resolution_clock::now();

        auto duration = chrono::duration_cast<chrono::microseconds>(t2 - t1).count();
        cout <<"\t"<< fun <<" time for " << EXPERIMENT_ITERATIONS
                  << " iterations = " << duration << "??s" << endl;

        bool ok = checkResults(h_rgba, h_grb, WIDTH * HEIGHT);

        if (ok)
        {
            sum += duration;
            cout << "\tExecuted!! Results OK." << endl;
        }
        else
        {
            cout << "\tExecuted!! Results NOT OK." << endl;
        }
    }

    float mean = sum/num_executions;
    cout << "\n\tThe mean of time (execution) in " << num_executions << " executions is " << mean <<"??s"<<endl;
    cout<<"-------------------------------------------------------------------"<<endl;

    return 0;
}
