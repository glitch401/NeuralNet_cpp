#include <iostream>
#include <cmath>
#include <cstdlib>
#include <fstream>

using namespace std;

int main(int argc, char **argv)
{
    
    //Generating random XOR data with two inputs and an output
    /*
    Pass200001: Inputs : 1 1 
: Outputs : 0.998817 
: targets : 1 
Net recent average error: 0.0005848
    */
   //cout<<argc<<endl;
   //cout<<argv[1]<<endl;
   int numTrainData=20000;

    if(argc>1)
        numTrainData=atoi(argv[1]);
    
    cout<<"Creating "<<numTrainData<<" training samples to trainingData.txt\n";
    ofstream myfile;
    myfile.open ("trainingData.txt");


    myfile<<"topology: 2 4 1\n";
    for(int i=numTrainData;i>=0;--i)
    {
        int num1= (int)(2.0 *rand() /double(RAND_MAX) ),
            num2= (int)(2.0 *rand() /double(RAND_MAX) ),
            t=num1^num2;
            myfile<<"in: "<<num1<<".0 "<<num2<<".0 \n";
            myfile<<"out: "<<num1<<".0 \n";
    }
    myfile.close();

    return 0;

}