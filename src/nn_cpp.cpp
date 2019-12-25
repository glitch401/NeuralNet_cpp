#include <vector>
#include <iostream>
#include <cstdlib>
#include <cassert>
#include <cmath>
#include <fstream>
#include <sstream>
using namespace std;

struct Connection
{
    double weight;
    double deltaweight;
};

/*
// to make the Connection as a class
class connection{
    public:
        Connection();
        Connection(unsigned double w,unsigned double dw)
    private:
       double weight,deltaweight; 
}
Connectio

*/


class TrainingData
{
    public:
        TrainingData(const string filename);
        bool isEof(void){return m_trainingDataFile.eof();}
        void getTopology(vector<unsigned> &topology);
        
        //Returns the no. of input values read from the file:
        unsigned getNextInputs(vector<double> &inputVals);
        unsigned getTargetOutputs(vector<double> &targetOutputVals);
    private:
        ifstream m_trainingDataFile; 
};
void TrainingData::getTopology(vector<unsigned> &topology)
{
    string line, label;
    getline(m_trainingDataFile, line);
    stringstream ss(line);
    ss>>label;
    if(this->isEof() || label.compare("topology:")!=0){
        abort();
    }
    while(!ss.eof()){
        unsigned n;
        ss>>n;
        topology.push_back(n);
    }
    return;
}

TrainingData::TrainingData(const string filename)
{
    m_trainingDataFile.open(filename.c_str());
}

unsigned TrainingData::getNextInputs(vector<double> &inputVals)
{
    inputVals.clear();
    
    string line, label;
    getline(m_trainingDataFile,line);
    stringstream ss(line);

    ss>>label;
    if(label.compare("in:")==0){
        double oneVal;
        while(ss>>oneVal){
            inputVals.push_back(oneVal);
        }
    }
    return inputVals.size();
}

unsigned TrainingData::getTargetOutputs(vector<double> &targetOutputVals)
{
    targetOutputVals.clear();

    string line, label;
    getline(m_trainingDataFile,line);
    stringstream ss(line);

    ss>>label;
    if(label.compare("out:")==0){
        double oneVal;
        while(ss>>oneVal){
            targetOutputVals.push_back(oneVal);
        }
    }
    return targetOutputVals.size();

}

class Neuron;

typedef vector<Neuron> Layer;

//**************** class Neuron ***************

class Neuron
{
    public:
        Neuron(unsigned numOutputs, unsigned myIndex);
        void setOutputVal(double val){m_outputVal=val;}
        double getOutputVal(void) const {return m_outputVal;}
        void feedForward(const Layer &prevLayer);
        void calculateOutputGradients(double targetVal);
        void calculateHiddenGradients(const Layer &nextLayer);
        void updateInputWeights(Layer &prevLayer);

    private:
        static double eta; //range: [0.0 .... 1.0] overall net trainig rate
        static double alpha; //range [0.0 ... n] multiplier to the last weight change(momentum)
        static double activationFunction(double x);
        static double activationFunctionDerivative(double x);
        //randomweight: 0 - 1
        static double randomWeight(void){ return rand()/double(RAND_MAX);}
        double sumDOW(const Layer &nextLayer) const;
        double m_outputVal;
        vector<Connection> m_outputWeights;
        unsigned m_myIndex;
        double m_gradient;
        

};

double Neuron::eta =0.15; //newt learning rate
double Neuron::alpha =0.5; //momentum

void Neuron::updateInputWeights(Layer &prevLayer)
{
    //The wtights to be updated are int rhe Connection container
    //in the neuroins in the preceding layer

    for(unsigned n=0;n<prevLayer.size();n++){
        Neuron &neuron=prevLayer[n];
        double oldDeltaWieght=neuron.m_outputWeights[m_myIndex].deltaweight;

        double newDeltaWeight=
                    //Individual input weights, magnified by the gradient and the training rate
                eta
                *neuron.getOutputVal()
                *m_gradient
                // addition to momentum= a fraction of the previous delta weight
                +alpha
                *oldDeltaWieght;

        neuron.m_outputWeights[m_myIndex].deltaweight=newDeltaWeight;
        neuron.m_outputWeights[m_myIndex].weight+=newDeltaWeight;
        
    }
}

double Neuron::sumDOW(const Layer &nextLayer) const
{
    double sum=0.0;
    // Sum our contributions of the errors at the nodes we feed
    for(unsigned n=0; n<nextLayer.size()-1;++n){
        sum+=m_outputWeights[n].weight*nextLayer[n].m_gradient;
    }
    return sum;
}

void Neuron::calculateHiddenGradients(const Layer &nextLayer)
{
    double dow=sumDOW(nextLayer);
    m_gradient=dow*Neuron::activationFunctionDerivative(m_outputVal);
}

void Neuron::calculateOutputGradients(double targetVal)
{
    double delta= targetVal-m_outputVal;
    m_gradient= delta* Neuron::activationFunctionDerivative(m_outputVal);
}

double Neuron::activationFunction(double x)
{
    //tanh-output range[-1.0 1.0]
    return tanh(x);
}

double Neuron::activationFunctionDerivative(double x)
{
    //tanh derivative
    return 1.0-x*x;
}


void Neuron::feedForward(const Layer &prevLayer)
{
    double sum=0.0;

    //Sum the previous layers outputs
    //include the bias node from the previous layer

    for(unsigned n=0; n<prevLayer.size(); ++n){
        sum+=prevLayer[n].getOutputVal()*
            prevLayer[n].m_outputWeights[m_myIndex].weight;
    }

    m_outputVal=Neuron::activationFunction(sum);
}

Neuron::Neuron(unsigned numOutputs, unsigned myIndex)
{
    for(unsigned c=0;c<numOutputs;++c){
        m_outputWeights.push_back(Connection());
        m_outputWeights.back().weight=randomWeight();
    }
     m_myIndex=myIndex;
}


//**************** class Net ******************
class Net
{
    public:
        Net(const vector<unsigned> &topology);
        void feedForward(const vector<double> &inputVal);
        void backProp(const vector<double> &targetVals);
        void getResults(vector<double> &resultVal) const;
        double getRecentAverageError(void) const {return m_recentAverageError;}
    private:
        vector<Layer> m_layers;
        double m_error;
        double m_recentAverageError;
        static double m_recentAverageSmoothingFactor;
};

// Number of training samples to average over
double Net::m_recentAverageSmoothingFactor=100.0; 

void Net::getResults(vector<double> &resultVal) const
{
    resultVal.clear();
    
    for(unsigned n=0;n<m_layers.back().size()-1;++n){
        resultVal.push_back(m_layers.back()[n].getOutputVal());
    }
}


void Net::feedForward(const vector<double> &inputVal)
{
    // assert that the num of inputVals euqal to neuronnum expect bias
    assert(inputVal.size()==m_layers[0].size()-1);

    //assign the input values into the input neurons
    for(unsigned i=0;i<= inputVal.size(); ++i){
        m_layers[0][i].setOutputVal(inputVal[i]);
    }

    //Forward propogation
    for(unsigned layerNum=1; layerNum<m_layers.size(); ++layerNum){
        Layer &prevLayer =m_layers[layerNum-1];
        for(unsigned n=0; n<m_layers[layerNum].size()-1; ++n){
            m_layers[layerNum][n].feedForward(prevLayer);
        }
    }
}

void Net::backProp(const std::vector<double> &targetVals)
{
	// Calculate overal net error (RMS of output neuron errors)

	Layer &outputLayer = m_layers.back();
	m_error = 0.0;

	for(unsigned n = 0; n < outputLayer.size() - 1; ++n)
	{
		double delta = targetVals[n] - outputLayer[n].getOutputVal();
		m_error += delta *delta;
	}
	m_error /= outputLayer.size() - 1; // get average error squared
	m_error = sqrt(m_error); // RMS

	// Implement a recent average measurement:

	m_recentAverageError = 
			(m_recentAverageError * m_recentAverageSmoothingFactor + m_error)
			/ (m_recentAverageSmoothingFactor + 1.0);
	// Calculate output layer gradients

	for(unsigned n = 0; n < outputLayer.size() - 1; ++n)
	{
		outputLayer[n].calculateOutputGradients(targetVals[n]);
	}
	// Calculate gradients on hidden layers

	for(unsigned layerNum = m_layers.size() - 2; layerNum > 0; --layerNum)
	{
		Layer &hiddenLayer = m_layers[layerNum];
		Layer &nextLayer = m_layers[layerNum + 1];

		for(unsigned n = 0; n < hiddenLayer.size(); ++n)
		{
			hiddenLayer[n].calculateHiddenGradients(nextLayer);
		}
	}

	// For all layers from outputs to first hidden layer,
	// update connection weights

	for(unsigned layerNum = m_layers.size() - 1; layerNum > 0; --layerNum)
	{
		Layer &layer = m_layers[layerNum];
		Layer &prevLayer = m_layers[layerNum - 1];

		for(unsigned n = 0; n < layer.size() - 1; ++n)
		{
			layer[n].updateInputWeights(prevLayer);
		}
	}
}


Net::Net(const vector<unsigned> &topology)
{
    unsigned numLayers=topology.size();
    for(unsigned layerNum=0; layerNum<numLayers; ++layerNum)
    {
        m_layers.push_back(Layer());
        unsigned numOutputs= layerNum==topology.size()-1 ? 0:topology[layerNum+1];

        //add a bias neuron to the layer
        for(unsigned neuronNum=0;neuronNum<=topology[layerNum];neuronNum++)
        {
            m_layers.back().push_back(Neuron(numOutputs,neuronNum));
            cout<<"Made a Neuron!"<<endl;
        }

        //Force the bias node's output value to 1.0
        m_layers.back().back().setOutputVal(1.0);
    }
}

void showVectorVals(string label, vector<double> &paramVector)
{
    cout<<label<<" ";
    for(unsigned i=0;i<paramVector.size();i++){
        cout<<paramVector[i]<<" ";
    }
    cout<<endl;
}


int main()
{
    TrainingData trainData("trainingData.txt");
    //ie {3,2,1}
    vector<unsigned> topology;
   // topology.push_back(3);
   // topology.push_back(2);
   // topology.push_back(1);

   trainData.getTopology(topology);
    Net mynet(topology);

    vector<double> inputVal,targetVals,resultVal;
    int trainingPass=0;
    while(!trainData.isEof())
    {
        ++trainingPass;
        cout<<endl<<"Pass"<<trainingPass;

        //get new input data and feed it forward:
        if(trainData.getNextInputs(inputVal)!=topology[0]){
            //cout<<"Input shape mismatch!"<<endl;
            break;
        }
        showVectorVals(": Inputs :", inputVal);
        mynet.feedForward(inputVal);

        //Collect the net's actual results:
        mynet.getResults(resultVal);
        showVectorVals(": Outputs :", resultVal);

        //train the net of what the outputs should have been:
        trainData.getTargetOutputs(targetVals);
        showVectorVals(": targets :", targetVals);
        assert(targetVals.size()==topology.back());

        mynet.backProp(targetVals);

        //Report on how the training process is working
        cout<<"Net recent average error: "
            <<mynet.getRecentAverageError()<<endl;
    }
    cout<<endl<<"Done!"<<endl;   
}