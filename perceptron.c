#include<stdio.h>
#include<math.h>
#include <stdlib.h>
#include <time.h>
float ***weights[];
float **data;
float LR;
int Number_neurons[3]={13,-1,3};
int currtime=0;
srand((unsigned int)time(NULL));
/*struct node
{
	float data[13];
	int dclass;
};*/
/*float sigmoidfun1(float x)
{
	return tanh(x);
}*/
float sigmoidfun2(float x)
{
	return pow(1+exp(-1*x),-1);
}
float derivativefun2(float x)
{
	return (float)sigmoidfun2(x)*(1-sigmoidfun2(x));
}
void randomweights(int index,int layer)
{
	/*
		Ideal initial range=√(6/(num_of_neuron_layer[i]+num_of_neuron_layer[i-1]))
	*/
	int i=0;
	
	for(;i<Number_neurons[layer];i++)
	{
		weight[currtime][layer][index][i]=((float)rand()/(float)(RAND_MAX))*(sqrt(6/(Number_neurons[layer]+Number_neurons[layer-1])));
	}
}
float summation(int index,int layer)
{
	float sum=0.0;
	for(i=0;i<Number_neurons[layer];i++)
		sum+=weight[currtime][layer][index][i]
	return sum;	
}
void update_weight(int layer,int index)
{
	w
}
