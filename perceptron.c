#include<stdio.h>
#include<math.h>
#include <stdlib.h>
#include <time.h>
/*
MULTILAYER PERCEPTRON!

FORMULA TO BE USED IN ERROR BETWEEN HIDDEN NODE AND OUTPUT LAYER:
E`(wrt node k in hidden layer)=-(desired_output_k-current_output_k)*f`(sum_of_weights_with_hiddennode_values)

FORMULA TO BE USED IN ERROR BETWEEN HIDDEN NODE AND OUTPUT LAYER:
E`(wrt node j in input layer)=sum(-error_in_hidden_layer*weight_of_kj*f`(sum_of_weights_with_hidden_node_values)*input_node_i)

3D weight matrix
1st Dimension - Layer
2nd & 3rd Dimension - locating the weight
*/
FILE *train,*test;
float **prevhiddenweights;
float **previnputweights;
float ***weights;
float *data;
float *hiddendata;
float LR;
int Number_neurons[3]={13,-1,3};
int currtime=0;
float **errorhidden;
float **errorinput;
srand((unsigned int)time(NULL));
void init(int n)
{
		Number_neurons[1]=n;
		weights=new float**[2];
		int i=0,j=0,k=0;
		for(;i<2;i++)
		{
			if(i)
			{
					weights[i]=new float*[Number_neurons[1]];
					for(j=0;j<Number_neuron[1];j++)
					{
							weights[i][j]=new float[Number_neurons[0]];
							for(k=0;k<Number_neurons[0];k++)
							{
								weights[i][j][k]=randomweights();
							}
					}
			}
			else
			{
				weights[i]=new float*[Number_neuron[2]];
				for(j=0;j<Number_neuron[2];j++)
				{
						weights[i][j]=new float[Number_neurons[1]];
						for(k=0;k<Number_neurons[1];k++)
						{
							weights[i][j][k]=randomweights();
						}
				}
			}
			
		}
}
float sigmoidfun2(float x)
{
	return pow(1+exp(-1*x),-1);
}
float derivativefun2(float x)
{
	return (float)sigmoidfun2(x)*(1-sigmoidfun2(x));
}
float randomweights()
{
	return ((float)rand()/(float)(RAND_MAX))*(sqrt(6/(Number_neurons[layer]+Number_neurons[layer-1])));
}
void backpropogation()
{
	
}