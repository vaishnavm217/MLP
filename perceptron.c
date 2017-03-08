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
float **data;
float *hiddendata;
float **testdata;
int Number_neurons[3]={13,-1,3};
int currtime=0;
float **errorhidden;
float **errorinput;
float output[3];
srand((unsigned int)time(NULL));
void loaddata(char *a,char *b)
{
	train=fopen(a,"r");
	int i=0;
	data=new float*[118];
	for(;i<118;i++)
		data[i]=new float[15];
	i=0;
	while(fscanf(train,"%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f",data[i][1],data[i][2],data[i][4],data[i][5],data[i][6],data[i][7],data[i][8],data[i][9],data[i][10],data[i][11],data[i][13],data[i][14]))
	{
		data[i][0]=1;
		i++;
	}
	test=fopen(b,"r");
	i=0;
	testdata=new float*[60];
	for(;i<118;i++)
		testdata[i]=new float[15];
	while(fscanf(test,"%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f",testdata[i][1],testdata[i][2],testdata[i][4],testdata[i][5],testdata[i][6],testdata[i][7],testdata[i][8],testdata[i][9],testdata[i][10],testdata[i][11],testdata[i][13],testdata[i][14]))
	{
		testdata[i][0]=1;
		i++;
	}
}
void init(int n)
{
	Number_neurons[1]=n;
	weights=new float**[2];
	int i=0,j=0,k=0;
	for(;i<2;i++)
	{
		if(i)
		{
			weights[i]=new float*[Number_neurons[1]+1];
			for(j=0;j<Number_neuron[1]+1;j++)
			{
				weights[i][j]=new float[Number_neurons[0]+1];
				for(k=0;k<Number_neurons[0]+1;k++)
				{
					weights[i][j][k]=randomweights();
				}
			}
		}
		else
		{
			weights[i]=new float*[Number_neuron[2]+1];
			for(j=0;j<Number_neuron[2]+1;j++)
			{
				weights[i][j]=new float[Number_neurons[1]+1];
				for(k=0;k<Number_neurons[1]+1;k++)
				{
					weights[i][j][k]=randomweights();
				}
			}
		}
		
	}
	
}
float sigmoidfun(float x)
{
	return pow(1+exp(-1*x),-1);
}
float derivativefun(float x)
{
	return (float)sigmoidfun2(x)*(1-sigmoidfun2(x));
}
float randomweights()
{
	return ((float)rand()/(float)(RAND_MAX))*(sqrt(6/(Number_neurons[layer]+Number_neurons[layer-1])));
}
void run_model()
{
	int i=0,j=0,k=0,datavar=0;float sum;
	hiddendata=new int[Number_neuron[1]+1];
	/* Calculation of value in hidden layer */
	for(i=0;i<Number_neuron[1]+1;i++)
	{
		sum=0.0;
		for(k=0;k<14;k++)
		{
			sum+=weight[0][i][k]*data[datavar][k];
		}
		hiddendata[i]=sigmoidfun(sum);
	}
	/*Calculate the 1st output*/
	printf("Values for input %d 1st epoch\n",datavar+1);
	for(i=0;i<3;i++)
	{
		sum=0.0;
		for(k=0;k<Number_neurons[1]+1;k++)
		{
			sum+=weight[0][i][k]*hiddendata[k];
		}
		output[i]=sigmoidfun(sum);
		printf("%f ",output[i]);
		
	}
	printf("\n");
	
}
void backpropogation()
{
	
}
