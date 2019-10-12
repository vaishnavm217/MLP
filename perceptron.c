#include<stdio.h>
#include<math.h>
#include<stdlib.h>
#include<time.h>
/*
MULTILAYER PERCEPTRON
FORMULA TO BE USED IN ERROR BETWEEN HIDDEN NODE AND OUTPUT LAYER:
E`(wrt node k in hidden layer)=-(desired_output_k-current_output_k)*f`(sum_of_weights_with_hiddennode_values)
FORMULA TO BE USED IN ERROR BETWEEN HIDDEN NODE AND OUTPUT LAYER:
E`(wrt node j in input layer)=sum(-error_in_hidden_layer*weight_of_kj*f`(sum_of_weights_with_hidden_node_values)*input_node_i)
3D weight matrix
1st Dimension - Layer
2nd & 3rd Dimension - locating the weight
*/
FILE *train,*test;
float *maxtest,*maxtrain,*mintest,*mintrain;
float **prevhiddenweights;
float **previnputweights;
float ***weights;
float **data;
float *hiddendata;
float **testdata;
int Number_neurons[3]={13,-1,3};
float *errorhidden;
float *errorinput;
float output[3];
float acc[10];
float ep[10];
float no[10];
int count;
float sigmoidfun(float x)
{
	return (float)((1.0)/(1+exp(-1*x)));
}
float derivativefun(float x)
{
	return (float)(sigmoidfun(x)*(1-sigmoidfun(x)));
}
float randomweights()
{
	return ((float)(rand()) / (float)(RAND_MAX)) * (0.5+0.5) -0.5;
}
void loaddata()
{
	int i=0,j=0;
	mintest=new float[13];
	mintrain=new float[13];
	maxtest=new float[13];
	maxtrain=new float[13];
	data=new float*[119];
	testdata=new float*[61];
	train=fopen("Wine data/train.csv","r");
	for(;i<118;i++)
		testdata[i]=new float[15];
	for(i=0;i<118;i++)
		data[i]=new float[15];
	i=0;
	while(fscanf(train,"%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f\n",
		&data[i][1],&data[i][2],&data[i][3],&data[i][4],
		&data[i][5],&data[i][6],&data[i][7],&data[i][8],
		&data[i][9],&data[i][10],&data[i][11],&data[i][12],
		&data[i][13],&data[i][14])!=EOF)
	{
		data[i][0]=1.0;
		i++;
	}
	fclose(train);
	test=fopen("Wine data/test.csv","r");
	i=0;
	while(fscanf(test,"%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f\n",
	&testdata[i][1],&testdata[i][2],&testdata[i][3],&testdata[i][4],
	&testdata[i][5],&testdata[i][6],&testdata[i][7],&testdata[i][8],
	&testdata[i][9],&testdata[i][10],&testdata[i][11],&testdata[i][12],
	&testdata[i][13],&testdata[i][14])!=EOF)
	{
		testdata[i][0]=1.0;
		i++;
	}
	fclose(test);
	for(j=1;j<14;j++)
  {
		maxtrain[j-1]=mintrain[j-1]=data[0][j];
		maxtest[j-1]=mintest[j-1]=testdata[0][j];

		for(i=0;i<118;i++)
		{
			if(maxtrain[j-1]<data[i][j])
				maxtrain[j-1]=data[i][j];
			if(mintrain[j-1]>data[i][j])
				mintrain[j-1]=data[i][j];
			if(i<60)
			{
				if(maxtest[j-1]<testdata[i][j])
					maxtest[j-1]=testdata[i][j];
				if(mintest[j-1]>testdata[i][j])
					mintest[j-1]=testdata[i][j];
			}
		}
	}
  for(i=0;i<118;i++)
  {
		for(j=0;j<13;j++)
		{
			data[i][j+1]=(data[i][j+1]-mintrain[j])/(maxtrain[j]-mintrain[j]);
			if(i<60)
				testdata[i][j+1]=(testdata[i][j+1]-mintest[j])/(maxtest[j]-mintest[j]);
		}
  }
}
void init(int n)
{
	Number_neurons[1]=n;
	weights=new float**[2];
	int i=0,j=0,k=0;
	for(;i<2;i++)
	{
		if(!i)
		{
			weights[i]=new float*[Number_neurons[0]+1];
			for(j=0;j<Number_neurons[0]+1;j++)
			{
				weights[i][j]=new float[Number_neurons[1]];
				for(k=0;k<Number_neurons[1];k++)
				{
					weights[i][j][k]=randomweights();
				}
			}
		}
		else
		{
			weights[i]=new float*[Number_neurons[1]+1];
			for(j=0;j<Number_neurons[1]+1;j++)
			{
				weights[i][j]=new float[Number_neurons[2]];
				for(k=0;k<Number_neurons[2];k++)
				{
					weights[i][j][k]=randomweights();
				}
			}
		}
	}
}
void run_model()
{
	int i=0,j=0,k=0,datavar=0,step=0;
	float sum,error=0,cor=0.0,cl=-1;
	float *netk=new float[3];
	float *exp=new float[3];
	float *netj=new float[Number_neurons[1]];
	float *dk=new float[3];
	float *dj=new float[Number_neurons[1]];
	hiddendata=new float[Number_neurons[1]+1];
	/* Calculation of value in hidden layer */

	for(i=0;i<Number_neurons[1];i++)
	{
		sum=0.0;
		for(k=0;k<14;k++)
		{
			sum+=weights[0][k][i]*data[datavar][k];
		}
		netj[i]=sum;
		hiddendata[i+1]=sigmoidfun(sum);
	}
	hiddendata[0]=1;//bias
	/*Calculate the 1st output*/

	for(i=0;i<3;i++)
	{
		sum=0.0;
		for(k=0;k<Number_neurons[1]+1;k++)
		{
			sum+=weights[1][k][i]*hiddendata[k];
		}
		netk[i]=sum;
		output[i]=sigmoidfun(sum);

	}
	/*
		hardcoding first backpropogation
	*/

	switch((int)data[datavar][14])
	{
		case 1:
			exp[0]=1.0;
			exp[1]=0.0;
			exp[2]=0.0;
			break;
		case 2:
			exp[0]=0.0;
			exp[1]=1.0;
			exp[2]=0.0;
			break;
		case 3:
			exp[0]=0.0;
			exp[1]=0.0;
			exp[2]=1.0;
	}
	/* hidden layer and output layer updation*/

	do
	{

			for(i=0;i<3;i++)
			{
			        error+=(exp[i]-output[i])*(exp[i]-output[i]);
				dk[i]=(exp[i]-output[i])*derivativefun(netk[i]);
				//for(j=0;j<Number_neurons[1]+1;j++)
				//{
					//weights[1][j][i]-=0.01*-1*dk[i]*hiddendata[j];
				//}
			}

			/* between hidden layer and input*/

			for(j=0;j<Number_neurons[1];j++)
			{
				sum=0.0;
				for(k=0;k<Number_neurons[2];k++)
				{
					sum+=dk[k]*weights[1][j][k]*derivativefun(netj[j]);
				}
				dj[j]=sum;
			}


			for(i=0;i<Number_neurons[0]+1;i++)
			{
				for(j=0;j<Number_neurons[1];j++)
				{
					weights[0][i][j]+=0.1*dj[j]*data[datavar][i];
				}
			}

			for(i=0;i<Number_neurons[1]+1;i++)
			{
				for(j=0;j<Number_neurons[2];j++)
				{
					weights[1][i][j]+=0.1*dk[j]*hiddendata[i];
				}
			}

			datavar++;
			if(datavar>117)
			{
				step++;
				datavar=datavar%117;
				if(error<0.01)
					break;
				error=0;
			}

			/* After first backpropogation*/
			for(i=0;i<Number_neurons[1];i++)
			{
				sum=0.0;
				for(k=0;k<14;k++)
				{
					sum+=weights[0][k][i]*data[datavar][k];
				}
				netj[i]=sum;
				hiddendata[i+1]=sigmoidfun(sum);
			}

			for(i=0;i<3;i++)
			{
				sum=0.0;
				for(k=0;k<Number_neurons[1]+1;k++)
				{
					sum+=weights[1][k][i]*hiddendata[k];
				}
				netk[i]=sum;
				output[i]=sigmoidfun(sum);
			}

			switch((int)data[datavar][14])
			{
				case 0:
					getchar();
					break;
				case 1:
					exp[0]=1.0;
					exp[1]=0.0;
					exp[2]=0.0;
					break;
				case 2:
					exp[0]=0.0;
					exp[1]=1.0;
					exp[2]=0.0;
					break;
				case 3:
					exp[0]=0.0;
					exp[1]=0.0;
					exp[2]=1.0;
			}

	} while(1);

	printf("Done! epochs:%d\n",step);
	datavar=0;
	while(datavar<118)
	{
		for(i=0;i<Number_neurons[1];i++)
		{
			sum=0.0;
			for(k=0;k<14;k++)
			{
				sum+=weights[0][k][i]*data[datavar][k];
			}
			netj[i]=sum;
			hiddendata[i+1]=sigmoidfun(sum);
		}
		for(i=0;i<3;i++)
		{
			sum=0.0;
			for(k=0;k<Number_neurons[1]+1;k++)
			{
				sum+=weights[1][k][i]*hiddendata[k];
			}
			netk[i]=sum;
			output[i]=sigmoidfun(sum);
			//printf("%f ",output[i]);

		}
		if(output[0]>output[1] && output[0]>output[2])
		{
			cl=1.0;
		}
		if(output[1]>output[0] && output[1]>output[2])
		{
		  cl=2.0;
		}
		if(output[2]>output[0] && output[2]>output[1])
		{
		  cl=3.0;
		}

		if(cl==data[datavar][14])
		  cor++;
		else
		{
		  printf("Entry: %d output: %f actual class: %f\n",datavar+1,cl,data[datavar][14]);
		}
		datavar++;
	}
	printf("%f %% accurate on train data\n",(cor/118.0)*100);
  datavar=0;
	cor=0.0;
	while(datavar<60)
	{
		for(i=0;i<Number_neurons[1];i++)
		{
			sum=0.0;
			for(k=0;k<14;k++)
			{
				sum+=weights[0][k][i]*testdata[datavar][k];
			}
			netj[i]=sum;
			hiddendata[i+1]=sigmoidfun(sum);
		}
		for(i=0;i<3;i++)
		{
			sum=0.0;
			for(k=0;k<Number_neurons[1]+1;k++)
			{
				sum+=weights[1][k][i]*hiddendata[k];
			}
			netk[i]=sum;
			output[i]=sigmoidfun(sum);
		}
		if(output[0]>output[1] && output[0]>output[2])
		{
	    cl=1.0;
		}
		if(output[1]>output[0] && output[1]>output[2])
		{
	    cl=2.0;
		}
		if(output[2]>output[0] && output[2]>output[1])
		{
	    cl=3.0;
		}
		if(cl==testdata[datavar][14])
	  {
			cor++;
		}
		else
		{
	    printf("Entry: %d output: %f actual class: %f\n",datavar+1,cl,testdata[datavar][14]);
		}
		datavar++;
	}
  printf("%f %% accurate on test data\n",(cor/60.0)*100);
  ep[count]=step;
  acc[count]=(cor/60.0)*100;
  no[count]=Number_neurons[1];

}

int main()
{
	srand((unsigned int)time(NULL));
	loaddata();
	int j=10;
	count=0;
  for(;j<21;j++)
	{
    printf("Number of Nodes: %d\n",j);
    init(j);
		run_model();
    count++;
		printf("\n");
	}
	for(j=0;j<count;j++)
  {
		printf("%f,%f,%f\n",ep[j],no[j],acc[j]);
  }
	return 0;
}
