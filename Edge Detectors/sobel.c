#include <stdio.h>                          /* Sobel.c */
#include <math.h>							/*Expected parameters: ./sobel inputIMage outputName highTreshold lowTreshold*/
#include <string.h>
#include <stdlib.h>

        int pic[256][256];
        int outpicx[256][256];
        int outpicy[256][256];
        int maskx[3][3] = {{-1,0,1},{-2,0,2},{-1,0,1}};
        int masky[3][3] = {{1,2,1},{0,0,0},{-1,-2,-1}};
        double ival[256][256];//,maxival; maxval is not used because instead of making the pixel values proportional to the original ones, the values of the pixels will be either 0 or 255 depending on wether its value is bellow or above the treshold (to make edges seem more sharp as mentioned in the notes). Also there is no need to worry about a pixel value greather than 255. Another aproach could be to apply the treshold after appling the formula (ival[i][j] / maxival) * 255 (however the it seems more intuitive to apply the previous approach) 

struct rec
	{
		char a[2];
		int w,h,graylevel;
	}my_rec;

main(argc,argv)
int argc;
char **argv;
{
        int i,j,p,q,mr,sum1,sum2;
        double hi_treshold, low_treshold, cur_hi, cur_low;
        FILE *fo1, *fo2, *fp1, *fopen();
        char *foobar, *highT, * lowT;
		
        argc--; argv++;
        foobar = *argv;
        fp1=fopen(foobar,"rb");

	argc--; argv++;
	foobar = *argv;
	highT = malloc(sizeof(foobar) + 2);
	highT[0] = 'H'; highT[1] = '\0'; strcat(highT,foobar);//name of image with high treshold has an H in front
	lowT = malloc(sizeof(foobar) + 2);
	lowT[0] = 'L'; lowT[1] = '\0'; strcat(lowT,foobar);//name of image with low treshold has an L in front 
	fo1=fopen(highT,"wb");//to save the image with high treshold
	fo2=fopen(lowT,"wb");//to save the image with low treshold

        
        argc--; argv++;
	foobar = *argv;
	hi_treshold = 4*atof(foobar);//(*4) takes into account the 1/4 eliminated from sobel's equation. It alows the person giving the treshold to think in terms of pixel values and provide a treshold in the range [0,255] instead of [0,1000]. But it will be the same if the person providing the treshold is aware of the missing 1/4 in that case the (*4) can be eliminated. 


		argc--; argv++;
	foobar = *argv;
	low_treshold = 4*atof(foobar);//(*4) takes into account the 1/4 eliminated from sobel's equation
	
	fread(&my_rec, sizeof(struct rec), 1,fp1);

        for (i=0;i<256;i++)
        { for (j=0;j<256;j++)
                {
                  pic[i][j]  =  getc (fp1);
                  pic[i][j]  &= 0377;
                }
        }

        mr = 1;
        for (i=mr;i<256-mr;i++)
        { for (j=mr;j<256-mr;j++)
          {
             sum1 = 0;
             sum2 = 0;
             for (p=-mr;p<=mr;p++)
             {
                for (q=-mr;q<=mr;q++)
                {
                   sum1 += pic[i+p][j+q] * maskx[p+mr][q+mr];
                   sum2 += pic[i+p][j+q] * masky[p+mr][q+mr];
                }
             }
             outpicx[i][j] = sum1;
             outpicy[i][j] = sum2;
          }
        }
		
        //maxival = 0;
        for (i=mr;i<256-mr;i++)
        { for (j=mr;j<256-mr;j++)
          {
             ival[i][j]=sqrt((double)((outpicx[i][j]*outpicx[i][j]) +
                                      (outpicy[i][j]*outpicy[i][j])));
             //if (ival[i][j] > maxival)
               // maxival = ival[i][j];

           }
        }



		//headers	
		//fprintf(fo1, "P5\n");
		//fprintf(fo1, "%d %d\n", 256, 256);
		//fprintf(fo1, "255\n");
		fwrite(&my_rec, sizeof(struct rec), 1,fo1);

		//fprintf(fo2, "P5\n");
		//fprintf(fo2, "%d %d\n", 256, 256);
		//fprintf(fo2, "255\n");
		fwrite(&my_rec, sizeof(struct rec), 1,fo2);

        for (i=0;i<256;i++)
          { for (j=0;j<256;j++)
            {
			
			 if(ival[i][j] > hi_treshold)
			 	cur_hi = 255;//(ival[i][j] / maxival) * 255;
			 else
				cur_hi = 0;

			 if(ival[i][j] > low_treshold)
			 	cur_low = 255;//(ival[i][j] / maxival) * 255;
			 else
				cur_low = 0;

             //ival[i][j] = (ival[i][j] / maxival) * 255;            
             fprintf(fo1,"%c",(char)((int)(cur_hi)));
			 fprintf(fo2,"%c",(char)((int)(cur_low)));
             
            }
          }

	free(highT);
	free(lowT);
}
