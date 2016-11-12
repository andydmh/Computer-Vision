#include <stdio.h>                  /*  Marr-Hildreth.c  (or marrh.c) */
#include <math.h>
#define  PICSIZE 256
#define  MAXMASK 100

         int    pic[PICSIZE][PICSIZE];
         double Xoutpic[PICSIZE][PICSIZE];
         double Youtpic[PICSIZE][PICSIZE];
         int    edgeflag[PICSIZE][PICSIZE];
         double Xmask[MAXMASK][MAXMASK];
         double Ymask[MAXMASK][MAXMASK];
         double Xconv[PICSIZE][PICSIZE];
         double Yconv[PICSIZE][PICSIZE];

        double ival[PICSIZE][PICSIZE];
        double ivalTemp[PICSIZE][PICSIZE];
double finals[PICSIZE][PICSIZE];
        int cand[PICSIZE][PICSIZE];

        struct rec
        {
            char a[2];
            int w,h,graylevel;
        }my_rec;

main(argc,argv)
int argc;
char **argv;
{
        int     i,j,p,q,s,t,mr,centx,centy, numbAbHi, k, px, py, more;
        double  maskval,Xsum,Ysum,sig,percent, maxival,minival,maxval,ZEROTOL,slope, sum,  HI, LO;
        FILE    *fo1, *fo2, *fo3, *fp1, *fopen();
        char    *foobar;
    

        argc--; argv++;
        foobar = *argv;
        fp1=fopen(foobar,"rb");

        fo1=fopen("magnitude.pgm","wb");
        fo2=fopen("peaks.pgm","wb");
        fo3=fopen("Edges.pgm","wb");

    
        argc--; argv++;
        foobar = *argv;
        sig = atoi(foobar) * 1.0;

        argc--; argv++;
        foobar = *argv;
        percent = atoi(foobar) * 1.0;
    printf("percent = %f\n", percent);
    
    numbAbHi = (int)((percent/100.0)*256.0*256.0);
    printf("abHi = %d\n", numbAbHi);
    
        mr = (int)(sig * 3);
    
        centx = (MAXMASK / 2);
        centy = (MAXMASK / 2);

    
    fread(&my_rec, sizeof(struct rec), 1,fp1);
    
        for (i=0;i<256;i++)
        { for (j=0;j<256;j++)
                {
                  pic[i][j]  =  getc (fp1);
                }
        }

    
    
        for (p=-mr;p<=mr;p++)
        {  for (q=-mr;q<=mr;q++)
           {
               
               maskval = p*exp(-1*((p*p+q*q)/(2*sig*sig)));//((2-(((p*p)+(q*q))/(sig*sig)))*(exp(-1*(((p*p)+(q*q))/(2*(sig*sig))))));
               
              (Xmask[p+centy][q+centx]) = maskval;
               
               maskval = q*exp(-1*((p*p+q*q)/(2*sig*sig)));
               
               (Ymask[p+centy][q+centx]) = maskval;
           }
        }

        for (i=mr;i<=PICSIZE-mr;i++)
        { for (j=mr;j<=PICSIZE-mr;j++)
          {
             Xsum = 0;
              Ysum = 0;
             for (p=-mr;p<=mr;p++)
             {
                for (q=-mr;q<=mr;q++)
                {
                   Xsum += pic[i+p][j+q] * Xmask[p+centy][q+centx];
                   Ysum += pic[i+p][j+q] * Ymask[p+centy][q+centx];
                }
             }
             Xoutpic[i][j] = Xsum;
             Youtpic[i][j] = Ysum;
              
             //Xconv[i][j] = Xsum;
              //Yconv[i][j] = Ysum;
          }
        }
    
    maxval = 0;//not sure if needed
    for (i=mr;i<256-mr;i++)
    {
        for (j=mr;j<256-mr;j++)
        {
            ival[i][j]=sqrt((double)((Xoutpic[i][j]*Xoutpic[i][j]) + (Youtpic[i][j]*Youtpic[i][j])));
            
            if (ival[i][j] > maxival)//not sure
                maxival = ival[i][j];// not sure
        }
    }
    
    for (i=0;i<256;i++)
    {
        for (j=0;j<256;j++)
    {
        
            ival[i][j] =  (ival[i][j] / maxival) * 255;        
        
    }
    }
    
    fwrite(&my_rec, sizeof(struct rec), 1,fo1);
    
    for (i=0;i<256;i++)
    { for (j=0;j<256;j++)
    {
        
        fprintf(fo1,"%c",(char)((ival[i][j])));
        
    }
    }
    //ok----------Part 1---------
        
    for(i=mr;i<256-mr;i++)
    {
        
        for(j = mr;j<256-mr;j++)
        {
            //printf("i = %d j = %d\n", i, j);
            if(Xoutpic[i][j] == 0.0)
            {
                Xoutpic[i][j] = 0.00001;
            }
            
            //printf("Xgrad = %f Ygrad = %f\n", Xoutpic[i][j], Youtpic[i][j]);
            slope = Youtpic[i][j]/ Xoutpic[i][j];
            
            if((slope <= 0.4142) && (slope > -0.4142))
            {
                if((ival[i][j] > ival[i][j-1]) && (ival[i][j] > ival[i][j+1]))
                {
                    cand[i][j] = 255;
                }
                else
                    cand[i][j] = 0;
                
            }
            else if((slope <= 2.4142) && (slope > 0.4142))
            {
                if((ival[i][j] > ival[i-1][j-1]) && (ival[i][j] > ival[i+1][j+1]))
                {
                    cand[i][j] = 255;
                }
                else
                    cand[i][j] = 0;
                
            }
            else if((slope <= -0.4142) && (slope > -2.4142))
            {
                if((ival[i][j] > ival[i+1][j-1]) && (ival[i][j] > ival[i-1][j+1]))
                {
                    cand[i][j] = 255;
                }
                else
                    cand[i][j] = 0;
                
            }
            else
            {
                if((ival[i][j] > ival[i-1][j]) && (ival[i][j] > ival[i+1][j]))
                {
                    cand[i][j] = 255;
                }
                else
                    cand[i][j] = 0;
                
            }
        }
    }
    
    fwrite(&my_rec, sizeof(struct rec), 1,fo2);
    
    for (i=0;i<256;i++)
    { for (j=0;j<256;j++)
    {
        
        fprintf(fo2,"%c",(char)((cand[i][j])));
        
    }
    }
    //----------------------------Part 2
    
    //Starting part 4---------------------------------
    
    for(i = 0; i < 256; i++)
    {
        for(j=0; j < 256; j++)
        {
            ivalTemp[i][j] = ival[i][j];
        }
    }
    
    sum = 0;
    for(k = 0; k < numbAbHi; k++)
    {
        maxival = 0;
        for(i = 0; i < 256; i++)
        {
            for(j = 0;j < 256; j++)
            {
                if(ivalTemp[i][j] > maxival)
                {
                    maxival = ivalTemp[i][j];
                    px = i;
                    py = j;
                }
            }
        }
        sum = sum + maxival;
        ivalTemp[px][py] = -1;
    }
    
    HI = (sum -1)/numbAbHi;
    LO = 0.35 * HI;
    
    //Part 4 is done--------------------
    
    //Starting part 3----------------------
    
    for(i = 0; i < 256; i++)
    {
        for(j = 0; j < 256; j++)
        {
            if(cand[i][j] == 255)
            {
                if(ival[i][j] > HI)
                {
                    finals[i][j] = 255;
                    cand[i][j] = 0;
                }
                else if(ival[i][j] < LO)
                {
                    finals[i][j] = 0;
                    cand[i][j] = 0;
                }
            }
            
        }
    }
    
    more = 1;
    
    while(more == 1)
    {
        more = 0;
        for(i = 0; i < 256; i++)
        {
            for(j = 0; j < 256; j++)
            {
                if(cand[i][j] == 255)
                {
                    for(p = -1; p <=1; p++)
                    {
                        for(q = -1; q<=1; q++)
                        {
                            if(finals[i+p][j+q] == 255)
                            {
                                cand[i][j] = 0;
                                finals[i][j] = 255;
                                more = 1;
                            }
                        }
                    }
                }
            }
        }
    }
    
    
    fwrite(&my_rec, sizeof(struct rec), 1,fo3);
    
    for (i=0;i<256;i++)
    { for (j=0;j<256;j++)
    {
        if(finals[i][j] == 255)
            fprintf(fo3,"%c",(char)((0)));
        else
            fprintf(fo3,"%c",(char)((255)));
        
        
    }
    }
    //Done part3 ------------------
    
}