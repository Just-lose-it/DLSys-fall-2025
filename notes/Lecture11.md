## L11. Hardware Accel.

### General Accel. Techniques

1. Vectorization

e.g. vec add
```C
for(int i=0;i<64;i++)
{
    //pseudo code
    //require memory aligned in 128 bits
    float4 a=load_float4(A+i*4);
    float4 b=load_float4(B+i*4);
    float4 c=add_float4(a,b);
    store_float4(C+i*4,c);
}
```

2. Data layout and strides

- Row major: `A[i,j]=Adata[i*A.shape[1]+j]`   
- Col major: `A[i,j]=Adata[j*A.shape[0]+i]     (old BLAS libraries)`
- Strides format: `A[i,j]=Adata[i*A.stride[0]+j*A.stride[1]]   (more general)`
    - advantage:zero copy on certain transformation:
      - slicing:change the begin offset and shape
      - transpose:swap strides
      - broadcast:insert a stride=0
    - disadvantage:non-continuous memory access
      - harder to vectorize
      - some libraries may require compact the array first

3.parallelization

```C
#pragma omp parallel for
for(int i=0;i<64;i++)
{
    //pseudo code
    //require memory aligned in 128 bits
    float4 a=load_float4(A+i*4);
    float4 b=load_float4(B+i*4);
    float4 c=add_float4(a,b);
    store_float4(C+i*4,c);
}
```


### Case study:Matrix Mult

e.g. C = dot(A,B^T)

Vanilla version:

```C
for(int i=0;i<n;i++)
    for(int j=0;j<n;j++)
    {
        C[i][j]=0;
        for(int k=0;k<n;k++)
        {
            C[i][j]+=A[i][k]*B[j][k];
        }
    }
```

Time Complexity:O(n^3)

CPU arch: L1 cache:0.5ns  L2: 7ns   DRAM:200ns

Architecture aware analysis:

```C
dram float A[n][n],B[n][n],C[n][n]
for(int i=0;i<n;i++)
    for(int j=0;j<n;j++)
    {
        register float c=0;
        for(int k=0;k<n;k++)
        {
            register float a=A[i][k];
            register float b=B[j][k];
            c+=a*b;
        }
        C[i][j]=c;
    }
```

Cost: 2n^3 dram load,3 registers


#### idea1:register tiling



```C
dram float A[n/v1][n/v3][v1][v3],B[n/v2][n/v3][v2][v3],C[n/v1][n/v2][v1][v2];
for(int i=0;i<n/v1;i++)
    for(int j=0;j<n/v2;j++)
    {
        register float c[v1][v2]=0;
        for(int k=0;k<n/v3;k++)
        {
            register float a[v1][v3]=A[i][k];
            register float b[v2][v3]=B[j][k];
            c+=dot(a,b.T);
        }
        C[i][j]=c;
    }
```

Cost:`n^3/v1+n^3/v2` loads,v1v2+v2v3+v1v3 registers 
Note:v3=1 won't affect anything

#### idea2:  L1 cache tiling
```C
dram float A[n/b1][n][b1],B[n/b2][n][b2],C[n/b1][n/b2][b1][b2];
for(int i=0;i<n/b1;i++)
{
    l1cache float a[b1][n]=A[i];
    for(int j=0;j<n/b2;j++)
    {
        l1cache float b[b2][n]=B[i];
        C[i][j]=dot(a,bT)//register optimized in idea 1
    }
}
    
```

Cost: `n^2+n^3/b1` loads from dram to L1 cache,`n^3/v1+n^3/v2`loads from L1 cache to registers

#### Tiling reuse patterns:

`C[i][j]=sum(a[i][k]*b[j][k] ,axis=k)`

a independent of j,so tile j by v times enables reuse of a by v times(load a first ,then use corresponding b for v times)

Note: possible reuse in convolution