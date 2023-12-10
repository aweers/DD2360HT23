#include "Particles.h"
#include "Alloc.h"
#include <cuda.h>
#include <cuda_runtime.h>

#define TPB 128

/** allocate particle arrays */
void particle_allocate(struct parameters* param, struct particles* part, int is)
{
    
    // set species ID
    part->species_ID = is;
    // number of particles
    part->nop = param->np[is];
    // maximum number of particles
    part->npmax = param->npMax[is];
    
    // choose a different number of mover iterations for ions and electrons
    if (param->qom[is] < 0){  //electrons
        part->NiterMover = param->NiterMover;
        part->n_sub_cycles = param->n_sub_cycles;
    } else {                  // ions: only one iteration
        part->NiterMover = 1;
        part->n_sub_cycles = 1;
    }
    
    // particles per cell
    part->npcelx = param->npcelx[is];
    part->npcely = param->npcely[is];
    part->npcelz = param->npcelz[is];
    part->npcel = part->npcelx*part->npcely*part->npcelz;
    
    // cast it to required precision
    part->qom = (FPpart) param->qom[is];
    
    long npmax = part->npmax;
    
    // initialize drift and thermal velocities
    // drift
    part->u0 = (FPpart) param->u0[is];
    part->v0 = (FPpart) param->v0[is];
    part->w0 = (FPpart) param->w0[is];
    // thermal
    part->uth = (FPpart) param->uth[is];
    part->vth = (FPpart) param->vth[is];
    part->wth = (FPpart) param->wth[is];
    
    
    //////////////////////////////
    /// ALLOCATION PARTICLE ARRAYS
    //////////////////////////////
    part->x = new FPpart[npmax];
    part->y = new FPpart[npmax];
    part->z = new FPpart[npmax];
    // allocate velocity
    part->u = new FPpart[npmax];
    part->v = new FPpart[npmax];
    part->w = new FPpart[npmax];
    // allocate charge = q * statistical weight
    part->q = new FPinterp[npmax];
    
}
/** deallocate */
void particle_deallocate(struct particles* part)
{
    // deallocate particle variables
    delete[] part->x;
    delete[] part->y;
    delete[] part->z;
    delete[] part->u;
    delete[] part->v;
    delete[] part->w;
    delete[] part->q;
}

/** particle mover */
int mover_PC(struct particles* part, struct EMfield* field, struct grid* grd, struct parameters* param)
{
    // print species and subcycling
    std::cout << "***  MOVER with SUBCYCLYING "<< param->n_sub_cycles << " - species " << part->species_ID << " ***" << std::endl;
 
    // auxiliary variables
    FPpart dt_sub_cycling = (FPpart) param->dt/((double) part->n_sub_cycles);
    FPpart dto2 = .5*dt_sub_cycling, qomdt2 = part->qom*dto2/param->c;
    FPpart omdtsq, denom, ut, vt, wt, udotb;
    
    // local (to the particle) electric and magnetic field
    FPfield Exl=0.0, Eyl=0.0, Ezl=0.0, Bxl=0.0, Byl=0.0, Bzl=0.0;
    
    // interpolation densities
    int ix,iy,iz;
    FPfield weight[2][2][2];
    FPfield xi[2], eta[2], zeta[2];
    
    // intermediate particle position and velocity
    FPpart xptilde, yptilde, zptilde, uptilde, vptilde, wptilde;
    
    printf("before kernel:\n");
    for (int i = 0; i < 10; i++)
        printf("part %d at %.10f, %.10f, %.10f\n", i + 159000, part->x[159000 + i], part->y[159000 + i], part->z[159000 + i]);

    // start subcycling
    for (int i_sub=0; i_sub <  part->n_sub_cycles; i_sub++){
        // move each particle with new fields
        for (int i=0; i <  part->nop; i++){
            xptilde = part->x[i];
            yptilde = part->y[i];
            zptilde = part->z[i];
            // calculate the average velocity iteratively
            for(int innter=0; innter < part->NiterMover; innter++){
                // interpolation G-->P
                ix = 2 +  int((part->x[i] - grd->xStart)*grd->invdx);
                iy = 2 +  int((part->y[i] - grd->yStart)*grd->invdy);
                iz = 2 +  int((part->z[i] - grd->zStart)*grd->invdz);
                
                // calculate weights
                xi[0]   = part->x[i] - grd->XN[ix - 1][iy][iz];
                eta[0]  = part->y[i] - grd->YN[ix][iy - 1][iz];
                zeta[0] = part->z[i] - grd->ZN[ix][iy][iz - 1];
                xi[1]   = grd->XN[ix][iy][iz] - part->x[i];
                eta[1]  = grd->YN[ix][iy][iz] - part->y[i];
                zeta[1] = grd->ZN[ix][iy][iz] - part->z[i];

                for (int ii = 0; ii < 2; ii++)
                    for (int jj = 0; jj < 2; jj++)
                        for (int kk = 0; kk < 2; kk++)
                            weight[ii][jj][kk] = xi[ii] * eta[jj] * zeta[kk] * grd->invVOL;
                
                // set to zero local electric and magnetic field
                Exl=0.0, Eyl = 0.0, Ezl = 0.0, Bxl = 0.0, Byl = 0.0, Bzl = 0.0;
                
                for (int ii=0; ii < 2; ii++)
                    for (int jj=0; jj < 2; jj++)
                        for(int kk=0; kk < 2; kk++){
                            Exl += weight[ii][jj][kk]*field->Ex[ix- ii][iy -jj][iz- kk ];
                            Eyl += weight[ii][jj][kk]*field->Ey[ix- ii][iy -jj][iz- kk ];
                            Ezl += weight[ii][jj][kk]*field->Ez[ix- ii][iy -jj][iz -kk ];
                            Bxl += weight[ii][jj][kk]*field->Bxn[ix- ii][iy -jj][iz -kk ];
                            Byl += weight[ii][jj][kk]*field->Byn[ix- ii][iy -jj][iz -kk ];
                            Bzl += weight[ii][jj][kk]*field->Bzn[ix- ii][iy -jj][iz -kk ];
                        }
                
                // end interpolation
                omdtsq = qomdt2*qomdt2*(Bxl*Bxl+Byl*Byl+Bzl*Bzl);
                denom = 1.0/(1.0 + omdtsq);
                // solve the position equation
                ut= part->u[i] + qomdt2*Exl;
                vt= part->v[i] + qomdt2*Eyl;
                wt= part->w[i] + qomdt2*Ezl;
                udotb = ut*Bxl + vt*Byl + wt*Bzl;
                // solve the velocity equation
                uptilde = (ut+qomdt2*(vt*Bzl -wt*Byl + qomdt2*udotb*Bxl))*denom;
                vptilde = (vt+qomdt2*(wt*Bxl -ut*Bzl + qomdt2*udotb*Byl))*denom;
                wptilde = (wt+qomdt2*(ut*Byl -vt*Bxl + qomdt2*udotb*Bzl))*denom;
                // update position
                part->x[i] = xptilde + uptilde*dto2;
                part->y[i] = yptilde + vptilde*dto2;
                part->z[i] = zptilde + wptilde*dto2;
                
                
            } // end of iteration
            // update the final position and velocity
            part->u[i]= 2.0*uptilde - part->u[i];
            part->v[i]= 2.0*vptilde - part->v[i];
            part->w[i]= 2.0*wptilde - part->w[i];
            part->x[i] = xptilde + uptilde*dt_sub_cycling;
            part->y[i] = yptilde + vptilde*dt_sub_cycling;
            part->z[i] = zptilde + wptilde*dt_sub_cycling;
            
            
            //////////
            //////////
            ////////// BC
                                        
            // X-DIRECTION: BC particles
            if (part->x[i] > grd->Lx){
                if (param->PERIODICX==true){ // PERIODIC
                    part->x[i] = part->x[i] - grd->Lx;
                } else { // REFLECTING BC
                    part->u[i] = -part->u[i];
                    part->x[i] = 2*grd->Lx - part->x[i];
                }
            }
                                                                        
            if (part->x[i] < 0){
                if (param->PERIODICX==true){ // PERIODIC
                   part->x[i] = part->x[i] + grd->Lx;
                } else { // REFLECTING BC
                    part->u[i] = -part->u[i];
                    part->x[i] = -part->x[i];
                }
            }
                
            
            // Y-DIRECTION: BC particles
            if (part->y[i] > grd->Ly){
                if (param->PERIODICY==true){ // PERIODIC
                    part->y[i] = part->y[i] - grd->Ly;
                } else { // REFLECTING BC
                    part->v[i] = -part->v[i];
                    part->y[i] = 2*grd->Ly - part->y[i];
                }
            }
                                                                        
            if (part->y[i] < 0){
                if (param->PERIODICY==true){ // PERIODIC
                    part->y[i] = part->y[i] + grd->Ly;
                } else { // REFLECTING BC
                    part->v[i] = -part->v[i];
                    part->y[i] = -part->y[i];
                }
            }
                                                                        
            // Z-DIRECTION: BC particles
            if (part->z[i] > grd->Lz){
                if (param->PERIODICZ==true){ // PERIODIC
                    part->z[i] = part->z[i] - grd->Lz;
                } else { // REFLECTING BC
                    part->w[i] = -part->w[i];
                    part->z[i] = 2*grd->Lz - part->z[i];
                }
            }
                                                                        
            if (part->z[i] < 0){
                if (param->PERIODICZ==true){ // PERIODIC
                    part->z[i] = part->z[i] + grd->Lz;
                } else { // REFLECTING BC
                    part->w[i] = -part->w[i];
                    part->z[i] = -part->z[i];
                }
            }
                                                                        
            
            
        }  // end of subcycling
    } // end of one particle
    printf("\nafter kernel:\n");
    for (int i = 0; i < 10; i++)
        printf("part %d at %.10f, %.10f, %.10f\n", i + 159000, part->x[159000 + i], part->y[159000 + i], part->z[159000 + i]);                                                                    
    return(0); // exit succcesfully
} // end of the mover

__global__ void mover_kernel(struct parameters* param, struct EMfield* field,
    struct particles* part, struct grid* grd,
    FPpart qomdt2, FPpart dto2, FPpart dt_sub_cycling) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= part->nop)
        return;

    FPpart omdtsq, denom, ut, vt, wt, udotb;
    // local (to the particle) electric and magnetic field
    FPfield Exl=0.0, Eyl=0.0, Ezl=0.0, Bxl=0.0, Byl=0.0, Bzl=0.0;

    // interpolation densities
    int ix,iy,iz;
    FPfield weight[2][2][2];
    FPfield xi[2], eta[2], zeta[2];
    FPpart xptilde, yptilde, zptilde, uptilde, vptilde, wptilde;

    xptilde = part->x[i];
    yptilde = part->y[i];
    zptilde = part->z[i];

    // calculate the average velocity iteratively
    for(int innter=0; innter < part->NiterMover; innter++){
        // interpolation G-->P
        ix = 2 +  int((part->x[i] - grd->xStart)*grd->invdx);
        iy = 2 +  int((part->y[i] - grd->yStart)*grd->invdy);
        iz = 2 +  int((part->z[i] - grd->zStart)*grd->invdz);
        
        // calculate weights
        xi[0]   = part->x[i] - grd->XN_flat[get_idx(ix - 1, iy, iz, grd->nyn, grd->nzn)];
        eta[0]  = part->y[i] - grd->YN_flat[get_idx(ix, iy - 1, iz, grd->nyn, grd->nzn)];
        zeta[0] = part->z[i] - grd->ZN_flat[get_idx(ix, iy, iz - 1, grd->nyn, grd->nzn)];
        xi[1]   = grd->XN_flat[get_idx(ix, iy, iz, grd->nyn, grd->nzn)] - part->x[i];
        eta[1]  = grd->YN_flat[get_idx(ix, iy, iz, grd->nyn, grd->nzn)] - part->y[i];
        zeta[1] = grd->ZN_flat[get_idx(ix, iy, iz, grd->nyn, grd->nzn)] - part->z[i];

        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    weight[ii][jj][kk] = xi[ii] * eta[jj] * zeta[kk] * grd->invVOL;

        // set to zero local electric and magnetic field
        Exl=0.0, Eyl = 0.0, Ezl = 0.0, Bxl = 0.0, Byl = 0.0, Bzl = 0.0;
        
        for (int ii=0; ii < 2; ii++)
            for (int jj=0; jj < 2; jj++)
                for(int kk=0; kk < 2; kk++){
                    Exl += weight[ii][jj][kk]*field->Ex_flat[get_idx(ix- ii, iy -jj, iz- kk, grd->nyn, grd->nzn)];
                    Eyl += weight[ii][jj][kk]*field->Ey_flat[get_idx(ix- ii, iy -jj, iz- kk, grd->nyn, grd->nzn)];
                    Ezl += weight[ii][jj][kk]*field->Ez_flat[get_idx(ix- ii, iy -jj, iz- kk, grd->nyn, grd->nzn)];
                    Bxl += weight[ii][jj][kk]*field->Bxn_flat[get_idx(ix- ii, iy -jj, iz- kk, grd->nyn, grd->nzn)];
                    Byl += weight[ii][jj][kk]*field->Byn_flat[get_idx(ix- ii, iy -jj, iz- kk, grd->nyn, grd->nzn)];
                    Bzl += weight[ii][jj][kk]*field->Bzn_flat[get_idx(ix- ii, iy -jj, iz- kk, grd->nyn, grd->nzn)];
                    
                }
        // end interpolation
        omdtsq = qomdt2*qomdt2*(Bxl*Bxl+Byl*Byl+Bzl*Bzl);
        denom = 1.0/(1.0 + omdtsq);
        // solve the position equation
        ut= part->u[i] + qomdt2*Exl;
        vt= part->v[i] + qomdt2*Eyl;
        wt= part->w[i] + qomdt2*Ezl;
        udotb = ut*Bxl + vt*Byl + wt*Bzl;
        // solve the velocity equation
        uptilde = (ut+qomdt2*(vt*Bzl -wt*Byl + qomdt2*udotb*Bxl))*denom;
        vptilde = (vt+qomdt2*(wt*Bxl -ut*Bzl + qomdt2*udotb*Byl))*denom;
        wptilde = (wt+qomdt2*(ut*Byl -vt*Bxl + qomdt2*udotb*Bzl))*denom;
        // update position
        part->x[i] = xptilde + uptilde*dto2;
        part->y[i] = yptilde + vptilde*dto2;
        part->z[i] = zptilde + wptilde*dto2;
    } // end of iteration
    // update the final position and velocity
    part->u[i]= 2.0*uptilde - part->u[i];
    part->v[i]= 2.0*vptilde - part->v[i];
    part->w[i]= 2.0*wptilde - part->w[i];
    part->x[i] = xptilde + uptilde*dt_sub_cycling;
    part->y[i] = yptilde + vptilde*dt_sub_cycling;
    part->z[i] = zptilde + wptilde*dt_sub_cycling;
    
    
    //////////
    //////////
    ////////// BC
                                
    // X-DIRECTION: BC particles
    if (part->x[i] > grd->Lx){
        if (param->PERIODICX==true){ // PERIODIC
            part->x[i] = part->x[i] - grd->Lx;
        } else { // REFLECTING BC
            part->u[i] = -part->u[i];
            part->x[i] = 2*grd->Lx - part->x[i];
        }
    }
                                                                
    if (part->x[i] < 0){
        if (param->PERIODICX==true){ // PERIODIC
            part->x[i] = part->x[i] + grd->Lx;
        } else { // REFLECTING BC
            part->u[i] = -part->u[i];
            part->x[i] = -part->x[i];
        }
    }
        
    
    // Y-DIRECTION: BC particles
    if (part->y[i] > grd->Ly){
        if (param->PERIODICY==true){ // PERIODIC
            part->y[i] = part->y[i] - grd->Ly;
        } else { // REFLECTING BC
            part->v[i] = -part->v[i];
            part->y[i] = 2*grd->Ly - part->y[i];
        }
    }
                                                                
    if (part->y[i] < 0){
        if (param->PERIODICY==true){ // PERIODIC
            part->y[i] = part->y[i] + grd->Ly;
        } else { // REFLECTING BC
            part->v[i] = -part->v[i];
            part->y[i] = -part->y[i];
        }
    }
                                                                
    // Z-DIRECTION: BC particles
    if (part->z[i] > grd->Lz){
        if (param->PERIODICZ==true){ // PERIODIC
            part->z[i] = part->z[i] - grd->Lz;
        } else { // REFLECTING BC
            part->w[i] = -part->w[i];
            part->z[i] = 2*grd->Lz - part->z[i];
        }
    }
                                                                
    if (part->z[i] < 0){
        if (param->PERIODICZ==true){ // PERIODIC
            part->z[i] = part->z[i] + grd->Lz;
        } else { // REFLECTING BC
            part->w[i] = -part->w[i];
            part->z[i] = -part->z[i];
        }
    }
    //printf("here\n");
  }

/** particle mover gpu */
int mover_PC_gpu(struct particles* part, struct EMfield* field, struct grid* grd, struct parameters* param)
{
    struct particles *part_d;
    struct EMfield *field_d;
    struct grid *grd_d;
    struct parameters *param_d;
    struct particles *part_d2;
    struct EMfield *field_d2;
    struct grid *grd_d2;
    // print species and subcycling
    std::cout << "***  MOVER with SUBCYCLYING "<< param->n_sub_cycles << " - species " << part->species_ID << " ***" << std::endl;
 
    // auxiliary variables
    FPpart dt_sub_cycling = (FPpart) param->dt/((double) part->n_sub_cycles);
    FPpart dto2 = .5*dt_sub_cycling, qomdt2 = part->qom*dto2/param->c;
    
    std::cout << "Allocating CPU memory" << std::endl;
    part_d2 = (struct particles*) malloc(sizeof(struct particles));
    memcpy(part_d2, part, sizeof(struct particles));
    field_d2 = (struct EMfield*) malloc(sizeof(struct EMfield));
    memcpy(field_d2, field, sizeof(struct EMfield));
    grd_d2 = (struct grid*) malloc(sizeof(struct grid));
    memcpy(grd_d2, grd, sizeof(struct grid));

    std::cout << "Allocating GPU memory" << std::endl;
    cudaMalloc((void**)&part_d2->x, part->nop * sizeof(FPpart));
    cudaMalloc((void**)&part_d2->y, part->nop * sizeof(FPpart));
    cudaMalloc((void**)&part_d2->z, part->nop * sizeof(FPpart));
    cudaMalloc((void**)&part_d2->u, part->nop * sizeof(FPpart));
    cudaMalloc((void**)&part_d2->v, part->nop * sizeof(FPpart));
    cudaMalloc((void**)&part_d2->w, part->nop * sizeof(FPpart));
    cudaMemcpy(part_d2->x, part->x, part->nop * sizeof(FPpart), cudaMemcpyHostToDevice);
    cudaMemcpy(part_d2->y, part->y, part->nop * sizeof(FPpart), cudaMemcpyHostToDevice);
    cudaMemcpy(part_d2->z, part->z, part->nop * sizeof(FPpart), cudaMemcpyHostToDevice);
    cudaMemcpy(part_d2->u, part->u, part->nop * sizeof(FPpart), cudaMemcpyHostToDevice);
    cudaMemcpy(part_d2->v, part->v, part->nop * sizeof(FPpart), cudaMemcpyHostToDevice);
    cudaMemcpy(part_d2->w, part->w, part->nop * sizeof(FPpart), cudaMemcpyHostToDevice);

    cudaMalloc((void**)&field_d2->Ex_flat, grd->nxn * grd->nyn * grd->nzn * sizeof(FPfield));
    cudaMalloc((void**)&field_d2->Ey_flat, grd->nxn * grd->nyn * grd->nzn * sizeof(FPfield));
    cudaMalloc((void**)&field_d2->Ez_flat, grd->nxn * grd->nyn * grd->nzn * sizeof(FPfield));
    cudaMalloc((void**)&field_d2->Bxn_flat, grd->nxn * grd->nyn * grd->nzn * sizeof(FPfield));
    cudaMalloc((void**)&field_d2->Byn_flat, grd->nxn * grd->nyn * grd->nzn * sizeof(FPfield));
    cudaMalloc((void**)&field_d2->Bzn_flat, grd->nxn * grd->nyn * grd->nzn * sizeof(FPfield));
    cudaMemcpy(field_d2->Ex_flat, field->Ex_flat, grd->nxn * grd->nyn * grd->nzn * sizeof(FPfield), cudaMemcpyHostToDevice);
    cudaMemcpy(field_d2->Ey_flat, field->Ey_flat, grd->nxn * grd->nyn * grd->nzn * sizeof(FPfield), cudaMemcpyHostToDevice);
    cudaMemcpy(field_d2->Ez_flat, field->Ez_flat, grd->nxn * grd->nyn * grd->nzn * sizeof(FPfield), cudaMemcpyHostToDevice);
    cudaMemcpy(field_d2->Bxn_flat, field->Bxn_flat, grd->nxn * grd->nyn * grd->nzn * sizeof(FPfield), cudaMemcpyHostToDevice);
    cudaMemcpy(field_d2->Byn_flat, field->Byn_flat, grd->nxn * grd->nyn * grd->nzn * sizeof(FPfield), cudaMemcpyHostToDevice);
    cudaMemcpy(field_d2->Bzn_flat, field->Bzn_flat, grd->nxn * grd->nyn * grd->nzn * sizeof(FPfield), cudaMemcpyHostToDevice);

    cudaMalloc((void**)&grd_d2->XN_flat, grd->nxn * grd->nyn * grd->nzn * sizeof(FPfield));
    cudaMalloc((void**)&grd_d2->YN_flat, grd->nxn * grd->nyn * grd->nzn * sizeof(FPfield));
    cudaMalloc((void**)&grd_d2->ZN_flat, grd->nxn * grd->nyn * grd->nzn * sizeof(FPfield));
    cudaMemcpy(grd_d2->XN_flat, grd->XN_flat, grd->nxn * grd->nyn * grd->nzn * sizeof(FPfield), cudaMemcpyHostToDevice);
    cudaMemcpy(grd_d2->YN_flat, grd->YN_flat, grd->nxn * grd->nyn * grd->nzn * sizeof(FPfield), cudaMemcpyHostToDevice);
    cudaMemcpy(grd_d2->ZN_flat, grd->ZN_flat, grd->nxn * grd->nyn * grd->nzn * sizeof(FPfield), cudaMemcpyHostToDevice);

    cudaMalloc((void**)&part_d, sizeof(struct particles));
    cudaMalloc((void**)&field_d, sizeof(struct EMfield));
    cudaMalloc((void**)&grd_d, sizeof(struct grid));
    cudaMalloc((void**)&param_d, sizeof(struct parameters));
    cudaMemcpy(part_d, part_d2, sizeof(struct particles), cudaMemcpyHostToDevice);
    cudaMemcpy(field_d, field_d2, sizeof(struct EMfield), cudaMemcpyHostToDevice);
    cudaMemcpy(grd_d, grd_d2, sizeof(struct grid), cudaMemcpyHostToDevice);
    cudaMemcpy(param_d, param, sizeof(struct parameters), cudaMemcpyHostToDevice);

    // Pass a stripe to each kernel
    //@@ Initialize the grid and block dimensions here
    dim3 DimGrid((part->nop + TPB - 1) / TPB, 1, 1);
    dim3 DimBlock(TPB, 1, 1);

    printf("before kernel:\n");
    for (int i = 0; i < 10; i++)
        printf("part %d at %.10f, %.10f, %.10f\n", i + 159000, part->x[159000 + i], part->y[159000 + i], part->z[159000 + i]);

    // start subcycling
    for (int i_sub=0; i_sub <  part->n_sub_cycles; i_sub++){
        std::cout << "Running " << (part->nop + TPB - 1) / TPB << " thread blocks" << std::endl;
        mover_kernel<<<DimGrid, DimBlock>>>(param_d, field_d, part_d, grd_d, qomdt2, dto2, dt_sub_cycling);
        cudaDeviceSynchronize();
    } // end of one particle

    cudaMemcpy(part->x, part_d2->x, part->nop * sizeof(FPpart), cudaMemcpyDeviceToHost);
    cudaMemcpy(part->y, part_d2->y, part->nop * sizeof(FPpart), cudaMemcpyDeviceToHost);
    cudaMemcpy(part->z, part_d2->z, part->nop * sizeof(FPpart), cudaMemcpyDeviceToHost);
    cudaMemcpy(part->u, part_d2->u, part->nop * sizeof(FPpart), cudaMemcpyDeviceToHost);
    cudaMemcpy(part->v, part_d2->v, part->nop * sizeof(FPpart), cudaMemcpyDeviceToHost);
    cudaMemcpy(part->w, part_d2->w, part->nop * sizeof(FPpart), cudaMemcpyDeviceToHost);
    
    printf("\nafter kernel:\n");
    for (int i = 0; i < 10; i++)
        printf("part %d at %.10f, %.10f, %.10f\n", i + 159000, part->x[159000 + i], part->y[159000 + i], part->z[159000 + i]);

    std::cout << "Freeing GPU memory" << std::endl;
    cudaFree(field_d2->Ex_flat);
    cudaFree(field_d2->Ey_flat);
    cudaFree(field_d2->Ez_flat);
    cudaFree(field_d2->Bxn_flat);
    cudaFree(field_d2->Byn_flat);
    cudaFree(field_d2->Bzn_flat);
    cudaFree(part_d2->x);
    cudaFree(part_d2->y);
    cudaFree(part_d2->z);
    cudaFree(part_d2->u);
    cudaFree(part_d2->v);
    cudaFree(part_d2->w);
    cudaFree(grd_d2->XN_flat);
    cudaFree(grd_d2->YN_flat);
    cudaFree(grd_d2->ZN_flat);
    cudaFree(part_d);
    cudaFree(grd_d);
    cudaFree(field_d);
    cudaFree(param_d);

    std::cout << "Freeing CPU memory" << std::endl;
    free(part_d2);
    free(field_d2);
    free(grd_d2);

    return(0); // exit succcesfully
} // end of the mover

/** Interpolation Particle --> Grid: This is for species */
void interpP2G(struct particles* part, struct interpDensSpecies* ids, struct grid* grd)
{
    // arrays needed for interpolation
    FPpart weight[2][2][2];
    FPpart temp[2][2][2];
    FPpart xi[2], eta[2], zeta[2];
    
    // index of the cell
    int ix, iy, iz;
    
    
    for (register long long i = 0; i < part->nop; i++) {
        
        // determine cell: can we change to int()? is it faster?
        ix = 2 + int (floor((part->x[i] - grd->xStart) * grd->invdx));
        iy = 2 + int (floor((part->y[i] - grd->yStart) * grd->invdy));
        iz = 2 + int (floor((part->z[i] - grd->zStart) * grd->invdz));
        
        // distances from node
        xi[0]   = part->x[i] - grd->XN[ix - 1][iy][iz];
        eta[0]  = part->y[i] - grd->YN[ix][iy - 1][iz];
        zeta[0] = part->z[i] - grd->ZN[ix][iy][iz - 1];
        xi[1]   = grd->XN[ix][iy][iz] - part->x[i];
        eta[1]  = grd->YN[ix][iy][iz] - part->y[i];
        zeta[1] = grd->ZN[ix][iy][iz] - part->z[i];
        
        // calculate the weights for different nodes
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    weight[ii][jj][kk] = part->q[i] * xi[ii] * eta[jj] * zeta[kk] * grd->invVOL;
        
        //////////////////////////
        // add charge density
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    ids->rhon[ix - ii][iy - jj][iz - kk] += weight[ii][jj][kk] * grd->invVOL;
        
        
        ////////////////////////////
        // add current density - Jx
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    temp[ii][jj][kk] = part->u[i] * weight[ii][jj][kk];
        
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    ids->Jx[ix - ii][iy - jj][iz - kk] += temp[ii][jj][kk] * grd->invVOL;
        
        
        ////////////////////////////
        // add current density - Jy
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    temp[ii][jj][kk] = part->v[i] * weight[ii][jj][kk];
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    ids->Jy[ix - ii][iy - jj][iz - kk] += temp[ii][jj][kk] * grd->invVOL;
        
        
        
        ////////////////////////////
        // add current density - Jz
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    temp[ii][jj][kk] = part->w[i] * weight[ii][jj][kk];
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    ids->Jz[ix - ii][iy - jj][iz - kk] += temp[ii][jj][kk] * grd->invVOL;
        
        
        ////////////////////////////
        // add pressure pxx
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    temp[ii][jj][kk] = part->u[i] * part->u[i] * weight[ii][jj][kk];
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    ids->pxx[ix - ii][iy - jj][iz - kk] += temp[ii][jj][kk] * grd->invVOL;
        
        
        ////////////////////////////
        // add pressure pxy
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    temp[ii][jj][kk] = part->u[i] * part->v[i] * weight[ii][jj][kk];
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    ids->pxy[ix - ii][iy - jj][iz - kk] += temp[ii][jj][kk] * grd->invVOL;
        
        
        
        /////////////////////////////
        // add pressure pxz
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    temp[ii][jj][kk] = part->u[i] * part->w[i] * weight[ii][jj][kk];
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    ids->pxz[ix - ii][iy - jj][iz - kk] += temp[ii][jj][kk] * grd->invVOL;
        
        
        /////////////////////////////
        // add pressure pyy
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    temp[ii][jj][kk] = part->v[i] * part->v[i] * weight[ii][jj][kk];
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    ids->pyy[ix - ii][iy - jj][iz - kk] += temp[ii][jj][kk] * grd->invVOL;
        
        
        /////////////////////////////
        // add pressure pyz
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    temp[ii][jj][kk] = part->v[i] * part->w[i] * weight[ii][jj][kk];
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    ids->pyz[ix - ii][iy - jj][iz - kk] += temp[ii][jj][kk] * grd->invVOL;
        
        
        /////////////////////////////
        // add pressure pzz
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    temp[ii][jj][kk] = part->w[i] * part->w[i] * weight[ii][jj][kk];
        for (int ii=0; ii < 2; ii++)
            for (int jj=0; jj < 2; jj++)
                for(int kk=0; kk < 2; kk++)
                    ids->pzz[ix -ii][iy -jj][iz - kk] += temp[ii][jj][kk] * grd->invVOL;
    
    }
   
}
