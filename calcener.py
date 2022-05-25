from numba import jit
from numpy import  zeros, rint, cos, sin, ones, pi

@jit(nopython=True,cache=True)
def enforces(rx, ry, rz, L, N):
    """ epsilon=1 , sigma=1 , rcut=3*sigma """
    c6  =  4.0
    cf6 = 24.0
    #c12 = c6
    #cf12= 2.*c12
    # standard cut-off = 3 for LJ systems
    r2cut=9./L**2
    # WCA cut-off in the minimum
    # r2cut=2.**(1./3.)
    dcut=  c6/r2cut**3
    acut= -dcut/r2cut**3
    fdx = zeros(N)
    fax = zeros(N)
    fdy = zeros(N)
    fay = zeros(N)
    fdz = zeros(N)
    faz = zeros(N)
    # zeroing
    enea=0.;
    vipa=0.
    ened=0.;
    vipd=0.
    # double loop on pairs
    for k in range(N-1) :
        for j in range(k+1,N) :
            dx = rx[k]-rx[j]
            dy = ry[k]-ry[j]
            dz = rz[k]-rz[j]
            dx-= rint(dx)
            dy-= rint(dy)
            dz-= rint(dz)
            r2 = dx**2 + dy**2 + dz**2
            if(r2<r2cut) :
                rr2 = 1./r2
                rr6 = rr2*rr2*rr2
                temp=c6*rr6
                ened+= (dcut - temp)
                enea+= (acut + temp*rr6)
                vir6 = cf6*rr6
                vir12=2.*vir6*rr6
                vipd-= vir6
                vipa+= vir12
                vir6 *=rr2
                vir12*=rr2
          # forces
                fdx[k]-= vir6*dx
                fax[k]+=vir12*dx
                fdy[k]-= vir6*dy
                fay[k]+=vir12*dy
                fdz[k]-= vir6*dz
                faz[k]+=vir12*dz
                fdx[j]+= vir6*dx
                fax[j]-=vir12*dx
                fdy[j]+= vir6*dy
                fay[j]-=vir12*dy
                fdz[j]+= vir6*dz
                faz[j]-=vir12*dz
    return ( enea, ened, vipa, vipd, fax, fdx, fay, fdy, faz, fdz )

@jit(nopython=True,cache=True)
def energymove(rx, ry, rz, L, N, rxt, ryt, rzt, nt ):
    c6 = 4.0
    cf6=24.0
    # standard cut-off = 3 for LJ systems
    r2cut=9./L**2
    # WCA cut-off in the minimum
    # r2cut=2.**(1./3.)
    dcut=  c6/r2cut**3
    acut= -dcut/r2cut**3
    # current position
    enea=0.;
    vira=0.
    ened=0.;
    vird=0.
    # double loop on pairs
    for k in range(N):
        dx = rx[nt]-rx[k]
        dy = ry[nt]-ry[k]
        dz = rz[nt]-rz[k]
        dx-= rint(dx)
        dy-= rint(dy)
        dz-= rint(dz)
        r2 = dx**2 + dy**2 + dz**2
        if(k!=nt and r2<r2cut):
            rr2 = 1./r2
            rr6 = rr2*rr2*rr2
            temp=c6*rr6
            ened+= (dcut - temp)
            enea+= (acut + temp*rr6)
            vir6 = cf6*rr6
            vird-= vir6
            vira+= 2.*vir6*rr6
    # New trial position 
    # zeroing
    enta=0.;
    vita=0.
    entd=0.;
    vitd=0.
    # double loop on pairs
    for k in range(N):
        dx = rxt-rx[k]
        dy = ryt-ry[k]
        dz = rzt-rz[k]
        dx-= rint(dx)
        dy-= rint(dy)
        dz-= rint(dz)
        r2 = dx**2 + dy**2 + dz**2
        if(k!=nt and r2<r2cut):
            rr2 = 1./r2
            rr6 = rr2*rr2*rr2
            temp=c6*rr6
            entd+= (dcut - temp)
            enta+= (acut + temp*rr6)
            vir6 = cf6*rr6
            vitd-= vir6
            vita+= 2.*vir6*rr6
    return (entd-ened,enta-enea,vitd-vird,vita-vira)

@jit(nopython=True,cache=True)
def energyadd(rx, ry, rz, L, N, rxt, ryt, rzt ):
    c6 = 4.0
    cf6=24.0
    # standard cut-off = 3 for LJ systems
    r2cut=9./L**2
    # WCA cut-off in the minimum
    # r2cut=2.**(1./3.)
    dcut=  c6/r2cut**3
    acut= -dcut/r2cut**3
    # new particle
    enea=0.;
    vira=0.
    ened=0.;
    vird=0.
    # double loop on pairs
    for k in range(N):
        dx = rxt-rx[k]
        dy = ryt-ry[k]
        dz = rzt-rz[k]
        dx-= rint(dx)
        dy-= rint(dy)
        dz-= rint(dz)
        r2 = dx**2 + dy**2 + dz**2
        if(r2<r2cut):
            rr2 = 1./r2
            rr6 = rr2*rr2*rr2
            temp=c6*rr6
            ened+= (dcut - temp)
            enea+= (acut + temp*rr6)
            vir6 = cf6*rr6
            vird-= vir6
            vira+= 2.*vir6*rr6
    return (ened,enea,vird,vira)

@jit(nopython=True,cache=True)
def energydel(rx, ry, rz, L, N, nt):
    c6 = 4.0
    cf6=24.0
    # standard cut-off = 3 for LJ systems
    r2cut=9./L**2
    # WCA cut-off in the minimum
    # r2cut=2.**(1./3.)
    dcut=  c6/r2cut**3
    acut= -dcut/r2cut**3
    # current position 
    enea=0.;
    vira=0.
    ened=0.;
    vird=0.
    # double loop on pairs
    for k in range(N):
        dx = rx[nt]-rx[k]
        dy = ry[nt]-ry[k]
        dz = rz[nt]-rz[k]
        dx-= rint(dx)
        dy-= rint(dy)
        dz-= rint(dz)
        r2 = dx**2 + dy**2 + dz**2
        if(k!=nt and r2<r2cut):
            rr2 = 1./r2
            rr6 = rr2*rr2*rr2
            temp=c6*rr6
            ened+= (dcut - temp)
            enea+= (acut + temp*rr6)
            vir6 = cf6*rr6
            vird-= vir6
            vira+= 2.*vir6*rr6
    return (ened,enea,vird,vira)


# +
@jit(nopython=True,cache=True)
def calcsofk(rx, ry, rz, nmax, N):
    
    cp = zeros((3, 2 * nmax + 1, N))
    sp = zeros((3, 2 * nmax + 1, N))
    cp[:, nmax, :] = 1.0     # this is the k=0 vector
    # these are the contribution of the first k-vector (1,0,0) and symmetries.
    cp[0, nmax+1, :] = cos(2 * pi * rx[:])
    cp[1, nmax+1, :] = cos(2 * pi * ry[:])
    cp[2, nmax+1, :] = cos(2 * pi * rz[:])
    sp[0, nmax+1, :] = sin(2 * pi * rx[:])
    sp[1, nmax+1, :] = sin(2 * pi * ry[:])
    sp[2, nmax+1, :] = sin(2 * pi * rz[:])
    #
    # for n in range(nmax-1):
    for n in range(nmax):
        m = nmax + 1 + n
        cp[:, m, :] = (cp[:, nmax+1, :] * cp[:, m-1, :] -
                       sp[:, nmax+1, :] * sp[:, m-1, :])
        sp[:, m, :] = (sp[:, nmax+1, :] * cp[:, m-1, :] +
                       cp[:, nmax+1, :] * sp[:, m-1, :])
    for n in range(1, nmax+1):
        mm = nmax-n
        mp = nmax+n
        # print(mp,mm)
        # impose the symmetry of cos and sin along the axes
        cp[:, mm, :] = cp[:, mp, :]
        sp[:, mm, :] = - sp[:, mp, :]

    # these are the sum of cos and sin over particles.
    # indices are along the three directions (nx, ny, nz)
    csum = zeros((2*nmax+1, 2*nmax+1, 2*nmax+1), dtype=float)
    ssum = zeros((2*nmax+1, 2*nmax+1, 2*nmax+1), dtype=float)

    for k in range(N):
        for nx in range(2 * nmax + 1):
            for ny in range(2 * nmax + 1):
                for nz in range(2 * nmax + 1):
                    csum[nx, ny, nz] += (cp[0, nx, k] * cp[1, ny, k] * cp[2, nz, k] -
                                         cp[0, nx, k] * sp[1, ny, k] * sp[2, nz, k] -
                                         sp[0, nx, k] * sp[1, ny, k] * cp[2, nz, k] +
                                         sp[0, nx, k] * cp[1, ny, k] * sp[2, nz, k])
                    ssum[nx, ny, nz] += (sp[0, nx, k] * cp[1, ny, k] * cp[2, nz, k] -
                                         sp[0, nx, k] * sp[1, ny, k] * sp[2, nz, k] +
                                         cp[0, nx, k] * sp[1, ny, k] * cp[2, nz, k] +
                                         cp[0, nx, k] * cp[1, ny, k] * sp[2, nz, k])
 
    sofk = zeros((2*nmax+1, 2*nmax+1, 2*nmax+1), dtype=float)
    sofk = (csum**2 + ssum**2) / N
    return sofk


    return(sofk)

