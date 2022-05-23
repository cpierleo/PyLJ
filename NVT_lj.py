def mdlj(nstep=1000, rho=.8, m=6, kt=.684, kt_init=.684, dt=0.005, freq=1, mode=0):
    """ 
    Numerical integration of Newton's equations for N interacting LJ-particles
    in a 3 dimensional space using reduced LJ units (mass,sigma,epsilon):
    Dynamics: microcanonical - velocity Verlet algorithm;
    INPUT: 
      initial configuration from file 'conf_in':
      - 1th row:  x_1 y_1 z_1 (initial coordinates of 1st particle)
      -  . . . 
      - Nth row   x_N y_N z_N (initial coordinates of Nth particle)
      initial velocities extracted from Maxwell Distribution 
   OUTPUT:
   - enep   total potential energy  E_p
   - enek   total kinetic energy E_k
   - enet   total energy Ep+Ek (constant of motion for microcanonical dynamics)
   - vcmx, vcmy, vcmz  center of mass momentum (constant of motion)
    """
    from numpy import random, sqrt, sum
    from PyLJ import LJ
#
    a=(4/rho)**(1./3.)
    Lref=a*m
    N=4*m**3
    md=LJ(N, Lref)
    g=3*N-3;
    gdr_out='gdrmd.out'
    print( "# initial (kinetic) temperature kt =%8.4f" % kt_init )
    print( "# integration time step dt =%8.4f" % dt )
    print( "# number of time steps for this run ns =%d" % nstep )
    print( "# reference side of the cubic MD-box L(0) = %8.4f " % Lref )
    print( "# number of particles  N =%d" % N )
    print( "# density rho =%8.4f"  % rho )
    if mode :
        # number of particles read from file (N=4*m**3)
        fpart=open('N.out','r')     
        N=int(fpart.read())
        fpart.close()
        md.read_input(N, conf_in='conf_in.b')
    else :
        # initial positions mode=0 from hexagonal lattice
        md.fcc(m)
        # initial velocities: realization of random gaussian process with zero mean 
        # so that initial momentum is exactly zero at the initial time 
        # this results in a stationary center of mass for the N particle system 
        # write number of particles to restart GCMC
        fpart=open('N.out','w')     
        fpart.write(" %d " % N)
        fpart.close()
#
    if mode%2 == 0:
        pstd=sqrt(md.mass*kt_init)
        md.px = random.normal(0., pstd, N)
        md.py = random.normal(0., pstd, N)
        md.pz = random.normal(0., pstd, N)
        vcmx  = sum(md.px)
        vcmy  = sum(md.py)
        vcmz  = sum(md.pz)
        md.px   -= vcmx/N
        md.py   -= vcmy/N
        md.pz   -= vcmz/N
        # reduced coordinates !
        md.px *= md.L
        md.py *= md.L
        md.pz *= md.L
    #intial call for temperature difference
    md.itemp = sum( md.px*md.px + md.py*md.py + md.pz*md.pz )/(g*md.mass*md.L**2)
    print(" initial momenta sampled from maxwellian at temperature %8.4f " % kt)
    print( "# starting with box side at initial time L(t=0) = %8.4f " % md.L )
    # ofstream eout("outmd.txt");
    tt, ekt, ept, vir, enht = md.nose_hoover(N, nstep, dt, g, kt, freq)
    pres = (2*ekt + vir)/(3.*md.L**3)
    print( "# average potential energy  ep = %10.5g " % ( ept/nstep) )
    print( "# average kinetic   energy  ek = %10.5g " % ( ekt/nstep) )
    print( "# average (potential, kinetic) virial and pressure  vir= (%10.5g,%10.5g)  pres=%10.5g" % (vir/nstep, 2.*ekt/nstep, pres/nstep) )
    print( "# average Nose-Hoover total energy  enh = %10.5g " % ( enht/nstep) )
    print( "# for temperature kT = %8.3f over %d timesteps " %(2.*ekt/(nstep*g),nstep) )
    # gdr final printout - use actual value of density rho 
    md.write_gdr( N, tt, rho, gdr_out )
#   end of md - visualization of G(r)
    from matplotlib.pyplot import plot, show
    from numpy import loadtxt,ones,size
    r,gdr = loadtxt(gdr_out, unpack=True, skiprows=1 )
    plot(r,gdr,'b.',r,ones(size(r)),'k-')
    show()
