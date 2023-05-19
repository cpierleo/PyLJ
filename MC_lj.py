def mclj(nstep=1000, rho=0.5, m=5, kt=1.1876, delta=0.25, freq=10, mode=0):
    """  
    Numerical integration of Newton's equations for N interacting LJ-particles
    in a 2 dimensional space using reduced LJ units (m,sigma,epsilon):
    Dynamics: microcanonical - Verlet algorithm;
    INPUT: 
      initial configuration from file 'conf_in':
      - 1th row:  x_1 y_1  (initial coordinates of 1st particle)
      -  . . . 
      - Nth row   x_N y_N  (initial coordinates of Nth particle)
      initial velocities extracted from Maxwell Distribution 
   OUTPUT:
   - enep   total potential energy  E_p
   - enek   total kinetic energy E_k
   - enet   total energy Ep+Ek (constant of motion for microcanonical dynamics)
   - vcmx, vcmy  center of mass momentum (constant of motion)
    """
    from numpy import random, sqrt, sum
    from PyLJ import LJ
#
    a=(4/rho)**(1./3.)
    Lref=a*m
    N=4*m**3
    mc=LJ(N, Lref)
    g=3*N-3;
    conf_out='conf_in'
    gdr_out='gdrmc.out'
    print( "# temperature kt =%8.4f" % kt )
    beta = 1./ kt
    print( "# maximum displacement =%8.4f" % delta )
    print( "# number of moves per particle for this run ns =%d" % nstep )
    print( "# reference side of the cubic MD-box L(0) = %8.4f " % Lref )
    print( "# number of particles  N =%d" % N )
    print( "# density rho =%8.4f"  % rho )
# initial positions mode=0 from hexagonal lattice
    if mode :
    # number of particles read from file (N=4*m**3)
        fpart=open('N.out','r')     
        N=int(fpart.read())
        fpart.close()
        mc.read_input(N, conf_in='conf_in.b')
    else :
        mc.fcc(m)
    # write number of particles to restart GCMC
        fpart=open('N.out','w')     
        fpart.write(" %d " % N)
        fpart.close()
#
# ofstream eout("outmd.txt");
    tt, ept, pres = mc.metropolis( N, beta, nstep, delta, freq)
    print( "# average potential energy  ep = %8.3f " % ( ept/nstep) )
    print( "# average potential virial  vir= %8.3f " % (pres/nstep) )
    print( "# for temperature kT = %8.3f over %d timesteps " %(kt,nstep) )
# gdr final printout - use actual value of density rho 
    mc.write_gdr( N, tt, rho, gdr_out )
#   end of md - visualization of G(r)
    from matplotlib.pyplot import plot, show
    %matplotlib widget
    from numpy import loadtxt,ones,size
    r,gdr = loadtxt(gdr_out, unpack=True, skiprows=1 )
    plot(r,gdr,'b.',r,ones(size(r)),'k-')
    show()
