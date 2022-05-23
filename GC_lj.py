def gclj(nstep=200,mux=0.07,rho=0.5,m=5,kt=1.1876,delta=0.25,freq=10):
    """
    Numerical Simulation of N interacting LJ-particles 
    in a 3 dimensional space using reduced LJ units (m,sigma,epsilon):
    Dynamics: microcanonical - Verlet algorithm;
    MonteCarlo: Metropolis canonical sampling
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
    from numpy import random, sqrt, sum, arange
    from PyLJ import LJ
#
    a=(4/rho)**(1./3.)
    Lref=a*m
    N=4*m**3
    gc=LJ(N, Lref)
    conf_out='conf_in'
    gdr_out='gdr.out'
    print( "# temperature kt =%8.4f" % kt )
    beta = 1./kt
    print( "# maximum displacement =%8.4f" % delta )
    print( "# number of moves per particle for this run ns =%d" % nstep )
    print( "# reference side of the cubic MD-box L(0) = %8.4f " % Lref )
    vol = Lref**3
    zexc = mux*vol
# number of particles read from file (N=4*m**3)
    fpart=open('N.out','r')     
    N=int(fpart.read())
    fpart.close()
    print( "# initial number of particles  N = %d " % N )
# initial positions from file
    gc.read_input(N, conf_in='conf_in.b')
    print( "# initial density rho =%8.4f "  % (N/vol) )
    # gran canonical montecarlo
    tt, Nf, Navg, ept, pres, Nhist = gc.gcmc( N, beta, zexc, nstep, delta, freq)
    print("# run:   nstep    <N>      kT    <Ep>     <Vir> ")
    print( "#  MC  %6d  %5d %7.4f  %8.3f  %8.3f" % ( nstep, Navg, kt, ept/nstep, pres/nstep) )
    # final printout
    fpart=open('N.out','w')     
    fpart.write(" %d " % Nf)
    fpart.close()
    # gdr final printout - use actual value of density rho 
    gc.write_gdr( N, tt, Navg/vol, gdr_out )
#   end of md - visualization of G(r) & hist(N)
    from matplotlib.pyplot import figure,plot, show
    from numpy import loadtxt,ones,size
    r,gdr = loadtxt(gdr_out, unpack=True, skiprows=1 )
    f = figure(0,  figsize=(9, 3), dpi=100)
    s1 = f.add_subplot(1, 2, 1)
    s2 = f.add_subplot(1, 2, 2)
    s1.plot(r,gdr,'b.',r,ones(size(r)),'k-')
    bins=arange(Nhist.min(),Nhist.max()+2)-0.5
    s2.hist(Nhist, bins=bins, density=False, rwidth=0.96)
    show()
