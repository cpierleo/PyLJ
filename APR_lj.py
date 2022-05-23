def mdlj(nstep=1000, rho=0.84, m=6, kt=0.694, pres=1., dt=0.005, freq=1, mode=0):
    """ Lausanne 9 Feb 2017 - APR isotropic-version
    Numerical integration of Newton's equations for N interacting LJ-particles
    in a 3 dimensional space using reduced LJ units (mass,sigma,epsilon):
    Dynamics:  
     - andersen -  9 step trotter propagator
    INPUT: 
      initial configuration and momenta from file 'conf_in.b':
      initial particle momenta optionally extracted from Maxwell Distribution 
   OUTPUT:
   - enep   total potential energy  Ep
   - enek   total kinetic energy Ek
   - ebox   potential+kinetic energy Eb for box coordinate L
   - enet   total energy Ep+Ek+Eb (constant of motion for andersen dynamics)
   - vcmx, vcmy, vcmz  center of mass momentum (constant of motion)
    """
    from numpy import random, sqrt, sum
    from PyLJ import LJ
#
    a=(4/rho)**(1./3.)
    Lref=a*m
    N=4*m**3
    md=LJ(N, Lref)
    md.pext=pres
    g=3*N-3;
    gdr_out='gdrmd.out'
    print( "# external pressure Pext =%8.4f" % pres)
    print( "# reference side for the cubic MD-box Lr = %8.4f " % Lref )
    print( "# initial (kinetic) temperature kt =%8.4f" % kt )
    print( "# integration time step dt =%8.4f" % dt )
    print( "# number of time steps for this run ns =%d" % nstep )
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
    if mode==2:
        pstd=sqrt(md.mass*kt)
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
        print(" initial momenta sampled from maxwellian at temperature %8.4f " % kt)
    print( "# starting with box side at initial time L(t=0) = %8.4f " % md.L )
# ofstream eout("outmd.txt");
    tt, ekt, ept, ebt, vpt, vot = md.aprrun(N, nstep, dt, freq)
    avp = (2*ekt + vpt)/(3.*md.L**3)
    print( "# average potential energy    ep = %10.5g " % ( ept/nstep) )
    print( "# average kinetic   energy    ek = %10.5g " % ( ekt/nstep) )
    print( "# average box energy(pot+kin) eb = %10.5g " % ( ebt/nstep) )
    print( "# average potential virial    vir= %10.5g " % ( vpt/nstep) )
    print( "# average box volume          vol= %10.5g " % ( vot/nstep) )
    print( "# for temperature        kT = %8.3f over %d timesteps " %(2.*ekt/(nstep*g),nstep) )
    print( "# for internal pressure Pint= %8.3g over %d timesteps " %(avp/nstep,nstep) )
# gdr final printout using L from average volume 
    rhoavg = N/(vot/nstep)
    md.write_gdr( N, tt, rhoavg, gdr_out )
#   end of md - visualization of G(r)
    from matplotlib.pyplot import plot, show 
    from numpy import loadtxt,ones,size
    r,gdr = loadtxt(gdr_out, unpack=True, skiprows=1 )
    plot(r,gdr,'b.',r,ones(size(r)),'k-')
    show()
