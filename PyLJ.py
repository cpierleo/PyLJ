class LJ :

  def __init__(self, N, Lr):
    from numpy import zeros,sqrt, max
    #-start-------------------------
    self.t = 0.
    self.mass= 1.
    self.L   = Lr
    self.Ngc = 3*N
    self.NHtau = 0.1
    self.NHchi = 0.0
    self.NHxi = 0.0
    self.itemp = 0.0
    self.PL  = 0.
    self.Q   = 100.*N
    #eps     = 1.
    #sig     = 1.
    self.rx      = zeros( self.Ngc )
    self.ry      = zeros( self.Ngc )
    self.rz      = zeros( self.Ngc )
    self.px      = zeros( self.Ngc )
    self.py      = zeros( self.Ngc )
    self.pz      = zeros( self.Ngc )
    self.fax     = zeros( N )
    self.fay     = zeros( N )
    self.faz     = zeros( N )
    self.fdx     = zeros( N )
    self.fdy     = zeros( N )
    self.fdz     = zeros( N )
    self.kg      = 512
    self.gcount  = zeros( self.kg)
    self.ekin    = 0.0
    self.ene     = 0.0
    self.etot    = 0.0
    self.pext    = 1.
    # L can fluctuate rmax for gdr limited to initial L_ref/2.5
    rmax = Lr/2.5
    self.r2max = rmax * rmax
    self.ldel = rmax/self.kg

  def calcgdr(self, N ):
    from numpy import sqrt, rint, zeros, int_
    for k in range(N-1) :
        j=k+1
        #for j in range(k+1,N) :
        dx = self.rx[k]-self.rx[j:N]
        dy = self.ry[k]-self.ry[j:N]
        dz = self.rz[k]-self.rz[j:N]
        dx[...]-= rint(dx)
        dy[...]-= rint(dy)
        dz[...]-= rint(dz)
        dx[...] = dx*self.L
        dy[...] = dy*self.L
        dz[...] = dz*self.L
        r2 = dx*dx + dy*dy + dz*dz
        # using the mask array "b" for speedup
        b = r2 < self.r2max
        lm  = sqrt(r2[b])
        #if lm<self.kg :
        for elm in lm :
            self.gcount[int(elm/self.ldel)]+=2.  # factor of 2 for gdr normalization
    #return

  def verlet(self, N, nstep, dt, freq=1):
    from numpy import sum, rint
    from calcener import enforces
    import pickle
    self.t    = 0
    tt   = 0
    ept  = 0.
    ekt  = 0.
    vir = 0.
    dth=0.5*dt
    dtm=dt/(self.mass*self.L**2)
    # initial call to compute starting energies, forces and virial
    epa,epd,vira,vird,self.fax,self.fdx,self.fay,self.fdy,self.faz,self.fdz = enforces(self.rx,self.ry,self.rz,self.L,N)
    self.fax/= self.L**12
    self.fdx/= self.L**6
    self.fay/= self.L**12
    self.fdy/= self.L**6
    self.faz/= self.L**12
    self.fdz/= self.L**6
    enep = epa/self.L**12+epd/self.L**6
    virp = vira/self.L**12 + vird/self.L**6
    enek = 0.5*sum( self.px[:N]**2 + self.py[:N]**2 + self.pz[:N]**2 )/(self.mass*self.L**2)
    vcmx = sum(self.px[:N])
    vcmy = sum(self.py[:N])
    vcmz = sum(self.pz[:N])
    data = 'NVE_data.out'
    out_data = open(data, 'w+')
    print( "   'time'    'enep'    'enek'  'vir_p'   'enet'        'vcmx'   'vcmy'   'vcmz'")
    print (" %8.3f %9.4g %9.4g %9.4g %10.7f %7.2e %7.2e %7.2e" % (self.t, enep/N, enek/N, virp, enep+enek, vcmx, vcmy,vcmz) )
    out_data.write("   'time'    'enep'    'enek'  'vir_p'   'enet'        'vcmx'   'vcmy'   'vcmz'\n")
    out_data.write(" %8.3f %9.4g %9.4g %9.4g %10.7f %7.2e %7.2e %7.2e\n" % (self.t, enep/N, enek/N, virp, enep+enek, vcmx, vcmy,vcmz) )
    for pas in range(nstep) :
        vcmx = 0.
        vcmy = 0. 
        vcmz = 0. 
        self.t   += dt
        # advance one step
        # momenta first 
        self.px[:N] += (self.fax+self.fdx)*dth
        self.py[:N] += (self.fay+self.fdy)*dth
        self.pz[:N] += (self.faz+self.fdz)*dth        
        # positions second
        self.rx[:N] += dtm*self.px[:N]
        self.ry[:N] += dtm*self.py[:N]
        self.rz[:N] += dtm*self.pz[:N]
        self.rx[:N] -= rint(self.rx[:N])
        self.ry[:N] -= rint(self.ry[:N])
        self.rz[:N] -= rint(self.rz[:N])
        # compute forces
        epa,epd,vira,vird,self.fax,self.fdx,self.fay,self.fdy,self.faz,self.fdz = enforces(self.rx,self.ry,self.rz,self.L,N)
        enep = epa/self.L**12 + epd/self.L**6
        virp = vira/self.L**12 + vird/self.L**6
        self.fax/= self.L**12
        self.fdx/= self.L**6
        self.fay/= self.L**12
        self.fdy/= self.L**6
        self.faz/= self.L**12
        self.fdz/= self.L**6
        # momenta thrid
        self.px[:N] += (self.fax+self.fdx)*dth
        self.py[:N] += (self.fay+self.fdy)*dth
        self.pz[:N] += (self.faz+self.fdz)*dth      
        vcmx = sum(self.px[:N])
        vcmy = sum(self.py[:N])
        vcmz = sum(self.pz[:N])
        enek = 0.5*sum( self.px[:N]**2 + self.py[:N]**2 + self.pz[:N]**2 )/(self.mass*self.L**2)
        # computing gdr and single step printout ...
        ekt += enek
        ept += enep
        vir += virp
        if (pas+1)%freq==0 : 
           # compute g(R)
           self.calcgdr( N)
           # save configuration for VMD in xyz format
           self.writexyz(N)
           print (" %8.3f %9.4g %9.4g %9.4g %10.7f %7.2e %7.2e %7.2e" % (self.t, enep/N, enek/N, virp, enep+enek, vcmx, vcmy,vcmz) )
           out_data.write(" %8.3f %9.4g %9.4g %9.4g %10.7f %7.2e %7.2e %7.2e\n" % (self.t, enep/N, enek/N, virp, enep+enek, vcmx, vcmy,vcmz) )
           tt += 1
        # end of md run
        # final configuration
    out_data.close()
    self.write_input(N, pas, conf_out='conf_in.b')
    return (tt, ekt, ept, vir)

  def nose_hoover(self, N, nstep, dt, G, kt, freq=1):
    from numpy import sum, rint, exp
    from calcener import enforces
    import pickle
    self.t    = 0
    tt   = 0
    enht=0.
    ept=0.
    ekt=0.
    vir=0.
    dth=0.5*dt
    dthh = .5*dth
    dtm=dt/(self.mass*self.L**2)
    # initial call to compute starting energies, forces and virial
    epa,epd,vira,vird,self.fax,self.fdx,self.fay,self.fdy,self.faz,self.fdz = enforces(self.rx,self.ry,self.rz,self.L,N)
    self.fax/= self.L**12
    self.fdx/= self.L**6
    self.fay/= self.L**12
    self.fdy/= self.L**6
    self.faz/= self.L**12
    self.fdz/= self.L**6 
    enh = G*kt*(.5*self.NHtau*self.NHchi**2 + self.NHxi)
    enep = epa/self.L**12+epd/self.L**6
    virp = vira/self.L**12 + vird/self.L**6
    enek = 0.5*sum( self.px[:N]**2 + self.py[:N]**2 + self.pz[:N]**2 )/(self.mass*self.L**2)
    vcmx = sum(self.px[:N])
    vcmy = sum(self.py[:N])
    vcmz = sum(self.pz[:N])
    data = 'NVT_data.out'
    out_data = open(data, 'w+')
    out_data.write("#   'time'    'enep'    'enek'  'vir_p'   'enh'   'enet'        'enht'        'vcmx'   'vcmy'   'vcmz'   'inst_T'   'Delta_T'\n")
    out_data.write(" %8.3f %9.4g %9.4g %9.4g %9.4g %10.7f %10.7f %7.2e %7.2e %7.2e %9.4g %9.4g\n" % (self.t, enep/N, enek/N, virp, enh, enep+enek, enep+enek+enh, vcmx, vcmy, vcmz, self.itemp, self.itemp-kt))
    print( "   'time'    'enep'    'enek'  'vir_p'   'enh'   'enet'        'enht'        'vcmx'   'vcmy'   'vcmz'   'inst_T'   'Delta_T'")
    print (" %8.3f %9.4g %9.4g %9.4g %9.4g %10.7f %10.7f %7.2e %7.2e %7.2e %9.4g %9.4g" % (self.t, enep/N, enek/N, virp, enh, enep+enek, enep+enek+enh, vcmx, vcmy, vcmz, self.itemp, self.itemp-kt) )
    for pas in range(nstep) :
        self.t   += dt
        # advance one step
        # Nose-Hoover coordinate first: 
        self.NHxi += dthh*self.NHchi
        # Nose-Hoover momentum second:
        self.itemp = 2.*enek/G
        fnh = (self.itemp/kt - 1.)/self.NHtau
        self.NHchi += dth*fnh
        # Nose-Hoover coordinate third
        self.NHxi += dthh*self.NHchi
        # momenta fourth
        #Scaling of momenta
        s = exp(-dth*self.NHchi)
        f = (1 - s)/self.NHchi
        self.px[:N] = s*self.px[:N] + f*(self.fax+self.fdx)
        self.py[:N] = s*self.py[:N] + f*(self.fay+self.fdy)
        self.pz[:N] = s*self.pz[:N] + f*(self.faz+self.fdz)     
        # positions fifth
        self.rx[:N] += dtm*self.px[:N]
        self.ry[:N] += dtm*self.py[:N]
        self.rz[:N] += dtm*self.pz[:N]
        self.rx[:N] -= rint(self.rx[:N])
        self.ry[:N] -= rint(self.ry[:N])
        self.rz[:N] -= rint(self.rz[:N])
        # compute forces
        epa,epd,vira,vird,self.fax,self.fdx,self.fay,self.fdy,self.faz,self.fdz = enforces(self.rx,self.ry,self.rz,self.L,N)
        enep = epa/self.L**12 + epd/self.L**6
        virp = vira/self.L**12 + vird/self.L**6
        self.fax/= self.L**12
        self.fdx/= self.L**6
        self.fay/= self.L**12
        self.fdy/= self.L**6
        self.faz/= self.L**12
        self.fdz/= self.L**6
        # momenta seventh
        #Scaling of momenta
        s = exp(-dth*self.NHchi)
        f = (1 - s)/self.NHchi
        self.px[:N] = s*self.px[:N] + f*(self.fax+self.fdx)
        self.py[:N] = s*self.py[:N] + f*(self.fay+self.fdy)
        self.pz[:N] = s*self.pz[:N] + f*(self.faz+self.fdz)     
        vcmx = sum(self.px[:N])
        vcmy = sum(self.py[:N])
        vcmz = sum(self.pz[:N])
        enek = 0.5*sum( self.px[:N]**2 + self.py[:N]**2 + self.pz[:N]**2 )/(self.mass*self.L**2)
        # Nose-Hoover coordinate seventh
        self.NHxi += dthh*self.NHchi
        # Nose-Hoover momentum eighth
        self.itemp = 2.*enek/G
        fnh = (self.itemp/kt - 1.)/self.NHtau
        self.NHchi += dth*fnh
        # Nose-Hoover coordinate nineth
        self.NHxi += dthh*self.NHchi
        enh = G*kt*(.5*self.NHtau*self.NHchi**2 + self.NHxi)
        # computing gdr and single step printout ...
        enht+= enh
        ekt += enek
        ept += enep
        vir += virp
        if (pas+1)%freq==0 : 
           # compute g(R)
           self.calcgdr(N)
           # save configuration for VMD in xyz format
           self.writexyz(N)
           print (" %8.3f %9.4g %9.4g %9.4g %9.4g %10.7f %10.7f %7.2e %7.2e %7.2e %9.4g %9.4g" % (self.t, enep/N, enek/N, virp, enh, enep+enek, enep+enek+enh, vcmx, vcmy, vcmz, self.itemp, self.itemp-kt) )
           out_data.write(" %8.3f %9.4g %9.4g %9.4g %9.4g %10.7f %10.7f %7.2e %7.2e %7.2e %9.4g %9.4g\n" % (self.t, enep/N, enek/N, virp, enh, enep+enek, enep+enek+enh, vcmx, vcmy, vcmz, self.itemp, self.itemp-kt))
           tt += 1
        # end of md run
        # final configuration
    out_data.close()
    self.write_input(N, pas, conf_out='conf_in.b')
    return (tt, ekt, ept, vir, enht)


  def aprrun(self, N, nstep, dt, freq=1):
      from numpy import sum, rint
      from calcener import enforces
      import pickle
      self.t   = 0
      tt  = 0
      ept = 0.
      ekt = 0.
      vpt = 0.
      vot = 0.
      ebt = 0.
      dth=0.5*dt
      dt4=0.5*dth
      dtq=dth/self.Q
      dtm=dt/self.mass
      # initial call to compute starting energies, forces and virial
      epa,epd,vira,vird,self.fax,self.fdx,self.fay,self.fdy,self.faz,self.fdz = enforces(self.rx,self.ry,self.rz,self.L,N)
      enep = epa/self.L**12+epd/self.L**6
      virp = vira/self.L**12 + vird/self.L**6
      virk  = sum( self.px[:N]**2 + self.py[:N]**2 + self.pz[:N]**2 )/self.mass
      enek = 0.5*virk/self.L**2
      volu = self.L**3
      ebox = 0.5*self.PL*self.PL/self.Q + self.pext*volu
      enet = enep+enek+ebox
      vcmx = sum(self.px[:N])
      vcmy = sum(self.py[:N])
      vcmz = sum(self.pz[:N])
      print( "   'time'    'enep'    'enek'  'vir_p'   'vol'    'enet'        'vcmx'   'vcmy'   'vcmz'")
      print (" %8.3f %9.4g %9.4g %9.4g %9.4g %10.7f %7.2e %7.2e %7.2e" % (self.t, enep/N, enek/N, virp, volu, enet, vcmx, vcmy,vcmz) )
      #
      for pas in range(nstep) :
          vcmx = 0.
          vcmy = 0. 
          vcmz = 0. 
          self.t   += dt
      # start integration step
          # G1(h/4)
          self.PL += (virk/self.L**2 +virp -3.*self.pext*self.L**3)*dt4/self.L
          # G2(h/2) 
          self.px[:N] += (self.fax/self.L**12+self.fdx/self.L**6)*dth
          self.py[:N] += (self.fay/self.L**12+self.fdy/self.L**6)*dth
          self.pz[:N] += (self.faz/self.L**12+self.fdz/self.L**6)*dth
          virk = sum( self.px[:N]**2 + self.py[:N]**2 + self.pz[:N]**2 )/self.mass
          # G1(h/4)
          self.PL += (virk/self.L**2+virp -3.*self.pext*self.L**3)*dt4/self.L
          # G3(h/2)
          self.L  += self.PL*dtq
          # G4(h) 
          self.rx[:N] += self.px[:N]*dtm/self.L**2
          self.ry[:N] += self.py[:N]*dtm/self.L**2
          self.rz[:N] += self.pz[:N]*dtm/self.L**2
          self.rx[:N] -= rint(self.rx[:N])
          self.ry[:N] -= rint(self.ry[:N])
          self.rz[:N] -= rint(self.rz[:N])
          epa,epd,vira,vird,self.fax,self.fdx,self.fay,self.fdy,self.faz,self.fdz = enforces(self.rx,self.ry,self.rz,self.L,N)
          # G3(h/2)
          self.L  += self.PL*dtq
          # question L is changed --> enep and virp
          enep = epa/self.L**12 + epd/self.L**6
          virp = vira/self.L**12 + vird/self.L**6
          # G1(h/4)
          self.PL += (virk/self.L**2+virp -3.*self.pext*self.L**3)*dt4/self.L
          # G2(h/2) 
          self.px += (self.fax/self.L**12+self.fdx/self.L**6)*dth
          self.py += (self.fay/self.L**12+self.fdy/self.L**6)*dth
          self.pz += (self.faz/self.L**12+self.fdz/self.L**6)*dth
          virk = sum( self.px[:N]**2 + self.py[:N]**2 + self.pz[:N]**2 )/self.mass
          enek = 0.5*virk/self.L**2
          # G1(h/4)
          self.PL += (virk/self.L**2+virp -3.*self.pext*self.L**3)*dt4/self.L
          volu = self.L**3
          ebox = 0.5*self.PL*self.PL/self.Q + self.pext*volu
      # end of integration step     
          vcmx = sum(self.px)
          vcmy = sum(self.py)
          vcmz = sum(self.pz)
          # computing gdr and single step printout ...
          ekt += enek
          ept += enep
          vpt += virp
          vot += volu
          ebt += ebox
          enet = enep+enek+ebox
          if (pas+1)%freq==0 : 
              # compute g(R)
              self.calcgdr( N)
              # save configuration for VMD in xyz format
              self.writexyz(N)
              print (" %8.3f %9.4g %9.4g %9.4g %9.4g %10.7f %7.2e %7.2e %7.2e" % (self.t, enep/N, enek/N, virp, volu, enet, vcmx, vcmy,vcmz) )
              tt += 1
          # end of md run
          # final configuration
      self.write_input(N, pas, conf_out='conf_in.b')
      return (tt, ekt, ept, ebt, vpt, vot)

  def metropolis(self, N, beta, nstep, delta, freq=100):
    from numpy import random, rint, exp
    from calcener import enforces, energymove
    tt   = 0
    ept  = 0.
    pres = 0.
    # initial energy and virial of forces 
    delta /= self.L
    epa,epd,vira,vird,self.fax,self.fdx,self.fay,self.fdy,self.faz,self.fdz = enforces(self.rx,self.ry,self.rz,self.L,N)
    enep = epa/self.L**12 + epd/self.L**6
    virp = vira/self.L**12 + vird/self.L**6
    print("# initial values enep = %8.3f virial = %8.3f " % (enep,virp) )
    print( "     'pas'    'enep'  'virial'   'acc' ")
    acc = 0.
    for pas in range(nstep) :
        self.t   += 1
        # advance one step with one move per particle (on average)
        randomparticle = random.random_integers(N, size=(N))-1
        csi = random.random(N) 
        deltax = delta*(2*random.random(N) - 1.)  
        deltay = delta*(2*random.random(N) - 1.)  
        deltaz = delta*(2*random.random(N) - 1.)  
        for i in range(N) :
        # trial positions
            nt = randomparticle[i]
            rxt = self.rx[nt] + deltax[i]
            ryt = self.ry[nt] + deltay[i]
            rzt = self.rz[nt] + deltaz[i]
            rxt -= rint(rxt)
            ryt -= rint(ryt)
            rzt -= rint(rzt)
            # computing energy difference
            (ened,enea,vird,vira) = energymove(self.rx, self.ry, self.rz, self.L, N, rxt, ryt, rzt, nt )
            enediff = enea/self.L**12 + ened/self.L**6
            #print(' # check  ', enediff)
            if csi[i] < exp( -beta*enediff ) :
                self.rx[nt] = rxt
                self.ry[nt] = ryt
                self.rz[nt] = rzt
                acc += 1
                enep += enediff
                virp += (vira/self.L**12 + vird/self.L**6)
                #print(' # check ',rxt,ryt,rzt)
        ept += enep
        pres+= virp 
        # computing gdr and single step printout ...
        if (pas+1)%freq==0 : 
           self.calcgdr( N )
           print(" %8d %9.4f %9.4f  %6.2f " % (self.t, enep/N, virp/N, acc*100./(self.t*N)) )
           tt += 1
        # end of mc run   
    # final configuration
    self.write_input(N, pas, conf_out='conf_in.b')
    self.writexyz( N )
    return (tt, ept, pres)

  def gcmc(self, N, beta, zexc, nstep, delta, freq=100):
    from numpy import random, exp, rint, zeros, int32
    from calcener import enforces, energymove, energyadd, energydel
    # initial energy and virial of forces 
    delta *= 2./self.L
    epa,epd,vira,vird,self.fax,self.fdx,self.fay,self.fdy,self.faz,self.fdz = enforces(self.rx,self.ry,self.rz,self.L,N)
    enep = epa/self.L**12 + epd/self.L**6
    virp = vira/self.L**12 + vird/self.L**6
    print("# initial values enep = %8.3f virial = %8.3f" % (enep,virp) )
    print( "     'pas'    'N'    'enep'  'virial'   'acc'")
    tt   = 0
    ept  = 0.
    pres = 0.
    Navg = 0
    Nhist=zeros(nstep, dtype=int32)
    acc = 0.
    for pas in range(nstep) :
        self.t   += 1
        randomchoice = random.random_integers(3, size=(self.Ngc))
        csi = random.random(self.Ngc) 
        dx =  random.random(self.Ngc)    
        dy =  random.random(self.Ngc)  
        dz =  random.random(self.Ngc)
        dx -= rint(dx)
        dy -= rint(dy)
        dz -= rint(dz)
        for i in range(self.Ngc) :
        # advance one step with (on average)  
        # one move per particle and nexc particle insertions/removal
            if randomchoice[i]==1 : 
            # moving particle nt, trial positions
                nt = random.random_integers(N)-1
                rxt = self.rx[nt] + delta*dx[i]
                ryt = self.ry[nt] + delta*dy[i]
                rzt = self.rz[nt] + delta*dz[i]
                rxt -= rint(rxt)
                ryt -= rint(ryt)
                rzt -= rint(rzt)
                # computing energy difference
                (ened,enea,vird,vira) = energymove(self.rx, self.ry, self.rz, self.L, N, rxt, ryt, rzt, nt )
                enediff = enea/self.L**12 + ened/self.L**6
                if csi[i] < exp( -beta*enediff ) :
                    self.rx[nt] = rxt
                    self.ry[nt] = ryt
                    self.rz[nt] = rzt
                    acc += 1
                    enep += enediff
                    virp += (vira/self.L**12 + vird/self.L**6)
            elif  randomchoice[i]==2 :
            # inserting one particle in position xt, yt, zt
                rxt = dx[i]
                ryt = dy[i]
                rzt = dz[i]
                # computing energy difference
                (ened,enea,vird,vira) = energyadd(self.rx, self.ry, self.rz, self.L, N, rxt, ryt, rzt )
                ent = enea/self.L**12 + ened/self.L**6
                if csi[i] < zexc*exp(-beta*ent)/(N+1)  :
                    self.rx[N] = rxt
                    self.ry[N] = ryt
                    self.rz[N] = rzt
                    N += 1
                    acc += 1
                    enep += ent
                    virp += (vira/self.L**12 + vird/self.L**6)
            elif randomchoice[i]==3 :
            # removing particle nt
                nt = random.random_integers(N)-1
                # computing energy difference
                (ened,enea,vird,vira) = energydel(self.rx, self.ry, self.rz, self.L, N, nt )
                ent = enea/self.L**12 + ened/self.L**6
                #print (N, ent, N*exp(beta*ent)/zexc)
                if csi[i] < N*exp(beta*ent)/zexc  :
                    N -= 1
                    self.rx[nt] = self.rx[N]
                    self.ry[nt] = self.ry[N]
                    self.rz[nt] = self.rz[N]
                    acc += 1
                    enep -= ent
                    virp -= (vira/self.L**12 + vird/self.L**6)
            # end of inner cycle
        Nhist[pas]=N
        Navg+= N
        ept += enep
        pres+= virp       
        # computing gdr and single step printout ...
        if (pas+1)%freq==0 : 
            self.calcgdr( N )
            print(" %8d  %5d %9.4f %9.4f  %6.2f" % (self.t, N, enep/N, virp/N, acc*100./(self.t*self.Ngc)) )
            tt += 1
        # end of gc run
    # final configuration
    self.write_input(N, pas, conf_out='conf_in.b')
    self.writexyz( N )
    Ngc = Navg/nstep
    return (tt, N, Ngc, ept, pres, Nhist)

  def read_input(self, N, conf_in='conf_in.b'): 
      import pickle 
      with open(conf_in, 'rb') as ftrj:
         (Nr, pas) = pickle.load(ftrj)
         if N!=Nr :
             print(' ??? reading %d particle from step %d configuration expected %d' % (Nr,pas,N) )
         ( self.rx[:N], self.ry[:N], self.rz[:N], self.L ) = pickle.load( ftrj)
         ( self.px[:N], self.py[:N], self.pz[:N], self.PL) = pickle.load( ftrj)
 
  def write_input(self, N, pas, conf_out='conf_in.b'): 
      import pickle 
      with open(conf_out, 'wb') as ftrj:
          pickle.dump( (N, pas) , ftrj, pickle.HIGHEST_PROTOCOL)
          pickle.dump( ( self.rx[:N], self.ry[:N], self.rz[:N], self.L ), ftrj, pickle.HIGHEST_PROTOCOL)
          pickle.dump( ( self.px[:N], self.py[:N], self.pz[:N], self.PL), ftrj, pickle.HIGHEST_PROTOCOL)

  def writexyz(self, N):
      from numpy import zeros, empty, str_#, savetxt, column_stack
      dx = zeros(N)
      dy = zeros(N)
      dz = zeros(N)
      #ar = empty(N,(str_,2))
      sig=3.4 # in Angstroem for argon   
      rout=open('trajectory.xyz','a')
      rout.write('  %d \n' % N )
      rout.write('\n')
      for i in range(N):
          #ar[i] = "Ar"
          dx[i] = sig*self.rx[i]*self.L
          dy[i] = sig*self.ry[i]*self.L
          dz[i] = sig*self.rz[i]*self.L
          rout.write('Ar   %12.5g   %12.5g   %12.5g\n' % (dx[i],dy[i],dz[i]) )
      rout.close()      

  def write_gdr(self, N, T, rho, gdr_out='gdr.out'):
      """ here L and rho from time averages ? """
      from numpy import zeros, pi, savetxt, column_stack
      V = zeros(self.kg) 
      r = zeros(self.kg)
      g = zeros(self.kg) 
      for lm in range(self.kg) :
          V[lm] = 4./3.*pi*(self.ldel**3)*(3*lm*lm +3*lm + 1); 
          g[lm] = self.gcount[lm]/(V[lm]*(N -1)*T*rho);
          r[lm] = (lm+0.5)*self.ldel
      gout = column_stack( (r, g) )
      savetxt(gdr_out, gout , fmt=('%12.7g ','%12.7g'), header="    'r'     'g(r)'" )          

  def fcc(self, m):
      from numpy import  random, rint
      print( "# number of lattice cells m^3 = %d" % (m**3) )
      a = self.L/m
      print( "# lattice parameter a =%f" % a )
      natom = 4*m**3
      print( "# number of particles  %d " % natom )
      print( "# sides of md-box L = [ %.2f %.2f %.2f ]" % (a*m, a*m, a*m) )
      j  = 0
      xi = 0.
      yi = 0.
      zi = 0.
      delta=0.025
      rrx = random.normal(0., delta, natom)
      rry = random.normal(0., delta, natom)
      rrz = random.normal(0., delta, natom)
      for nx in range(m) :
          for ny in range(m) :
             for nz in range(m) :
                 self.rx[j] = xi + a*nx + rrx[j]
                 self.ry[j] = yi + a*ny + rry[j]             
                 self.rz[j] = zi + a*nz + rrz[j]
                 print( "  %d   %8.3f   %8.3f   %8.3f " % (j, self.rx[j], self.ry[j], self.rz[j]) )
                 # reduced box coordinates in [-0.5:0.5]^3
                 self.rx[j]/= self.L
                 self.ry[j]/= self.L
                 self.rz[j]/= self.L
                 j +=1
                 self.rx[j] = xi + a*nx + rrx[j] + 0.5*a
                 self.ry[j] = yi + a*ny + rry[j] + 0.5*a     
                 self.rz[j] = zi + a*nz + rrz[j]
                 print( "  %d   %8.3f   %8.3f   %8.3f " % (j, self.rx[j], self.ry[j], self.rz[j]) )
                 # reduced box coordinates in [-0.5:0.5]^3
                 self.rx[j]/= self.L
                 self.ry[j]/= self.L
                 self.rz[j]/= self.L
                 j +=1
                 self.rx[j] = xi + a*nx + rrx[j] + 0.5*a
                 self.ry[j] = yi + a*ny + rry[j]             
                 self.rz[j] = zi + a*nz + rrz[j] + 0.5*a
                 print( "  %d   %8.3f   %8.3f   %8.3f " % (j, self.rx[j], self.ry[j], self.rz[j]) )
                 # reduced box coordinates in [-0.5:0.5]^3
                 self.rx[j]/= self.L
                 self.ry[j]/= self.L
                 self.rz[j]/= self.L
                 j +=1
                 self.rx[j] = xi + a*nx + rrx[j] 
                 self.ry[j] = yi + a*ny + rry[j] + 0.5*a            
                 self.rz[j] = zi + a*nz + rrz[j] + 0.5*a
                 print( "  %d   %8.3f   %8.3f    %8.3f " % (j, self.rx[j], self.ry[j], self.rz[j]) )
                 # reduced box coordinates in [-0.5:0.5]^3
                 self.rx[j]/= self.L
                 self.rx[j]-= rint(self.rx[j])
                 self.ry[j]/= self.L
                 self.ry[j]-= rint(self.ry[j])
                 self.rz[j]/= self.L
                 self.rz[j]-= rint(self.rz[j])
                 j +=1
      print( "# end of initial fcc lattice construction")
